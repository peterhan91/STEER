"""ReWOO + Planner-Judge Agent.

Two-stage architecture:
1. ReWOO Exploration: Fast plan-execute-solve loop (3 iterations) to discover optimal plan
2. Judge Execution: Careful step-by-step execution with the optimized plan

Faithful reimplementation of ReWOO from MIMIC-Plain, integrated with LangChain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.planner_judge_agent import (
    PlannerJudgeAgent,
    PlanStep,
)


# ReWOO-specific prompts
REWOO_REFLECTION_TEMPLATE = """{system_tag_start}You are reviewing a diagnostic workup attempt. Analyze the evidence and identify gaps.

Your task:
1. SUSPICIOUS FINDINGS: Identify any abnormal or concerning values in labs/imaging that warrant follow-up
2. MISSING TESTS: Note essential tests that were NOT ordered but should have been given the patient presentation
3. FOLLOW-UP NEEDED: Based on current findings, what additional tests would confirm or rule out the suspected diagnosis

Write 3-6 concrete improvement bullets for the next planning attempt.
Be specific: reference exact lab test names, imaging modalities/regions, or diagnostic criteria.
Return only bullet lines starting with '- '.{system_tag_end}{user_tag_start}Prior Plan:
{prior_plan}

Observed Evidence:
{evidence}

Diagnosis Result:
{diagnosis}

Prior Reflections:
{prior_reflections}

Now analyze the evidence above. What suspicious findings need follow-up? What tests are missing?{user_tag_end}{ai_tag_start}"""

REWOO_PLANNER_WITH_REFLECTION_TEMPLATE = """{system_tag_start}You are an experienced clinician. Using your medical knowledge and the patient's presentation, propose a focused and efficient plan for evidence gathering that ensures diagnostic precision and minimizes unnecessary tests.

Return ONLY a sequence of lines using this schema:

Plan: <concise rationale for next step>
#E1 = <Tool>[<Input>]
Plan: <next step>
#E2 = <Tool>[<Input possibly informed by #E1>]
...

Rules:
- Minimize the number of steps, but ensure you gather sufficient information for an accurate and safe final diagnosis.
- Prefer high-yield, patient-condition specific steps.
- Use the exact tool names and input formats described below.
- Do not conclude with a diagnosis here; your output is ONLY the plan.
- Output only Plan/#E lines; do not add headers, bullets, or code fences.

{reflections_block}

Tools:
{tool_descriptions}{system_tag_end}{user_tag_start}Patient History:
{input}

{guideline_context}

Begin! Output only Plan/#E lines as specified. Keep it concise and tailored.
{user_tag_end}{ai_tag_start}"""

REWOO_SOLVER_TEMPLATE = """{system_tag_start}You are a medical artificial intelligence assistant. Based on the patient history and gathered evidence, provide a preliminary diagnosis assessment.

Output format:
Thought: <brief reasoning based on evidence>
Preliminary Diagnosis: <most likely diagnosis>
Confidence: <high/medium/low>
Missing Information: <what additional evidence would help, or "None" if sufficient>{system_tag_end}{user_tag_start}Patient History:
{input}

Evidence:
{evidence}{user_tag_end}{ai_tag_start}Thought:"""


@dataclass
class ReWOOIterationResult:
    """Result from a single ReWOO iteration."""
    iteration: int
    plan_raw: str
    plan_steps: List[PlanStep]
    evidence: str
    diagnosis: str
    executed_steps: List[Dict[str, Any]]


class ReWOOPlannerJudgeAgent(PlannerJudgeAgent):
    """
    Two-stage agent combining ReWOO exploration with Judge execution.

    Stage 1 (ReWOO): Run 3 plan-execute-solve iterations to discover optimal plan
    Stage 2 (Judge): Execute the optimized plan with real-time judge adaptation
    """

    def __init__(
        self,
        *,
        rewoo_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rewoo_iterations = rewoo_iterations

        # Build ReWOO-specific chains
        tags = kwargs.get("tags", {})
        planner_tags = kwargs.get("planner_tags", tags)

        reflection_prompt = PromptTemplate(
            template=REWOO_REFLECTION_TEMPLATE,
            input_variables=["prior_plan", "evidence", "diagnosis", "prior_reflections"],
            partial_variables={
                "system_tag_start": tags.get("system_tag_start", ""),
                "system_tag_end": tags.get("system_tag_end", ""),
                "user_tag_start": tags.get("user_tag_start", ""),
                "user_tag_end": tags.get("user_tag_end", ""),
                "ai_tag_start": tags.get("ai_tag_start", ""),
            },
        )
        self._reflection_chain = LLMChain(llm=self.llm, prompt=reflection_prompt)

        rewoo_planner_prompt = PromptTemplate(
            template=REWOO_PLANNER_WITH_REFLECTION_TEMPLATE,
            input_variables=["input", "guideline_context", "reflections_block"],
            partial_variables={
                "tool_descriptions": self._tool_descriptions(),
                "system_tag_start": planner_tags.get("system_tag_start", ""),
                "system_tag_end": planner_tags.get("system_tag_end", ""),
                "user_tag_start": planner_tags.get("user_tag_start", ""),
                "user_tag_end": planner_tags.get("user_tag_end", ""),
                "ai_tag_start": planner_tags.get("ai_tag_start", ""),
            },
        )
        self._rewoo_planner_chain = LLMChain(llm=self.planner_llm, prompt=rewoo_planner_prompt)

        solver_prompt = PromptTemplate(
            template=REWOO_SOLVER_TEMPLATE,
            input_variables=["input", "evidence"],
            partial_variables={
                "system_tag_start": tags.get("system_tag_start", ""),
                "system_tag_end": tags.get("system_tag_end", ""),
                "user_tag_start": tags.get("user_tag_start", ""),
                "user_tag_end": tags.get("user_tag_end", ""),
                "ai_tag_start": tags.get("ai_tag_start", ""),
            },
        )
        self._solver_chain = LLMChain(llm=self.llm, prompt=solver_prompt)

    def run(self, patient_history: str) -> Dict[str, Any]:
        """
        Two-stage execution:
        1. ReWOO exploration to discover optimal plan
        2. Judge execution with the optimized plan
        """
        # Stage 1: ReWOO Exploration
        optimized_plan, exploration_log = self._rewoo_exploration(patient_history)

        # Stage 2: Judge Execution with optimized plan
        result = self._judge_execution(patient_history, optimized_plan)

        # Add exploration log to result
        result["rewoo_exploration"] = exploration_log
        return result

    def _rewoo_exploration(
        self,
        patient_history: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run ReWOO plan-execute-solve loop for N iterations.
        Returns the best plan and exploration log.
        """
        reflections: List[str] = []
        iteration_results: List[ReWOOIterationResult] = []

        for i in range(self.rewoo_iterations):
            # 1. Plan (with reflections if available)
            plan_raw = self._rewoo_plan(patient_history, reflections)
            plan_text = self._split_planner_sections(plan_raw)
            _, plan_steps = self._parse_planner_output(plan_text)

            if not plan_steps:
                plan_steps = [PlanStep(eid="#E1", tool="Physical Examination", raw_input="")]

            # 2. Execute (batch - all steps at once, like ReWOO)
            executed_steps = self._batch_execute(plan_steps)
            evidence = self._format_evidence(executed_steps)

            # 3. Solve (preliminary diagnosis)
            diagnosis = self._rewoo_solve(patient_history, evidence)

            iteration_results.append(ReWOOIterationResult(
                iteration=i + 1,
                plan_raw=plan_raw,
                plan_steps=plan_steps,
                evidence=evidence,
                diagnosis=diagnosis,
                executed_steps=executed_steps,
            ))

            # 4. Reflect (skip on last iteration)
            if i < self.rewoo_iterations - 1:
                reflection = self._generate_reflection(
                    plan_raw, evidence, diagnosis, reflections
                )
                if reflection:
                    reflections.append(reflection)

        # Use the last iteration's plan as the optimized plan
        # (it incorporates all learnings from previous iterations)
        best_result = iteration_results[-1]
        optimized_plan = best_result.plan_raw

        exploration_log = {
            "iterations": [
                {
                    "iteration": r.iteration,
                    "plan": r.plan_raw,
                    "evidence": r.evidence,
                    "diagnosis": r.diagnosis,
                }
                for r in iteration_results
            ],
            "reflections": reflections,
            "final_plan": optimized_plan,
        }

        return optimized_plan, exploration_log

    def _rewoo_plan(
        self,
        patient_history: str,
        reflections: List[str],
    ) -> str:
        """Generate plan with optional reflections from prior iterations."""
        guideline_context = self._build_guideline_context(patient_history, evidence=None)

        # Build reflections block
        reflections_block = ""
        if reflections:
            lines = [
                "IMPORTANT - Learnings from prior attempts (you MUST address these in your plan):",
                "The following issues were identified - ensure your plan includes tests to address them:",
            ]
            for r in reflections:
                r_stripped = r.strip()
                if r_stripped and not r_stripped.startswith("-"):
                    r_stripped = "- " + r_stripped
                if r_stripped:
                    lines.append(r_stripped)
            reflections_block = "\n".join(lines)

        plan_raw = self._rewoo_planner_chain.predict(
            input=patient_history,
            guideline_context=guideline_context,
            reflections_block=reflections_block,
            stop=self.planner_stop_words,
            temperature=self.planner_temperature,
            top_p=self.planner_top_p,
        )
        return plan_raw or ""

    def _batch_execute(self, plan_steps: List[PlanStep]) -> List[Dict[str, Any]]:
        """Execute all plan steps in batch (ReWOO-style)."""
        executed_steps: List[Dict[str, Any]] = []
        evidences: Dict[str, str] = {}

        for step in plan_steps:
            resolved_input = step.raw_input
            for var in re.findall(r"#E\d+", resolved_input or ""):
                if var in evidences:
                    resolved_input = resolved_input.replace(var, evidences[var])
            parsed = self._parse_tool_input(step.tool, resolved_input)
            if parsed is None:
                continue

            obs = self._run_tool(step.tool, parsed)
            input_pretty = self._format_action_input(step.tool, parsed)

            executed_steps.append({
                "eid": step.eid,
                "tool": step.tool,
                "raw_input": step.raw_input,
                "parsed_input": parsed,
                "input_pretty": input_pretty,
                "observation": obs,
            })
            evidences[step.eid] = obs

        return executed_steps

    def _rewoo_solve(self, patient_history: str, evidence: str) -> str:
        """Generate preliminary diagnosis based on gathered evidence."""
        diagnosis = self._solver_chain.predict(
            input=patient_history,
            evidence=evidence,
            stop=self.judge_stop_words,
            temperature=self.judge_temperature,
            top_p=1.0,
        )
        return diagnosis or ""

    def _generate_reflection(
        self,
        prior_plan: str,
        evidence: str,
        diagnosis: str,
        prior_reflections: List[str],
    ) -> str:
        """Generate actionable reflection for next iteration."""
        prior_reflections_text = "\n".join(prior_reflections) if prior_reflections else "None."

        raw = self._reflection_chain.predict(
            prior_plan=prior_plan,
            evidence=evidence,
            diagnosis=diagnosis,
            prior_reflections=prior_reflections_text,
            stop=[],
            temperature=0.3,
            top_p=0.95,
        )

        # Keep only bullet lines
        bullets = []
        for line in (raw or "").splitlines():
            s = line.strip()
            if s.startswith("- "):
                bullets.append(s)

        return "\n".join(bullets) if bullets else raw.strip()

    def _judge_execution(
        self,
        patient_history: str,
        optimized_plan: str,
    ) -> Dict[str, Any]:
        """Execute with Judge loop using the ReWOO-optimized plan."""
        # Parse the optimized plan (skip planner call)
        plan_text = self._split_planner_sections(optimized_plan)
        plan_lines, plan_steps = self._parse_planner_output(plan_text)

        if not plan_steps:
            plan_steps = [PlanStep(eid="#E1", tool="Physical Examination", raw_input="")]

        plan_summary = self._format_plan(plan_lines, plan_steps)

        executed_steps: List[Dict[str, Any]] = []
        intermediate_steps: List[Tuple[Any, str]] = []
        judge_log: List[Dict[str, str]] = []

        step_idx = 0
        plan_idx = 0

        while step_idx < self.max_steps and plan_idx < len(plan_steps):
            planned = plan_steps[plan_idx]

            # First step executes directly, subsequent steps go through judge
            if not executed_steps:
                selected = planned
                plan_idx += 1
            else:
                decision = self._judge_next_action(
                    patient_history,
                    plan_summary,
                    executed_steps,
                    planned,
                )
                judge_log.append({
                    "decision": decision.decision,
                    "action": decision.tool or "",
                    "action_input": decision.tool_input or "",
                    "rationale": decision.rationale or "",
                })

                if decision.decision == "stop":
                    break
                if decision.decision == "skip":
                    plan_idx += 1
                    continue
                if decision.decision in {"modify", "add"} and decision.tool:
                    candidate = PlanStep(
                        eid=planned.eid if decision.decision == "modify" else "judge",
                        tool=decision.tool,
                        raw_input=decision.tool_input or "",
                    )
                    if decision.decision == "modify":
                        plan_idx += 1
                else:
                    candidate = planned
                    plan_idx += 1
                selected = candidate

            parsed = self._parse_tool_input(selected.tool, selected.raw_input)
            if parsed is None:
                step_idx += 1
                continue

            obs = self._run_tool(selected.tool, parsed)
            input_pretty = self._format_action_input(selected.tool, parsed)

            executed_steps.append({
                "tool": selected.tool,
                "raw_input": selected.raw_input,
                "parsed_input": parsed,
                "input_pretty": input_pretty,
                "observation": obs,
            })

            from agents.AgentAction import AgentAction
            intermediate_steps.append((
                AgentAction(
                    tool=selected.tool,
                    tool_input={"action_input": parsed},
                    log=f"Action: {selected.tool}\nAction Input: {selected.raw_input}",
                    custom_parsings=0,
                ),
                obs,
            ))
            step_idx += 1

        # Final diagnosis
        evidence_text = self._format_evidence(executed_steps)
        evidence_text = self._truncate_for_context(
            evidence_text,
            self.llm.tokenizer,
            self.max_context_length,
            patient_history,
        )

        final = self._final_chain.predict(
            input=patient_history,
            evidence=evidence_text,
            stop=self.judge_stop_words,
            temperature=self.judge_temperature,
            top_p=1.0,
        )

        return {
            "output": final,
            "intermediate_steps": intermediate_steps,
            "planner_raw": optimized_plan,
            "planner_plan": plan_text,
            "planner_summary": plan_summary,
            "judge_log": judge_log,
        }


class ReWOOPlannerJudgeExecutor:
    """Executor wrapper for ReWOOPlannerJudgeAgent."""

    def __init__(self, agent: ReWOOPlannerJudgeAgent):
        self.agent = agent

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        patient_history = inputs.get("input", "")
        return self.agent.run(patient_history)


def build_agent_executor_ReWOOPlannerJudge(
    *,
    patient: Dict[str, Any],
    llm,
    planner_llm,
    lab_test_mapping_path: str,
    max_context_length: int,
    tags: Dict[str, str],
    planner_tags: Dict[str, str],
    include_ref_range: bool,
    bin_lab_results: bool,
    provide_diagnostic_criteria: bool,
    planner_stop_words: List[str],
    judge_stop_words: List[str],
    planner_temperature: float,
    planner_top_p: float,
    judge_temperature: float,
    max_steps: int,
    rewoo_iterations: int = 3,
    use_guideline_retrieval: bool = False,
    guidelines_path: Optional[str] = None,
    guidelines_max_lines: Optional[int] = 2000,
    guidelines_source_filter: Optional[str] = None,
    guidelines_chunk_size: int = 1200,
    guidelines_chunk_overlap: int = 150,
    guidelines_top_k: int = 4,
    guidelines_top_n: int = 5,
    guidelines_snippet_tokens: int = 400,
    guidelines_context_tokens: int = 600,
    guidelines_query_tokens: int = 300,
) -> ReWOOPlannerJudgeExecutor:
    """Build ReWOO + Planner-Judge agent executor."""
    from utils.pickle_compat import safe_pickle_load

    with open(lab_test_mapping_path, "rb") as handle:
        lab_test_mapping_df = safe_pickle_load(handle)

    agent = ReWOOPlannerJudgeAgent(
        llm=llm,
        planner_llm=planner_llm,
        lab_test_mapping_df=lab_test_mapping_df,
        include_ref_range=include_ref_range,
        bin_lab_results=bin_lab_results,
        provide_diagnostic_criteria=provide_diagnostic_criteria,
        tags=tags,
        planner_tags=planner_tags,
        max_context_length=max_context_length,
        planner_stop_words=planner_stop_words,
        judge_stop_words=judge_stop_words,
        planner_temperature=planner_temperature,
        planner_top_p=planner_top_p,
        judge_temperature=judge_temperature,
        max_steps=max_steps,
        rewoo_iterations=rewoo_iterations,
        use_guideline_retrieval=use_guideline_retrieval,
        guidelines_path=guidelines_path,
        guidelines_max_lines=guidelines_max_lines,
        guidelines_source_filter=guidelines_source_filter,
        guidelines_chunk_size=guidelines_chunk_size,
        guidelines_chunk_overlap=guidelines_chunk_overlap,
        guidelines_top_k=guidelines_top_k,
        guidelines_top_n=guidelines_top_n,
        guidelines_snippet_tokens=guidelines_snippet_tokens,
        guidelines_context_tokens=guidelines_context_tokens,
        guidelines_query_tokens=guidelines_query_tokens,
    )

    # Inject patient results into tool handlers
    for tool in agent._tools:
        tool.action_results = patient

    return ReWOOPlannerJudgeExecutor(agent)
