from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.AgentAction import AgentAction
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from agents.prompts import (
    FINAL_DIAGNOSIS_TEMPLATE,
    JUDGE_TEMPLATE,
    PLANNER_TEMPLATE,
)
from tools.Tools import (
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
    RunImaging,
    RunLaboratoryTests,
)
from tools.utils import (
    UNIQUE_MODALITY_TO_ORGAN_MAPPING,
    action_input_pretty_printer,
    count_radiology_modality_and_organ_matches,
)
from utils.nlp import calculate_num_tokens, truncate_text


PLAN_HEADER_RE = re.compile(r"^#{2,}\s*plan\s*:?\s*$", re.IGNORECASE)

@dataclass
class PlanStep:
    eid: str
    tool: str
    raw_input: str


@dataclass
class JudgeDecision:
    decision: str
    tool: Optional[str]
    tool_input: Optional[str]
    rationale: str


class PlannerJudgeAgent:
    def __init__(
        self,
        *,
        llm,
        planner_llm,
        lab_test_mapping_df,
        include_ref_range: bool,
        bin_lab_results: bool,
        provide_diagnostic_criteria: bool,
        tags: Dict[str, str],
        planner_tags: Dict[str, str],
        max_context_length: int,
        planner_stop_words: List[str],
        judge_stop_words: List[str],
        planner_temperature: float,
        planner_top_p: float,
        judge_temperature: float,
        max_steps: int,
    ):
        self.llm = llm
        self.planner_llm = planner_llm
        self.lab_test_mapping_df = lab_test_mapping_df
        self.include_ref_range = include_ref_range
        self.bin_lab_results = bin_lab_results
        self.provide_diagnostic_criteria = bool(provide_diagnostic_criteria)
        self.max_context_length = max_context_length
        self.max_steps = max(1, int(max_steps))
        self.planner_stop_words = planner_stop_words or []
        self.judge_stop_words = judge_stop_words or []
        self.planner_temperature = planner_temperature
        self.planner_top_p = planner_top_p
        self.judge_temperature = judge_temperature

        self._lab_parser = DiagnosisWorkflowParser(
            lab_test_mapping_df=self.lab_test_mapping_df
        )

        self._tools = self._build_tools()
        self._tool_map = {tool.name: tool for tool in self._tools}
        self._tool_names = list(self._tool_map.keys())

        planner_prompt = PromptTemplate(
            template=PLANNER_TEMPLATE,
            input_variables=["input"],
            partial_variables={
                "tool_descriptions": self._tool_descriptions(),
                "system_tag_start": planner_tags["system_tag_start"],
                "system_tag_end": planner_tags["system_tag_end"],
                "user_tag_start": planner_tags["user_tag_start"],
                "user_tag_end": planner_tags["user_tag_end"],
                "ai_tag_start": planner_tags["ai_tag_start"],
            },
        )
        judge_prompt = PromptTemplate(
            template=JUDGE_TEMPLATE,
            input_variables=["input", "plan", "evidence", "next_action"],
            partial_variables={
                "tool_names": ", ".join(self._tool_names),
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
            },
        )
        final_prompt = PromptTemplate(
            template=FINAL_DIAGNOSIS_TEMPLATE,
            input_variables=["input", "evidence"],
            partial_variables={
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
            },
        )

        self._planner_chain = LLMChain(llm=self.planner_llm, prompt=planner_prompt)
        self._judge_chain = LLMChain(llm=self.llm, prompt=judge_prompt)
        self._final_chain = LLMChain(llm=self.llm, prompt=final_prompt)

    def run(self, patient_history: str) -> Dict[str, Any]:
        plan_raw = self._planner_chain.predict(
            input=patient_history,
            stop=self.planner_stop_words,
            temperature=self.planner_temperature,
            top_p=self.planner_top_p,
        )
        plan_text = self._split_planner_sections(plan_raw or "")
        plan_lines, plan_steps = self._parse_planner_output(plan_text)
        if not plan_steps:
            plan_steps = [PlanStep(eid="#E1", tool="Physical Examination", raw_input="")]
        plan_summary = self._format_plan(plan_lines, plan_steps)

        executed_steps: List[Dict[str, Any]] = []
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        judge_log: List[Dict[str, str]] = []

        step_idx = 0
        plan_idx = 0
        while step_idx < self.max_steps and plan_idx < len(plan_steps):
            planned = plan_steps[plan_idx]
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
                judge_log.append(
                    {
                        "decision": decision.decision,
                        "action": decision.tool or "",
                        "action_input": decision.tool_input or "",
                        "rationale": decision.rationale or "",
                    }
                )
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
            executed_steps.append(
                {
                    "tool": selected.tool,
                    "raw_input": selected.raw_input,
                    "parsed_input": parsed,
                    "input_pretty": input_pretty,
                    "observation": obs,
                }
            )
            intermediate_steps.append(
                (
                    AgentAction(
                        tool=selected.tool,
                        tool_input={"action_input": parsed},
                        log=f"Action: {selected.tool}\nAction Input: {selected.raw_input}",
                        custom_parsings=0,
                    ),
                    obs,
                )
            )
            step_idx += 1

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
            "planner_raw": plan_raw,
            "planner_plan": plan_text,
            "planner_summary": plan_summary,
            "judge_log": judge_log,
        }

    def _build_tools(self) -> List[Any]:
        tools = [
            DoPhysicalExamination(action_results={}),
            RunLaboratoryTests(
                action_results={},
                lab_test_mapping_df=self.lab_test_mapping_df,
                include_ref_range=self.include_ref_range,
                bin_lab_results=self.bin_lab_results,
            ),
            RunImaging(action_results={}),
        ]
        if self.provide_diagnostic_criteria:
            tools.append(ReadDiagnosticCriteria())
        return tools

    def _tool_descriptions(self) -> str:
        lines = [
            "Physical Examination[input]: Perform physical examination; input can be left blank.",
            "Laboratory Tests[input]: Order specific tests; provide a comma-separated list.",
            "Imaging[input]: Do specific imaging; input must specify region and modality.",
            "  - Imaging Regions: Abdomen, Chest, Head, Neck, Pelvis",
            "  - Imaging Modalities: Ultrasound, CT, MRI, Radiograph.",
        ]
        if self.provide_diagnostic_criteria:
            lines.append(
                "Diagnostic Criteria[input]: Examine diagnostic criteria for a pathology."
            )
        return "\n".join(lines)

    def _split_planner_sections(self, text: str) -> str:
        lines = str(text or "").splitlines()
        plan_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if PLAN_HEADER_RE.match(stripped):
                plan_idx = i
                break
        if plan_idx is None:
            return str(text or "").strip()
        return "\n".join(lines[plan_idx + 1 :]).strip()

    def _parse_planner_output(self, text: str) -> Tuple[List[str], List[PlanStep]]:
        plan_lines: List[str] = []
        steps: List[PlanStep] = []
        s = self._strip_special_blocks(text)
        for line in s.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            normalized = re.sub(r"^[-*\d.()\s]+", "", stripped).strip()
            if normalized.lower().startswith("plan:"):
                plan_lines.append(normalized)
                continue
            step = self._parse_plan_step_line(normalized)
            if step:
                steps.append(step)
        steps.sort(key=lambda step: int(re.sub(r"[^0-9]", "", step.eid) or 0))
        return plan_lines, steps

    def _parse_plan_step_line(self, line: str) -> Optional[PlanStep]:
        match = re.match(r"^#?E(\d+)\s*[:=\-]\s*(.+)$", line)
        if not match:
            return None
        eid = f"#E{match.group(1)}"
        tool_call = match.group(2).strip()
        tool, arg = self._extract_tool_and_input(tool_call)
        tool = self._normalize_tool_name(tool)
        return PlanStep(eid=eid, tool=tool, raw_input=arg or "")

    def _strip_special_blocks(self, text: str) -> str:
        s = str(text or "")
        s = re.sub(r"<think>.*?</think>", "\n", s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r"<tool_call[^>]*>.*?</tool_call>", "\n", s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r"```.*?```", "\n", s, flags=re.DOTALL)
        s = s.replace("```", "\n")
        s = s.replace("<|im_start|>", "\n").replace("<|im_end|>", "\n")
        return s

    def _extract_tool_and_input(self, raw: str) -> Tuple[str, Optional[str]]:
        s = (raw or "").strip()
        if "[" in s and s.endswith("]"):
            tool, rest = s.split("[", 1)
            return tool.strip(), rest[:-1].strip()
        match = re.match(r"^(.*?)\s*(?:\:\s+|\-\s+)(.+)$", s)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return s, None

    def _normalize_tool_name(self, name: str) -> str:
        low = str(name or "").strip().lower()
        if "physical" in low:
            return "Physical Examination"
        if "lab" in low:
            return "Laboratory Tests"
        if "diagnostic criteria" in low:
            return "Diagnostic Criteria"
        if "imaging" in low or any(
            mod in low for mod in ["ct", "mri", "ultrasound", "radiograph", "x-ray", "xray"]
        ):
            return "Imaging"
        return name.strip()

    def _parse_tool_input(self, tool: str, raw_input: str) -> Optional[Any]:
        if tool == "Physical Examination":
            return ""
        if tool == "Laboratory Tests":
            raw = (raw_input or "").strip()
            if not raw:
                return None
            self._lab_parser.action_input = raw
            self._lab_parser.parse_lab_tests_action_input()
            return self._lab_parser.action_input
        if tool == "Imaging":
            raw = (raw_input or "").strip()
            if not raw:
                return None
            modality, modality_count, region, region_count = (
                count_radiology_modality_and_organ_matches(raw)
            )
            if region_count == 0 and modality in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
                region = UNIQUE_MODALITY_TO_ORGAN_MAPPING[modality]
                region_count = 1
            if region_count == 0 or modality_count == 0:
                return None
            return {"modality": modality, "region": region}
        if tool == "Diagnostic Criteria":
            return (raw_input or "").strip()
        return None

    def _run_tool(self, tool: str, parsed_input: Any) -> str:
        handler = self._tool_map.get(tool)
        if not handler:
            return f"Tool Error: {tool} is not available."
        if tool == "Physical Examination":
            return handler._run("")
        return handler._run(parsed_input)

    def _format_action_input(self, tool: str, parsed_input: Any) -> str:
        if tool == "Laboratory Tests":
            return action_input_pretty_printer(parsed_input, self.lab_test_mapping_df)
        if tool == "Imaging":
            return action_input_pretty_printer(parsed_input, self.lab_test_mapping_df)
        if tool == "Diagnostic Criteria":
            return str(parsed_input or "")
        return ""

    def _format_evidence(self, steps: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, step in enumerate(steps, 1):
            input_text = step.get("input_pretty") or step.get("raw_input") or ""
            if input_text:
                lines.append(f"Step {idx}: {step['tool']}[{input_text}]")
            else:
                lines.append(f"Step {idx}: {step['tool']}")
            lines.append(step["observation"])
        return "\n".join(lines).strip() or "None."

    def _format_plan(self, plan_lines: List[str], steps: List[PlanStep]) -> str:
        lines: List[str] = []
        if plan_lines:
            lines.extend(plan_lines)
        for step in steps:
            if step.raw_input:
                lines.append(f"{step.eid} = {step.tool}[{step.raw_input}]")
            else:
                lines.append(f"{step.eid} = {step.tool}[]")
        return "\n".join(lines).strip()

    def _truncate_for_context(
        self,
        evidence: str,
        tokenizer,
        max_context_length: int,
        patient_history: str,
    ) -> str:
        prompt_tokens = calculate_num_tokens(
            tokenizer,
            [patient_history, evidence],
        )
        if prompt_tokens < max_context_length - 200:
            return evidence
        available_tokens = max(50, max_context_length - 200)
        return truncate_text(tokenizer, evidence, available_tokens)

    def _judge_next_action(
        self,
        patient_history: str,
        plan_summary: str,
        executed_steps: List[Dict[str, Any]],
        planned: PlanStep,
    ) -> JudgeDecision:
        evidence = self._format_evidence(executed_steps)
        next_action = f"{planned.tool}[{planned.raw_input}]".strip()
        raw = self._judge_chain.predict(
            input=patient_history,
            plan=plan_summary or "None.",
            evidence=evidence,
            next_action=next_action,
            stop=self.judge_stop_words,
            temperature=self.judge_temperature,
            top_p=1.0,
        )
        return self._parse_judge_output(raw)

    def _parse_judge_output(self, text: str) -> JudgeDecision:
        decision = self._extract_between(text, "Decision:", ["Action:", "Action Input:", "Rationale:"])
        action = self._extract_between(text, "Action:", ["Action Input:", "Rationale:"])
        action_input = self._extract_between(text, "Action Input:", ["Rationale:"])
        rationale = self._extract_between(text, "Rationale:")

        decision_norm = (decision or "").strip().lower()
        if decision_norm not in {"proceed", "skip", "modify", "add", "stop"}:
            decision_norm = "proceed"

        action_norm = self._normalize_tool_name(action) if action else None
        if action_norm not in self._tool_map:
            action_norm = None

        return JudgeDecision(
            decision=decision_norm,
            tool=action_norm,
            tool_input=(action_input or "").strip() if action_input else None,
            rationale=(rationale or "").strip(),
        )

    def _extract_between(
        self, text: str, start_key: str, end_keys: Optional[List[str]] = None
    ) -> str:
        if not text:
            return ""
        lower = text.lower()
        start = lower.rfind(start_key.lower())
        if start == -1:
            return ""
        start += len(start_key)
        end = len(text)
        if end_keys:
            for key in end_keys:
                idx = lower.find(key.lower(), start)
                if idx != -1:
                    end = min(end, idx)
        return text[start:end].strip().strip("\n\r :-")


class PlannerJudgeExecutor:
    def __init__(self, agent: PlannerJudgeAgent):
        self.agent = agent

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        patient_history = inputs.get("input", "")
        return self.agent.run(patient_history)


def build_agent_executor_PlannerJudge(
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
) -> PlannerJudgeExecutor:
    from utils.pickle_compat import safe_pickle_load

    with open(lab_test_mapping_path, "rb") as handle:
        lab_test_mapping_df = safe_pickle_load(handle)

    # Inject patient results into tool handlers
    tools_agent = PlannerJudgeAgent(
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
    )
    for tool in tools_agent._tools:
        tool.action_results = patient

    return PlannerJudgeExecutor(tools_agent)
