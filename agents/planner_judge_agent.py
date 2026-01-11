from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import os
import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.AgentAction import AgentAction
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from agents.prompts import (
    DIFFERENTIAL_TEMPLATE,
    FINAL_DIAGNOSIS_TEMPLATE,
    GUIDELINE_SUMMARY_TEMPLATE,
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
from tools.guidelines_retriever import iter_guideline_docs, build_bm25_retriever
from utils.nlp import calculate_num_tokens, truncate_text


PLAN_HEADER_RE = re.compile(r"^#{2,}\s*plan\s*:?\s*$", re.IGNORECASE)


@lru_cache(maxsize=4)
def _load_guideline_retriever(
    path: str,
    max_lines: Optional[int],
    source_filter: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
):
    docs = list(iter_guideline_docs(path, max_lines, source_filter))
    if not docs:
        return None
    return build_bm25_retriever(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=4,
    )


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
            input_variables=["input", "guideline_context"],
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
            input_variables=[
                "input",
                "plan",
                "evidence",
                "next_action",
                "guideline_context",
            ],
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

        self._guideline_enabled = False
        self._guideline_retriever = None
        self._differential_chain = None
        self._guideline_summary_chain = None
        self._guidelines_top_k = max(1, int(guidelines_top_k))
        self._guidelines_top_n = max(1, int(guidelines_top_n))
        self._guidelines_snippet_tokens = max(50, int(guidelines_snippet_tokens))
        self._guidelines_context_tokens = max(50, int(guidelines_context_tokens))
        self._guidelines_query_tokens = max(50, int(guidelines_query_tokens))
        self._guidelines_source_filter = (
            guidelines_source_filter.strip()
            if guidelines_source_filter and guidelines_source_filter.strip()
            else None
        )
        self._guidelines_max_lines = (
            None
            if guidelines_max_lines is not None and guidelines_max_lines < 0
            else guidelines_max_lines
        )
        self._guidelines_chunk_size = int(guidelines_chunk_size)
        self._guidelines_chunk_overlap = int(guidelines_chunk_overlap)
        self._guidelines_path = (guidelines_path or "").strip()
        self._retrieval_llm = self.planner_llm or self.llm

        if use_guideline_retrieval and self._guidelines_path:
            if os.path.exists(self._guidelines_path):
                self._guideline_retriever = _load_guideline_retriever(
                    self._guidelines_path,
                    self._guidelines_max_lines,
                    self._guidelines_source_filter,
                    self._guidelines_chunk_size,
                    self._guidelines_chunk_overlap,
                )
                self._guideline_enabled = self._guideline_retriever is not None
            else:
                self._guideline_enabled = False

        if self._guideline_enabled:
            diff_prompt = PromptTemplate(
                template=DIFFERENTIAL_TEMPLATE,
                input_variables=["input", "max_differentials"],
                partial_variables={
                    "system_tag_start": planner_tags["system_tag_start"],
                    "system_tag_end": planner_tags["system_tag_end"],
                    "user_tag_start": planner_tags["user_tag_start"],
                    "user_tag_end": planner_tags["user_tag_end"],
                    "ai_tag_start": planner_tags["ai_tag_start"],
                },
            )
            summary_prompt = PromptTemplate(
                template=GUIDELINE_SUMMARY_TEMPLATE,
                input_variables=["differentials", "snippets"],
                partial_variables={
                    "system_tag_start": planner_tags["system_tag_start"],
                    "system_tag_end": planner_tags["system_tag_end"],
                    "user_tag_start": planner_tags["user_tag_start"],
                    "user_tag_end": planner_tags["user_tag_end"],
                    "ai_tag_start": planner_tags["ai_tag_start"],
                },
            )
            self._differential_chain = LLMChain(
                llm=self._retrieval_llm,
                prompt=diff_prompt,
            )
            self._guideline_summary_chain = LLMChain(
                llm=self._retrieval_llm,
                prompt=summary_prompt,
            )

    def run(self, patient_history: str) -> Dict[str, Any]:
        guideline_context = self._build_guideline_context(patient_history, evidence=None)
        plan_raw = self._planner_chain.predict(
            input=patient_history,
            guideline_context=guideline_context,
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
        guideline_context = self._build_guideline_context(
            patient_history,
            evidence=evidence,
        )
        next_action = f"{planned.tool}[{planned.raw_input}]".strip()
        raw = self._judge_chain.predict(
            input=patient_history,
            plan=plan_summary or "None.",
            evidence=evidence,
            guideline_context=guideline_context,
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

    def _build_guideline_context(
        self,
        patient_history: str,
        evidence: Optional[str] = None,
    ) -> str:
        if not self._guideline_enabled:
            return ""
        context_seed = patient_history.strip()
        if evidence:
            context_seed = f"{context_seed}\nEvidence:\n{evidence.strip()}"
        context_seed = self._truncate_text_for_retrieval(context_seed)
        differentials = self._generate_differentials(context_seed)
        if not differentials:
            return ""
        snippets = self._retrieve_guideline_snippets(differentials, context_seed)
        if not snippets:
            return ""
        summary = self._guideline_summary_chain.predict(
            differentials="\n".join(f"- {item}" for item in differentials),
            snippets=snippets,
            stop=[],
            temperature=0.0,
            top_p=1.0,
        )
        summary = (summary or "").strip()
        if not summary:
            return ""
        summary = truncate_text(
            self._retrieval_llm.tokenizer,
            summary,
            self._guidelines_context_tokens,
        )
        ddx_list = "\n".join(f"- {item}" for item in differentials)
        return (
            "Guideline Context:\n"
            f"Top differentials:\n{ddx_list}\n"
            f"Guideline summary:\n{summary}"
        ).strip()

    def _truncate_text_for_retrieval(self, text: str) -> str:
        if not text:
            return ""
        return truncate_text(
            self._retrieval_llm.tokenizer,
            text,
            self._guidelines_query_tokens,
        )

    def _generate_differentials(self, context_text: str) -> List[str]:
        if not self._differential_chain or not context_text:
            return []
        raw = self._differential_chain.predict(
            input=context_text,
            max_differentials=str(self._guidelines_top_n),
            stop=[],
            temperature=self.planner_temperature,
            top_p=self.planner_top_p,
        )
        return self._parse_differentials(raw, self._guidelines_top_n)

    def _parse_differentials(self, text: str, limit: int) -> List[str]:
        items: List[str] = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cleaned = re.sub(r"^[-*\d.()\s]+", "", stripped).strip()
            if cleaned:
                items.append(cleaned)
        if not items and text:
            for chunk in re.split(r"[;,]", text):
                cleaned = chunk.strip()
                if cleaned:
                    items.append(cleaned)
        seen = set()
        deduped: List[str] = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _retrieve_guideline_snippets(
        self,
        differentials: List[str],
        context_seed: str,
    ) -> str:
        if not self._guideline_retriever:
            return ""
        snippets: List[str] = []
        base_context = context_seed.strip()
        for ddx in differentials:
            query = f"{ddx} diagnosis workup labs imaging\n{base_context}"
            self._guideline_retriever.k = self._guidelines_top_k
            docs = self._guideline_retriever.get_relevant_documents(query)
            snippet_text = self._format_guideline_documents(docs)
            if not snippet_text:
                snippet_text = "No relevant guideline snippets retrieved."
            snippets.append(f"DDx: {ddx}\n{snippet_text}")
        return "\n\n".join(snippets).strip()

    def _format_guideline_documents(self, docs: List[Any]) -> str:
        if not docs:
            return ""
        parts: List[str] = []
        for doc in docs:
            content = getattr(doc, "page_content", "") or ""
            content = content.strip()
            if not content:
                continue
            meta = getattr(doc, "metadata", {}) or {}
            source = meta.get("source") or "unknown"
            title = meta.get("title") or "unknown"
            header = f"[{source} | {title}]"
            parts.append(f"{header} {content}")
        if not parts:
            return ""
        combined = "\n".join(parts)
        return truncate_text(
            self._retrieval_llm.tokenizer,
            combined,
            self._guidelines_snippet_tokens,
        )


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
    for tool in tools_agent._tools:
        tool.action_results = patient

    return PlannerJudgeExecutor(tools_agent)
