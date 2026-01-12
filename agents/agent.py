import pickle
import os
import re
from utils.pickle_compat import safe_pickle_load
from typing import List, Tuple, Union, Dict, Any, Optional
from hashlib import sha256
import pandas as pd
from functools import lru_cache
from loguru import logger

from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents.mrkl.base import ZeroShotAgent
from pydantic.v1 import PrivateAttr
from langchain.schema.messages import BaseMessage
from langchain.schema import AgentAction
from langchain.callbacks import FileCallbackHandler


from agents.prompts import (
    CHAT_TEMPLATE,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    TOOL_USE_EXAMPLES,
    DIAG_CRIT_TOOL_USE_EXAMPLE,
    DIFFERENTIAL_TEMPLATE,
    GUIDELINE_SUMMARY_TEMPLATE,
)
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from tools.Tools import (
    RunLaboratoryTests,
    RunImaging,
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
)
from tools.utils import action_input_pretty_printer
from tools.guidelines_retriever import iter_guideline_docs, build_bm25_retriever
from utils.nlp import calculate_num_tokens, truncate_text

STOP_WORDS = ["Observation:", "Observations:", "observation:", "observations:"]


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


class TextSummaryCache:
    def __init__(self):
        self.cache = {}

    def hash_text(self, text):
        return sha256(text.encode()).hexdigest()

    def add_summary(self, text, summary):
        text_hash = self.hash_text(text)
        if text_hash in self.cache:
            return
        self.cache[text_hash] = summary

    def get_summary(self, text):
        text_hash = self.hash_text(text)
        return self.cache.get(text_hash, None)


class CustomZeroShotAgent(ZeroShotAgent):
    lab_test_mapping_df: pd.DataFrame = None
    observation_summary_cache: TextSummaryCache = TextSummaryCache()
    stop: List[str]
    max_context_length: int
    tags: Dict[str, str]
    summarize: bool
    use_guideline_retrieval: bool = False
    guidelines_path: str = ""
    guidelines_max_lines: Optional[int] = 2000
    guidelines_source_filter: Optional[str] = None
    guidelines_chunk_size: int = 1200
    guidelines_chunk_overlap: int = 150
    guidelines_top_k: int = 4
    guidelines_top_n: int = 5
    guidelines_snippet_tokens: int = 400
    guidelines_context_tokens: int = 600
    guidelines_query_tokens: int = 300
    guideline_temperature: float = 0.2
    guideline_top_p: float = 0.95
    _guideline_enabled: bool = PrivateAttr(default=False)
    _guideline_retriever: Any = PrivateAttr(default=None)
    _differential_chain: Any = PrivateAttr(default=None)
    _guideline_summary_chain: Any = PrivateAttr(default=None)
    _guideline_cache: Dict[str, str] = PrivateAttr(default=None)
    _last_guideline_context: str = PrivateAttr(default="")

    class Config:
        arbitrary_types_allowed = True

    # Allow for multiple stop criteria instead of just taking the observation prefix string
    @property
    def _stop(self) -> List[str]:
        return self.stop

    # Need to override to pass input so that we can calculate the number of tokes
    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        original_input = kwargs.get("input", "")
        guideline_context = self._build_guideline_context(
            original_input, intermediate_steps
        )
        kwargs["guideline_context"] = guideline_context
        thoughts, kwargs = self._construct_scratchpad(intermediate_steps, **kwargs)
        if kwargs.get("input", "") != original_input:
            guideline_context = self._build_guideline_context(
                kwargs.get("input", ""), intermediate_steps
            )
            kwargs["guideline_context"] = guideline_context
        new_inputs = {
            "agent_scratchpad": thoughts,
            "stop": self._stop,
            "guideline_context": kwargs.get("guideline_context", ""),
        }
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    # Construct the running thoughts and observations of the model. Summarize the convo if we hit our token limit
    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"{self.tags['ai_tag_end']}{self.tags['user_tag_start']}{self.observation_prefix}{observation.strip()}{self.tags['user_tag_end']}{self.tags['ai_tag_start']}{self.llm_prefix}"
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad=thoughts,
                        guideline_context=kwargs.get("guideline_context", ""),
                    )
                ],
            )
            >= self.max_context_length - 100
        ) and self.summarize:
            thoughts = self._summarize_steps(intermediate_steps)

        # Worst worst case, we are still over or close to the limit even after summarizing and thus should truncate and force a diagnosis
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad=thoughts,
                        guideline_context=kwargs.get("guideline_context", ""),
                    )
                ],
            )
            >= self.max_context_length - 100
        ):
            prompt_and_input_tokens = calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad="",
                        guideline_context=kwargs.get("guideline_context", ""),
                    )
                ],
            )
            # Could be that input is already over limit and we need to truncate input
            if prompt_and_input_tokens > self.max_context_length - 100:
                prompt_tokens = calculate_num_tokens(
                    self.llm_chain.llm.tokenizer,
                    [
                        self.llm_chain.prompt.format(
                            input="",
                            agent_scratchpad="",
                            guideline_context=kwargs.get("guideline_context", ""),
                        )
                    ],
                )
                kwargs["input"] = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    kwargs["input"],
                    self.max_context_length - prompt_tokens - 200,
                )
                thoughts = ""
            else:
                thoughts = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    thoughts,
                    self.max_context_length - prompt_and_input_tokens - 100,
                )  # give yourself 100 tokens for diagnosis and treatment and tags
            thoughts += f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}Final'

        # Also return kwargs so if we edited input, the change is propagated
        return " " + thoughts.strip(), kwargs

    # Takes all tool requests and observations and summarizes them one-by-one
    def _summarize_steps(self, intermediate_steps):
        prompt = PromptTemplate(
            template=SUMMARIZE_OBSERVATION_TEMPLATE,
            input_variables=["observation"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "user_tag_start": self.tags["user_tag_start"],
                "user_tag_end": self.tags["user_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)
        summaries = []
        summaries.append("A summary of information I know thus far:")
        for indx, (action, observation) in enumerate(intermediate_steps):
            # Only summarize valid actions
            if action.tool in self.allowed_tools:
                # Keep format as in instruction to re-enforce schema
                summaries.append("Action: " + action.tool)
                if action.tool in [
                    "Laboratory Tests",
                    "Imaging",
                    "Diagnostic Criteria",
                ]:
                    summaries.append(
                        "Action Input: "
                        + action_input_pretty_printer(
                            action.tool_input["action_input"], self.lab_test_mapping_df
                        )
                    )
                # Check cache to not re-summarize same observation
                summary = self.observation_summary_cache.get_summary(observation)
                if not summary:
                    # Summary of each step should be minimal and should not exceed max_context_length
                    prompt_tokens = calculate_num_tokens(
                        self.llm_chain.llm.tokenizer,
                        [
                            prompt.format(observation=""),
                        ],
                    )

                    observation = truncate_text(
                        self.llm_chain.llm.tokenizer,
                        observation,
                        self.max_context_length
                        - prompt_tokens
                        - 100,  # Gives a max of 100 tokens to generate for the summary if we are near context length limit. Usually only used when model does really weird infinite generations of action inputs and doesnt hit a stop token so shouldnt be much actual info to summarize anyway
                    )
                    summary = chain.predict(observation=observation, stop=[])
                    # Add to cache
                    self.observation_summary_cache.add_summary(observation, summary)
                summaries.append("Observation: " + summary)
            else:
                # Include invalid requests in summary to not run into infinite loop of same invalid tool being ordered
                invalid_request = action.log
                # Condense invalid request to the action and everything afterwards. Can remove thinking
                if "action:" in action.log.lower():
                    invalid_request = action.log[action.log.lower().index("action:") :]
                summaries.append(
                    f"I tried '{invalid_request}', but it was an invalid request."
                )
                # If invalid tool was final request, remind of valid tools and diagnosis option. Add string to last summary because we dont want to force newlines that the prompt templates maybe do not want
                if indx == len(intermediate_steps) - 1:
                    summaries[-1] = summaries[-1] + (
                        f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Please choose a valid tool from {self.allowed_tools} or provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}{self.llm_prefix}'
                    )
                    return "\n".join(summaries)
        summaries.append(self.llm_prefix)
        return "\n".join(summaries)

    def configure_guideline_retrieval(self) -> None:
        self._guideline_enabled = False
        self._guideline_retriever = None
        self._differential_chain = None
        self._guideline_summary_chain = None
        self._guideline_cache = {}
        logger.info(f"Configuring guideline retrieval: use_guideline_retrieval={self.use_guideline_retrieval}, guidelines_path={self.guidelines_path}")
        if not self.use_guideline_retrieval:
            logger.info("Guideline retrieval disabled: use_guideline_retrieval is False")
            return
        if not self.guidelines_path:
            logger.warning("Guideline retrieval disabled: guidelines_path is empty")
            return
        if not os.path.exists(self.guidelines_path):
            logger.warning(f"Guideline retrieval disabled: path does not exist: {self.guidelines_path}")
            return
        max_lines = (
            None
            if self.guidelines_max_lines is not None and self.guidelines_max_lines < 0
            else self.guidelines_max_lines
        )
        source_filter = (
            self.guidelines_source_filter.strip()
            if self.guidelines_source_filter and self.guidelines_source_filter.strip()
            else None
        )
        retriever = _load_guideline_retriever(
            self.guidelines_path,
            max_lines,
            source_filter,
            self.guidelines_chunk_size,
            self.guidelines_chunk_overlap,
        )
        if not retriever:
            logger.warning("Guideline retrieval disabled: failed to load retriever")
            return
        self._guideline_retriever = retriever
        self._guideline_enabled = True
        logger.info(f"Guideline retrieval ENABLED with retriever from {self.guidelines_path}")
        diff_prompt = PromptTemplate(
            template=DIFFERENTIAL_TEMPLATE,
            input_variables=["input", "max_differentials"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "user_tag_start": self.tags["user_tag_start"],
                "user_tag_end": self.tags["user_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        summary_prompt = PromptTemplate(
            template=GUIDELINE_SUMMARY_TEMPLATE,
            input_variables=["differentials", "snippets"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "user_tag_start": self.tags["user_tag_start"],
                "user_tag_end": self.tags["user_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        self._differential_chain = LLMChain(llm=self.llm_chain.llm, prompt=diff_prompt)
        self._guideline_summary_chain = LLMChain(
            llm=self.llm_chain.llm, prompt=summary_prompt
        )

    def _build_guideline_context(
        self, patient_history: str, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        if not self._guideline_enabled:
            logger.debug("_build_guideline_context: guideline not enabled, returning empty")
            self._last_guideline_context = ""
            return ""
        logger.debug("_build_guideline_context: building context...")
        evidence = self._format_evidence(intermediate_steps)
        context_seed = patient_history.strip()
        if evidence:
            context_seed = f"{context_seed}\nEvidence:\n{evidence.strip()}"
        context_seed = self._truncate_text_for_retrieval(context_seed)
        if not context_seed:
            self._last_guideline_context = ""
            return ""
        cache_key = sha256(context_seed.encode()).hexdigest()
        cached = self._guideline_cache.get(cache_key)
        if cached is not None:
            self._last_guideline_context = cached
            return cached
        differentials = self._generate_differentials(context_seed)
        if not differentials:
            self._last_guideline_context = ""
            return ""
        snippets = self._retrieve_guideline_snippets(differentials, context_seed)
        if not snippets:
            self._last_guideline_context = ""
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
            self._last_guideline_context = ""
            return ""
        summary = truncate_text(
            self.llm_chain.llm.tokenizer,
            summary,
            self.guidelines_context_tokens,
        )
        ddx_list = "\n".join(f"- {item}" for item in differentials)
        context = (
            "Guideline Context:\n"
            f"Top differentials:\n{ddx_list}\n"
            f"Guideline summary:\n{summary}"
        ).strip()
        self._guideline_cache[cache_key] = context
        self._last_guideline_context = context
        return context

    def _format_evidence(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        lines: List[str] = []
        for idx, (action, observation) in enumerate(intermediate_steps, 1):
            input_text = self._format_action_input(action)
            if input_text:
                lines.append(f"Step {idx}: {action.tool}[{input_text}]")
            else:
                lines.append(f"Step {idx}: {action.tool}")
            lines.append(observation)
        return "\n".join(lines).strip()

    def _format_action_input(self, action: AgentAction) -> str:
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input = tool_input.get("action_input")
        if tool_input in [None, ""]:
            return ""
        if action.tool in ["Laboratory Tests", "Imaging"]:
            if isinstance(tool_input, (list, dict)):
                return action_input_pretty_printer(tool_input, self.lab_test_mapping_df)
            return str(tool_input or "")
        if action.tool == "Diagnostic Criteria":
            return str(tool_input or "")
        return ""

    def _truncate_text_for_retrieval(self, text: str) -> str:
        if not text:
            return ""
        return truncate_text(
            self.llm_chain.llm.tokenizer,
            text,
            self.guidelines_query_tokens,
        )

    def _generate_differentials(self, context_text: str) -> List[str]:
        if not self._differential_chain or not context_text:
            return []
        raw = self._differential_chain.predict(
            input=context_text,
            max_differentials=str(self.guidelines_top_n),
            stop=[],
            temperature=self.guideline_temperature,
            top_p=self.guideline_top_p,
        )
        return self._parse_differentials(raw, self.guidelines_top_n)

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
        self, differentials: List[str], context_seed: str
    ) -> str:
        if not self._guideline_retriever:
            return ""
        snippets: List[str] = []
        base_context = context_seed.strip()
        for ddx in differentials:
            query = f"{ddx} diagnosis workup labs imaging\n{base_context}"
            self._guideline_retriever.k = self.guidelines_top_k
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
            self.llm_chain.llm.tokenizer,
            combined,
            self.guidelines_snippet_tokens,
        )


def create_prompt(
    tags, tool_names, add_tool_descr, tool_use_examples
) -> PromptTemplate:
    template = PromptTemplate(
        template=CHAT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "guideline_context"],
        partial_variables={
            "tool_names": action_input_pretty_printer(tool_names, None),
            "add_tool_descr": add_tool_descr,
            "examples": tool_use_examples,
            "system_tag_start": tags["system_tag_start"],
            "user_tag_start": tags["user_tag_start"],
            "ai_tag_start": tags["ai_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_end": tags["user_tag_end"],
        },
    )
    return template


def build_agent_executor_ZeroShot(
    patient,
    llm,
    lab_test_mapping_path,
    logfile,
    max_context_length,
    tags,
    include_ref_range,
    bin_lab_results,
    include_tool_use_examples,
    provide_diagnostic_criteria,
    summarize,
    model_stop_words,
    use_guideline_retrieval: bool = False,
    guidelines_path: str = "",
    guidelines_max_lines: Optional[int] = 2000,
    guidelines_source_filter: Optional[str] = None,
    guidelines_chunk_size: int = 1200,
    guidelines_chunk_overlap: int = 150,
    guidelines_top_k: int = 4,
    guidelines_top_n: int = 5,
    guidelines_snippet_tokens: int = 400,
    guidelines_context_tokens: int = 600,
    guidelines_query_tokens: int = 300,
    guideline_temperature: float = 0.2,
    guideline_top_p: float = 0.95,
):
    with open(lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = safe_pickle_load(f)

    # Define which tools the agent can use to answer user queries
    tools = [
        DoPhysicalExamination(action_results=patient),
        RunLaboratoryTests(
            action_results=patient,
            lab_test_mapping_df=lab_test_mapping_df,
            include_ref_range=include_ref_range,
            bin_lab_results=bin_lab_results,
        ),
        RunImaging(action_results=patient),
    ]

    # Go through options and see if we want to add any extra tools.
    add_tool_use_examples = ""
    add_tool_descr = ""
    if provide_diagnostic_criteria:
        tools.append(ReadDiagnosticCriteria())
        add_tool_descr += DIAG_CRIT_TOOL_DESCR
        add_tool_use_examples += DIAG_CRIT_TOOL_USE_EXAMPLE

    tool_names = [tool.name for tool in tools]

    # Create prompt
    tool_use_examples = ""
    if include_tool_use_examples:
        tool_use_examples = TOOL_USE_EXAMPLES.format(
            add_tool_use_examples=add_tool_use_examples
        )
    prompt = create_prompt(tags, tool_names, add_tool_descr, tool_use_examples)

    # Create output parser
    output_parser = DiagnosisWorkflowParser(lab_test_mapping_df=lab_test_mapping_df)

    # Initialize logging callback if file provided
    handler = None
    if logfile:
        handler = [FileCallbackHandler(logfile)]

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handler)

    # Create agent
    agent = CustomZeroShotAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=list(STOP_WORDS + model_stop_words),
        allowed_tools=tool_names,
        verbose=True,
        return_intermediate_steps=True,
        max_context_length=max_context_length,
        tags=tags,
        lab_test_mapping_df=lab_test_mapping_df,
        summarize=summarize,
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
        guideline_temperature=guideline_temperature,
        guideline_top_p=guideline_top_p,
    )
    agent.configure_guideline_retrieval()

    # Init agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        return_intermediate_steps=True,
        callbacks=handler,
    )

    return agent_executor
