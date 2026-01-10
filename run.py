import argparse
import json
import os
import re
from os.path import join
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import langchain
import hydra
import numpy as np
import torch
from langchain.schema import AgentAction
from loguru import logger
from omegaconf import DictConfig

from dataset.utils import load_hadm_from_file
from utils.pickle_compat import safe_pickle_load
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM
from agents.agent import build_agent_executor_ZeroShot
from agents.planner_judge_agent import build_agent_executor_PlannerJudge
from agents.AgentAction import AgentAction as CustomAgentAction

HF_ID_TO_MODEL_CONFIG = {
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama3Instruct70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama3.1Instruct70B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3Instruct70B",
    "aaditya/OpenBioLLM-Llama3-70B": "OpenBioLLM70B",
    "axiong/PMC_LLaMA_13B": "PMCLlama13B",
    "google/medgemma-27b-text-it": "MedGemma27B",
    "openai/gpt-oss-20b": "GPTOss20B",
    "openai/gpt-oss-120b": "GPTOss120B",
    "peterhan91/oss-20B-planner": "GPTOss20BPlanner",
}

CLI_ADAPTATION_WARNINGS = []


def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


def _quote_for_hydra(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_override(key: str, value: str, quote: bool = False) -> str:
    if value is None:
        return f"{key}=null"
    return f"{key}={_quote_for_hydra(value) if quote else value}"


def _parse_ref_range_entry(entry):
    if isinstance(entry, dict):
        lower = (
            entry.get("lower")
            or entry.get("low")
            or entry.get("ref_range_lower")
            or entry.get("min")
        )
        upper = (
            entry.get("upper")
            or entry.get("high")
            or entry.get("ref_range_upper")
            or entry.get("max")
        )
        if lower is None:
            for key, val in entry.items():
                if "low" in key.lower() or "min" in key.lower():
                    lower = val
                    break
        if upper is None:
            for key, val in entry.items():
                if "high" in key.lower() or "max" in key.lower():
                    upper = val
                    break
        return lower, upper
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[0], entry[1]
    return None, None


def _apply_reference_ranges(hadm_info_clean, ref_ranges_json_path: str):
    if not ref_ranges_json_path:
        return
    if not os.path.exists(ref_ranges_json_path):
        CLI_ADAPTATION_WARNINGS.append(
            f"Reference range file not found: {ref_ranges_json_path}"
        )
        return
    with open(ref_ranges_json_path, "r") as handle:
        ref_data = json.load(handle)
    parsed_ranges = {}
    for raw_key, entry in ref_data.items():
        try:
            itemid_int = int(raw_key)
        except (ValueError, TypeError):
            itemid_int = raw_key
        lower, upper = _parse_ref_range_entry(entry)
        if lower is None or upper is None:
            continue
        parsed_ranges[itemid_int] = (lower, upper)
    if not parsed_ranges:
        CLI_ADAPTATION_WARNINGS.append(
            f"No usable reference ranges parsed from {ref_ranges_json_path}"
        )
        return
    for hadm_entry in hadm_info_clean.values():
        lower_dict = hadm_entry.setdefault("Reference Range Lower", {})
        upper_dict = hadm_entry.setdefault("Reference Range Upper", {})
        for itemid, (lower, upper) in parsed_ranges.items():
            if itemid not in lower_dict:
                lower_dict[itemid] = lower
            if itemid not in upper_dict:
                upper_dict[itemid] = upper
            itemid_str = str(itemid)
            if itemid_str not in lower_dict:
                lower_dict[itemid_str] = lower
            if itemid_str not in upper_dict:
                upper_dict[itemid_str] = upper


def _extract_between(text: str, start_key: str, end_keys: Optional[list] = None) -> str:
    """Extract substring that follows start_key up to the first of end_keys or end of text.

    Case-insensitive header match; returns stripped text or empty string if not found.
    """
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


def _parse_structured_from_output(output_text: str) -> Dict[str, Optional[str]]:
    """Parse final_diagnosis, treatment, and rationale from the agent's final output text.

    Heuristics are aligned with evaluator parsing rules but simplified for logging.
    """
    if output_text is None:
        output_text = ""

    # Prefer sections around explicit headers; tolerate small deviations
    # Rationale: take the last Thought: ... before Final Diagnosis or Treatment
    rationale = _extract_between(
        output_text, "Thought:", ["Final Diagnosis:", "Treatment:"]
    )
    if not rationale:
        # Sometimes models write 'Reasoning:' or 'Rationale:'
        rationale = _extract_between(
            output_text, "Rationale:", ["Final Diagnosis:", "Treatment:"]
        ) or _extract_between(output_text, "Reasoning:", ["Final Diagnosis:", "Treatment:"])

    # Final Diagnosis
    # Capture up to a line that looks like Treatment: or end
    final_diagnosis = _extract_between(output_text, "Final Diagnosis:", ["Treatment:"])
    if not final_diagnosis:
        # Fallback to 'Diagnosis:'
        final_diagnosis = _extract_between(output_text, "Diagnosis:", ["Treatment:"])

    # Treatment
    treatment = _extract_between(output_text, "Treatment:")

    # Normalize empties to None for cleaner JSON
    def nz(val: str) -> Optional[str]:
        val = (val or "").strip()
        return val if val else None

    return {
        "final_diagnosis": nz(final_diagnosis),
        "treatment": nz(treatment),
        "rationale": nz(rationale),
    }


def _build_reference_from_hadm(hadm_entry: Dict[str, Any]) -> Tuple[str, List[str], List, List, List]:
    discharge = (
        hadm_entry.get("Discharge Diagnosis")
        or hadm_entry.get("Discharge")
        or ""
    )
    icd = hadm_entry.get("ICD Diagnosis") or hadm_entry.get("ICD Diagnoses") or []
    if isinstance(icd, str):
        icd = [icd]
    procedures_icd9 = hadm_entry.get("Procedures ICD9") or []
    if isinstance(procedures_icd9, str):
        procedures_icd9 = [procedures_icd9]
    procedures_icd10 = hadm_entry.get("Procedures ICD10") or []
    if isinstance(procedures_icd10, str):
        procedures_icd10 = [procedures_icd10]
    procedures_discharge = hadm_entry.get("Procedures Discharge") or []
    if isinstance(procedures_discharge, str):
        procedures_discharge = [procedures_discharge]
    return (
        discharge,
        icd,
        procedures_icd9,
        procedures_icd10,
        procedures_discharge,
    )


def _format_executed_plan(intermediate_steps: Any) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    if not isinstance(intermediate_steps, list):
        return executed
    for step in intermediate_steps:
        if not isinstance(step, (list, tuple)) or len(step) < 1:
            continue
        action = step[0]
        tool = getattr(action, "tool", None)
        tool_input = getattr(action, "tool_input", None)
        action_input = None
        if isinstance(tool_input, dict):
            action_input = tool_input.get("action_input")
        else:
            action_input = tool_input
        executed.append(
            {
                "tool": tool,
                "action_input": action_input,
            }
        )
    return executed


def _compute_judge_metrics(judge_log: Any) -> Dict[str, Any]:
    if not isinstance(judge_log, list):
        return {}
    total = len(judge_log)
    counts = {"proceed": 0, "skip": 0, "modify": 0, "add": 0, "stop": 0}
    for entry in judge_log:
        decision = str((entry or {}).get("decision", "")).strip().lower()
        if decision in counts:
            counts[decision] += 1
    modified = counts["skip"] + counts["modify"] + counts["add"]
    mod_rate = (modified / total) if total else 0.0
    return {
        "judge_decision_counts": counts,
        "step_modification_rate": mod_rate,
        "judge_decision_total": total,
    }


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, AgentAction):
        return {
            "tool": value.tool,
            "tool_input": _make_json_safe(value.tool_input),
            "log": value.log,
        }
    if isinstance(value, CustomAgentAction):
        return {
            "tool": value.tool,
            "tool_input": _make_json_safe(value.tool_input),
            "log": value.log,
        }
    return str(value)


def _serialize_result_for_json(result: Any) -> Any:
    if isinstance(result, dict):
        serialized: Dict[str, Any] = {}
        for key, value in result.items():
            if key == "intermediate_steps" and isinstance(value, list):
                steps = []
                for step in value:
                    if isinstance(step, (list, tuple)) and len(step) == 2:
                        action, observation = step
                        steps.append(
                            {
                                "action": _make_json_safe(action),
                                "observation": _make_json_safe(observation),
                            }
                        )
                    else:
                        steps.append(_make_json_safe(step))
                serialized[key] = steps
            else:
                serialized[key] = _make_json_safe(value)
        return serialized
    return _make_json_safe(result)


def _load_patient_data(args: DictConfig):
    hadm_pickle = getattr(args, "hadm_pickle_path", None)
    base_mimic = getattr(args, "base_mimic", "")
    if hadm_pickle:
        hadm_path = hadm_pickle
        if not os.path.isabs(hadm_path) and base_mimic:
            hadm_path = join(base_mimic, hadm_path)
        with open(hadm_path, "rb") as handle:
            hadm_info_clean = safe_pickle_load(handle)
    else:
        hadm_info_clean = load_hadm_from_file(
            f"{args.pathology}_hadm_info_first_diag", base_mimic=base_mimic
        )
    ref_ranges_json_path = getattr(args, "ref_ranges_json_path", "")
    _apply_reference_ranges(hadm_info_clean, ref_ranges_json_path)
    return hadm_info_clean


def _load_patient_id_list(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    ids = []
    for line in raw.splitlines():
        for chunk in re.split(r"[,\s]+", line.strip()):
            if not chunk:
                continue
            try:
                ids.append(int(chunk))
            except ValueError:
                ids.append(chunk)
    return ids


def _adapt_slurm_cli_args():
    global CLI_ADAPTATION_WARNINGS
    if len(sys.argv) <= 1:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paths")
    parser.add_argument("--pathology")
    parser.add_argument("--hadm-pkl")
    parser.add_argument("--lab-map-pkl")
    parser.add_argument("--ref-ranges-json")
    parser.add_argument("--hf-model-id")
    parser.add_argument("--agent-type")
    parser.add_argument("--include-ref-range", action="store_true")
    parser.add_argument("--bin-lab-results", action="store_true")
    parser.add_argument("--use-calculator", action="store_true")
    parser.add_argument("--calculator-include-units", action="store_true")
    parser.add_argument("--local-logging-dir")
    parser.add_argument("--base-model-cache")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--eval-accuracy", action="store_true")
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    recognized = any(
        [
            parsed.paths,
            parsed.pathology,
            parsed.hadm_pkl,
            parsed.lab_map_pkl,
            parsed.ref_ranges_json,
            parsed.hf_model_id,
            parsed.agent_type,
            parsed.include_ref_range,
            parsed.bin_lab_results,
            parsed.use_calculator,
            parsed.calculator_include_units,
            parsed.local_logging_dir,
            parsed.base_model_cache,
            parsed.reasoning_effort,
            parsed.sample_n is not None,
            parsed.eval_accuracy,
        ]
    )
    if not recognized:
        return

    overrides = []

    if parsed.paths:
        overrides.append(_format_override("paths", parsed.paths))
    elif parsed.hadm_pkl and "/cbica/" in parsed.hadm_pkl:
        overrides.append(_format_override("paths", "cbica"))

    if parsed.hadm_pkl:
        overrides.append(
            _format_override("hadm_pickle_path", parsed.hadm_pkl, quote=True)
        )
        inferred_pathology = parsed.pathology or Path(parsed.hadm_pkl).stem
        overrides.append(
            _format_override("pathology", inferred_pathology.replace("-", "_"))
        )
    elif parsed.pathology:
        overrides.append(
            _format_override("pathology", parsed.pathology.replace("-", "_"))
        )

    if parsed.lab_map_pkl:
        overrides.append(
            _format_override("lab_test_mapping_path", parsed.lab_map_pkl, quote=True)
        )

    if parsed.ref_ranges_json:
        overrides.append(
            _format_override("ref_ranges_json_path", parsed.ref_ranges_json, quote=True)
        )

    if parsed.hf_model_id:
        mapped_model = HF_ID_TO_MODEL_CONFIG.get(parsed.hf_model_id)
        if mapped_model:
            overrides.append(_format_override("model", mapped_model))
        else:
            overrides.append(
                _format_override("model_name", parsed.hf_model_id, quote=True)
            )
            CLI_ADAPTATION_WARNINGS.append(
                f"No pre-defined model config for {parsed.hf_model_id}; "
                "using direct model_name override."
            )

    if parsed.agent_type:
        agent_type = parsed.agent_type.lower()
        if agent_type == "zeroshot":
            overrides.append(_format_override("agent", "ZeroShot"))
        elif agent_type in {"plannerjudge", "planner-judge", "planner_judge", "steer"}:
            overrides.append(_format_override("agent", "PlannerJudge"))
        else:
            overrides.append(_format_override("agent", parsed.agent_type))

    if parsed.include_ref_range:
        overrides.append("include_ref_range=true")
    if parsed.bin_lab_results:
        overrides.append("bin_lab_results=true")
    if parsed.local_logging_dir:
        overrides.append(
            _format_override("local_logging_dir", parsed.local_logging_dir, quote=True)
        )
    if parsed.base_model_cache:
        overrides.append(
            _format_override("base_models", parsed.base_model_cache, quote=True)
        )
    if parsed.reasoning_effort:
        overrides.append(
            _format_override("gpt_oss_reasoning_effort", parsed.reasoning_effort)
        )
    if parsed.sample_n is not None:
        overrides.append(_format_override("sample_n", str(parsed.sample_n)))
    if parsed.eval_accuracy:
        overrides.append("eval_accuracy=true")

    if parsed.use_calculator:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --use-calculator (feature not available in this repo)."
        )
    if parsed.calculator_include_units:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --calculator-include-units (feature not available in this repo)."
        )

    sys.argv = [sys.argv[0]] + overrides + remaining


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    if not args.self_consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load patient data
    hadm_info_clean = _load_patient_data(args)
    patient_list_path = getattr(args, "patient_list_path", None)
    if patient_list_path:
        try:
            patient_ids = _load_patient_id_list(patient_list_path)
            if patient_ids:
                hadm_info_clean = {
                    k: hadm_info_clean[k]
                    for k in patient_ids
                    if k in hadm_info_clean
                }
        except OSError as exc:
            logger.warning(f"Failed to read patient_list_path: {exc}")
    sample_n = getattr(args, "sample_n", None)
    if sample_n and sample_n > 0 and sample_n < len(hadm_info_clean):
        rng = random.Random(args.seed)
        sampled = rng.sample(list(hadm_info_clean.keys()), int(sample_n))
        hadm_info_clean = {k: hadm_info_clean[k] for k in sampled}

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    # Load desired model
    llm = CustomLLM(
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        load_in_4bit=getattr(args, "load_in_4bit", None),
        load_in_8bit=getattr(args, "load_in_8bit", None),
        torch_dtype=getattr(args, "torch_dtype", None),
        attn_implementation=getattr(args, "attn_implementation", None),
        seed=args.seed,
        self_consistency=args.self_consistency,
        gpt_oss_reasoning_effort=args.gpt_oss_reasoning_effort,
        openai_reasoning_effort=getattr(args, "openai_reasoning_effort", None),
        openai_text_verbosity=getattr(args, "openai_text_verbosity", None),
        openai_max_output_tokens=getattr(args, "openai_max_output_tokens", None),
    )
    llm.load_model(args.base_models)

    agent_name = str(getattr(args, "agent", "ZeroShot") or "ZeroShot").lower()
    use_planner_judge = agent_name in {"plannerjudge", "planner_judge", "planner-judge", "steer"}
    planner_llm = None
    planner_tags = tags
    planner_stop_words = args.stop_words
    if use_planner_judge:
        planner_tags = {
            "system_tag_start": args.planner_system_tag_start
            if args.planner_system_tag_start is not None
            else args.system_tag_start,
            "user_tag_start": args.planner_user_tag_start
            if args.planner_user_tag_start is not None
            else args.user_tag_start,
            "ai_tag_start": args.planner_ai_tag_start
            if args.planner_ai_tag_start is not None
            else args.ai_tag_start,
            "system_tag_end": args.planner_system_tag_end
            if args.planner_system_tag_end is not None
            else args.system_tag_end,
            "user_tag_end": args.planner_user_tag_end
            if args.planner_user_tag_end is not None
            else args.user_tag_end,
            "ai_tag_end": args.planner_ai_tag_end
            if args.planner_ai_tag_end is not None
            else args.ai_tag_end,
        }
        planner_stop_words = (
            args.planner_stop_words
            if getattr(args, "planner_stop_words", None) is not None
            else args.stop_words
        )
        planner_model_name = (
            args.planner_model_name
            if getattr(args, "planner_model_name", None) is not None
            else args.model_name
        )
        planner_openai_api_key = (
            args.planner_openai_api_key
            if getattr(args, "planner_openai_api_key", None) is not None
            else args.openai_api_key
        )
        planner_max_context = (
            args.planner_max_context_length
            if getattr(args, "planner_max_context_length", None) is not None
            else args.max_context_length
        )
        planner_exllama = (
            args.planner_exllama
            if getattr(args, "planner_exllama", None) is not None
            else args.exllama
        )
        planner_llm = CustomLLM(
            model_name=planner_model_name,
            openai_api_key=planner_openai_api_key,
            tags=planner_tags,
            max_context_length=planner_max_context,
            exllama=planner_exllama,
            load_in_4bit=getattr(args, "load_in_4bit", None),
            load_in_8bit=getattr(args, "load_in_8bit", None),
            torch_dtype=getattr(args, "torch_dtype", None),
            attn_implementation=getattr(args, "attn_implementation", None),
            seed=args.seed,
            self_consistency=args.self_consistency,
            gpt_oss_reasoning_effort=args.gpt_oss_reasoning_effort,
            openai_reasoning_effort=getattr(args, "openai_reasoning_effort", None),
            openai_text_verbosity=getattr(args, "openai_text_verbosity", None),
            openai_max_output_tokens=getattr(args, "openai_max_output_tokens", None),
        )
        planner_llm.load_model(args.base_models)

    # Simplified, structured output directory and filenames
    # Folder structure: <local_logging_dir>/<pathology>/<model_tag>/<YYYYMMDD-HHMMSS>
    date_time = datetime.fromtimestamp(time.time())
    timestamp = date_time.strftime("%Y%m%d-%H%M%S")
    model_tag = args.model_name.split("/")[-1]
    run_dir = join(args.local_logging_dir, args.pathology, model_tag, timestamp)

    os.makedirs(run_dir, exist_ok=True)

    # Setup logfile and results files with simple names
    log_path = join(run_dir, "run.log")
    results_json_path = join(run_dir, "results.json")
    json_results: Dict[str, Any] = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r", encoding="utf-8") as json_file:
                existing_results = json.load(json_file)
                if isinstance(existing_results, dict):
                    json_results = {str(key): value for key, value in existing_results.items()}
        except json.JSONDecodeError:
            logger.warning(
                f"Existing results JSON at {results_json_path} could not be parsed; starting fresh."
            )
        except OSError as exc:
            logger.warning(
                f"Unable to read existing results JSON at {results_json_path}: {exc}"
            )
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True, serialize=True)
    langchain.debug = True
    for warning in CLI_ADAPTATION_WARNINGS:
        logger.warning(warning)
    CLI_ADAPTATION_WARNINGS.clear()

    # Set langsmith project name (optional)
    # os.environ["LANGCHAIN_PROJECT"] = f"{args.pathology}-{model_tag}-{timestamp}"

    # Predict for all patients
    first_patient_seen = False
    for _id in hadm_info_clean.keys():
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue

        logger.info(f"Processing patient: {_id}")

        # Build
        if use_planner_judge:
            agent_executor = build_agent_executor_PlannerJudge(
                patient=hadm_info_clean[_id],
                llm=llm,
                planner_llm=planner_llm,
                lab_test_mapping_path=args.lab_test_mapping_path,
                max_context_length=args.max_context_length,
                tags=tags,
                planner_tags=planner_tags,
                include_ref_range=args.include_ref_range,
                bin_lab_results=args.bin_lab_results,
                provide_diagnostic_criteria=args.provide_diagnostic_criteria,
                planner_stop_words=planner_stop_words,
                judge_stop_words=args.stop_words,
                planner_temperature=args.planner_temperature,
                planner_top_p=args.planner_top_p,
                judge_temperature=args.judge_temperature,
                max_steps=args.planner_max_steps,
            )
        else:
            agent_executor = build_agent_executor_ZeroShot(
                patient=hadm_info_clean[_id],
                llm=llm,
                lab_test_mapping_path=args.lab_test_mapping_path,
                logfile=log_path,
                max_context_length=args.max_context_length,
                tags=tags,
                include_ref_range=args.include_ref_range,
                bin_lab_results=args.bin_lab_results,
                include_tool_use_examples=args.include_tool_use_examples,
                provide_diagnostic_criteria=args.provide_diagnostic_criteria,
                summarize=args.summarize,
                model_stop_words=args.stop_words,
            )

        # Run
        result = agent_executor(
            {"input": hadm_info_clean[_id]["Patient History"].strip()}
        )
        # Build structured output alongside raw output
        try:
            raw_output_text = result.get("output", "")
        except Exception:
            raw_output_text = ""
        structured = _parse_structured_from_output(str(raw_output_text))
        # Preserve original under output_raw and set output to structured dict
        result_dict_safe = dict(result)
        result_dict_safe["output_raw"] = raw_output_text
        result_dict_safe["output"] = structured

        if use_planner_judge:
            result_dict_safe["plan_initial"] = (
                result.get("planner_summary")
                or result.get("planner_plan")
                or result.get("planner_raw")
            )
            result_dict_safe["plan_executed"] = _format_executed_plan(
                result.get("intermediate_steps")
            )
            result_dict_safe.update(_compute_judge_metrics(result.get("judge_log")))

        if getattr(args, "eval_accuracy", True):
            try:
                evaluator = load_evaluator(args.pathology)
                reference = _build_reference_from_hadm(hadm_info_clean[_id])
                eval_result = evaluator._evaluate_agent_trajectory(
                    prediction=str(raw_output_text),
                    input=hadm_info_clean[_id]["Patient History"].strip(),
                    agent_trajectory=result.get("intermediate_steps", []),
                    reference=reference,
                )
                result_dict_safe["evaluation"] = eval_result
                result_dict_safe["diagnosis_accuracy"] = eval_result.get(
                    "scores", {}
                ).get("Diagnosis")
            except Exception as exc:
                logger.warning(f"Evaluation failed for {_id}: {exc}")

        json_results[str(_id)] = _serialize_result_for_json(result_dict_safe)
        try:
            with open(results_json_path, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error(
                f"Failed to write results JSON to {results_json_path}: {exc}"
            )

if __name__ == "__main__":
    _adapt_slurm_cli_args()
    run()
