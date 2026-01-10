import os
import re
from os.path import join
from typing import Any, List, Mapping, Dict, Optional

import torch
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import GenerationConfig, StoppingCriteriaList
from langchain.llms.base import LLM
# from exllamav2.generator import ExLlamaV2Sampler
import tiktoken

from models.utils import create_stop_criteria, create_stop_criteria_exllama
from agents.agent import STOP_WORDS
from utils.nlp import extract_sections


class CustomLLM(LLM):
    model_name: str
    max_context_length: int
    probabilities: torch.Tensor = None
    exllama: bool = False
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    truncation_side: str = "left"
    model: Any
    generator: Any
    tokenizer: Any
    seed: int
    self_consistency: bool = False
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = None

    openai_api_key: str = None
    tags: Dict[str, str] = None
    gpt_oss_reasoning_effort: str = None
    openai_reasoning_effort: Optional[str] = None
    openai_text_verbosity: Optional[str] = None
    openai_max_output_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> Any:
        return "custom"

    @property
    def _llm_name(self) -> str:
        return self.model_name

    @property
    def _llm_device(self) -> str:
        return self.model.device

    @property
    def _llm_8bit(self) -> bool:
        return self.load_in_8bit is True

    @property
    def _llm_4bit(self) -> bool:
        return self.load_in_4bit is True

    @property
    def _llm_truncation_side(self) -> str:
        return self.truncation_side

    def _resolve_torch_dtype(self) -> Optional[torch.dtype]:
        if self.torch_dtype is None:
            return None
        if isinstance(self.torch_dtype, torch.dtype):
            return self.torch_dtype
        if isinstance(self.torch_dtype, str):
            key = self.torch_dtype.strip().lower()
            mapping = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            if key in mapping:
                return mapping[key]
            try:
                return getattr(torch, key)
            except AttributeError as exc:
                raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype}") from exc
        raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype}")

    def load_model(self, base_models: str) -> None:
        torch.cuda.empty_cache()

        if self.model_name == "Human":
            return
        elif self.openai_api_key:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            openai.api_key = self.openai_api_key
            return
        elif (
            self.model_name
            == "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
        ):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k",
                device_map="auto",
                torch_dtype=torch.float16,
            )

        elif "GPTQ" in self.model_name:
            if self.exllama:
                from exllamav2 import ExLlamaV2Cache
                from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
                from models.exllamav2_generator_base_custom import (
                    ExLlamaV2BaseGenerator,
                )

                torch.cuda._lazy_init()
                config = ExLlamaV2Config()
                config.model_dir = join(base_models, self.model_name)
                config.prepare()
                config.max_seq_len = self.max_context_length
                config.scale_pos_emb = 1.0
                config.scale_alpha_value = 1.0
                config.no_flash_attn = False
                self.model = ExLlamaV2(config)
                self.model.load()
                self.tokenizer = ExLlamaV2Tokenizer(config)
                cache = ExLlamaV2Cache(self.model)
                self.generator = ExLlamaV2BaseGenerator(
                    self.model, cache, self.tokenizer
                )
                self.generator.warmup()

            else:
                from transformers import LlamaTokenizer, LlamaForCausalLM

                base_model = join(base_models, self.model_name)

                self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                # Lazily import to avoid importing auto_gptq unless needed
                from auto_gptq import exllama_set_max_input_length

                self.model = exllama_set_max_input_length(
                    self.model, self.max_context_length
                )

        elif (
            self.model_name == "meta-llama/Meta-Llama-3-70B-Instruct"
            or self.model_name == "aaditya/OpenBioLLM-Llama3-70B"
            or self.model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct"
            or self.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        ):
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            print(f"loading from {base_models}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=base_models,
                )
            except Exception as exc_fast:
                print(
                    f"failed to load fast tokenizer for {self.model_name}: {exc_fast}. "
                    "Retrying with use_fast=False."
                )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=base_models,
                        use_fast=False,
                    )
                except Exception as exc_slow_auto:
                    # Some newer tokenizers JSON require a newer `tokenizers` wheel.
                    # Fall back to the pure Python SentencePiece-based tokenizer.
                    print(
                        "AutoTokenizer(use_fast=False) also failed. "
                        f"Falling back to LlamaTokenizer. Error: {exc_slow_auto}"
                    )
                    from transformers import LlamaTokenizer

                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=base_models,
                    )

            eot = "<|eot_id|>"
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            except Exception:
                eot_id = None
            # If the tokenizer doesn't know about <|eot_id|>, fall back to eos
            if eot_id is None or (hasattr(self.tokenizer, "unk_token_id") and eot_id == self.tokenizer.unk_token_id):
                eot = getattr(self.tokenizer, "eos_token", None) or "<|endoftext|>"
                eot_id = getattr(self.tokenizer, "eos_token_id", None)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id

            print("loaded tokenizer")
            use_4bit = self.load_in_4bit
            use_8bit = self.load_in_8bit
            if use_4bit is None and use_8bit is None:
                use_4bit = True
            if use_4bit and use_8bit:
                raise ValueError("Only one of load_in_4bit or load_in_8bit can be true.")

            model_kwargs = {
                "cache_dir": base_models,
                "device_map": "auto",
            }
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

            resolved_dtype = self._resolve_torch_dtype()
            if use_4bit or use_8bit:
                bnb_kwargs = {
                    "load_in_4bit": bool(use_4bit),
                    "load_in_8bit": bool(use_8bit),
                }
                if use_4bit:
                    bnb_kwargs["bnb_4bit_compute_dtype"] = (
                        resolved_dtype or torch.bfloat16
                    )
                bb_cfg = BitsAndBytesConfig(**bnb_kwargs)
                model_kwargs["quantization_config"] = bb_cfg
            else:
                if resolved_dtype is not None:
                    model_kwargs["torch_dtype"] = resolved_dtype

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            print("loaded model")

        elif self.model_name == "google/medgemma-27b-text-it":
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            print(f"loading from {base_models}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=base_models,
                )
            except Exception as exc_fast:
                print(
                    f"failed to load fast tokenizer for {self.model_name}: {exc_fast}. "
                    "Retrying with use_fast=False."
                )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=base_models,
                        use_fast=False,
                    )
                except Exception as exc_slow_auto:
                    print(
                        "AutoTokenizer(use_fast=False) also failed. "
                        f"Falling back to GemmaTokenizer. Error: {exc_slow_auto}"
                    )
                    from transformers import GemmaTokenizer

                    self.tokenizer = GemmaTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=base_models,
                    )

            eot = "<end_of_turn>"
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            except Exception:
                eot_id = None
            if eot_id is None or (
                hasattr(self.tokenizer, "unk_token_id")
                and eot_id == self.tokenizer.unk_token_id
            ):
                eot = None
                eot_id = None
            if self.tokenizer.pad_token_id is None:
                pad_token = eot or getattr(self.tokenizer, "eos_token", None)
                pad_token_id = eot_id or getattr(self.tokenizer, "eos_token_id", None)
                if pad_token is not None:
                    self.tokenizer.pad_token = pad_token
                if pad_token_id is not None:
                    self.tokenizer.pad_token_id = pad_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            print("loaded model")

        elif (
            self.model_name.startswith("openai/gpt-oss")
            or self.model_name == "peterhan91/oss-20B-planner"
        ):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"loading from {base_models}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                trust_remote_code=True,
            )
            torch_dtype = (
                torch.bfloat16
                if self.model_name == "peterhan91/oss-20B-planner"
                else "auto"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("loaded model")

        elif self.model_name == "axiong/PMC_LLaMA_13B":
            from transformers import LlamaTokenizer, LlamaForCausalLM

            self.tokenizer = LlamaTokenizer.from_pretrained("axiong/PMC_LLaMA_13B")
            self.model = LlamaForCausalLM.from_pretrained(
                "axiong/PMC_LLaMA_13B", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "google/flan-t5-xxl":
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "bigscience/T0pp":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            orig = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

            self.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "bigscience/T0pp", device_map="auto", torch_dtype=torch.float16
            )

            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = orig

        elif self.model_name.startswith("togethercomputer/RedPajama-INCITE"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

        elif self.model_name.startswith("tiiuae/falcon"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.load_in_8bit = "40b" in self.model_name
            if torch.cuda.device_count() > 1:
                self.load_in_8bit = False
                device_map = "balanced_low_0"
            else:
                device_map = "sequential"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=self.load_in_8bit,
                device_map=device_map,
            )

        else:
            raise ValueError("Model name not recognized")

        if not self.model_name.startswith("tiiuae/falcon") and not self.exllama:
            self.model.eval()
            if torch.__version__ >= "2":
                self.model = torch.compile(self.model)

        self.tokenizer.truncation_side = "left"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def _supports_responses_api(self) -> bool:
        return hasattr(openai, "OpenAI")

    def _extract_responses_text(self, response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "output_text"):
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text
        if isinstance(response, dict):
            if response.get("output_text"):
                return response.get("output_text") or ""
            output = response.get("output") or []
        else:
            output = getattr(response, "output", None) or []
        for item in output:
            if isinstance(item, dict):
                if item.get("type") != "message":
                    continue
                content = item.get("content") or []
            else:
                if getattr(item, "type", None) != "message":
                    continue
                content = getattr(item, "content", None) or []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "output_text" and part.get("text"):
                        return part.get("text") or ""
                else:
                    if getattr(part, "type", None) == "output_text":
                        text = getattr(part, "text", None)
                        if text:
                            return text
        return ""

    def _openai_responses_create(self, messages, do_sample, temperature, top_p) -> str:
        client = openai.OpenAI(api_key=self.openai_api_key)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": messages,
        }
        if self.openai_reasoning_effort:
            payload["reasoning"] = {"effort": self.openai_reasoning_effort}
        if self.openai_text_verbosity:
            payload["text"] = {"verbosity": self.openai_text_verbosity}
        if self.openai_max_output_tokens is not None:
            payload["max_output_tokens"] = int(self.openai_max_output_tokens)

        reasoning_effort = (self.openai_reasoning_effort or "none").lower()
        if reasoning_effort == "none":
            payload["temperature"] = temperature if do_sample else 0.0
            payload["top_p"] = top_p

        response = client.responses.create(**payload)
        return self._extract_responses_text(response)

    def remove_input_tokens(self, output_tokens, ids):
        # Truncate the larger tensor to match the size of the smaller one
        min_size = min(output_tokens.size(1), ids.size(1))
        truncated_output_tokens = output_tokens[:, :min_size]
        truncated_ids = ids[:, :min_size]

        # Element-wise comparison and cumulative product to count length of common prefix
        common_prefix = (
            (truncated_output_tokens == truncated_ids).cumprod(dim=0).sum().item()
        )

        return output_tokens[:, common_prefix:]

    def _call(
        self,
        prompt: str,
        stop: List[str],
        do_sample=True,
        temperature=0.01,
        top_k=1,
        top_p=0.95,
        num_beams=1,
        repetition_penalty=1.2,
        length_penalty=1.0,
        **kwargs,
    ) -> str:
        self.probabilities = None
        if self.model_name == "Human":
            output = input(prompt)

        elif (
            self.model_name.startswith("openai/gpt-oss")
            or self.model_name == "peterhan91/oss-20B-planner"
        ):
            messages = extract_sections(
                prompt,
                self.tags,
            )
            if not messages:
                messages = [{"role": "user", "content": prompt}]

            chat_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
            if self.gpt_oss_reasoning_effort:
                chat_kwargs["reasoning_effort"] = self.gpt_oss_reasoning_effort
            try:
                input_ids = self.tokenizer.apply_chat_template(messages, **chat_kwargs).to(
                    self.model.device
                )
            except TypeError:
                fallback_kwargs = {k: v for k, v in chat_kwargs.items() if k != "reasoning_effort"}
                if len(fallback_kwargs) == len(chat_kwargs):
                    raise
                input_ids = self.tokenizer.apply_chat_template(messages, **fallback_kwargs).to(
                    self.model.device
                )

            gen_kwargs = dict(
                max_new_tokens=kwargs.get("max_new_tokens", self.max_context_length),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            if top_k is not None:
                gen_kwargs["top_k"] = top_k

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **gen_kwargs)

            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
            output = self._extract_gpt_oss_final_message(decoded)

        elif self.openai_api_key:
            messages = extract_sections(
                prompt,
                self.tags,
            )

            if self.model_name.startswith("gpt-5") and self._supports_responses_api():
                output = self._openai_responses_create(
                    messages=messages,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                response = self.completion_with_backoff(
                    model=self.model_name,
                    messages=messages,
                    stop=STOP_WORDS,
                    temperature=0.0,
                    seed=self.seed,
                )
                output = response["choices"][0]["message"]["content"]
        elif self.exllama:
            with torch.inference_mode():
                ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
                tokens_prompt = ids.shape[-1]

                settings = ExLlamaV2Sampler.Settings()
                if self.self_consistency:
                    settings = settings.clone()
                    settings.temperature = 0.7
                    seed = None
                else:
                    settings = settings.greedy_clone()
                    seed = self.seed

                stop_criteria = create_stop_criteria_exllama(
                    stop, self.tokenizer.eos_token_id, self.tokenizer
                )

                output_tokens, self.probabilities = self.generator.generate_simple(
                    prompt,
                    gen_settings=settings,
                    num_tokens=self.max_context_length - tokens_prompt,
                    seed=seed,
                    token_healing=True,
                    encode_special_tokens=True,
                    decode_special_tokens=False,
                    stop_criteria=stop_criteria,
                )

                output_tokens = self.remove_input_tokens(output_tokens, ids)
                output = self.tokenizer.decode(
                    output_tokens, decode_special_tokens=False
                )[0]
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(self.model.device)

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

            stop_criteria = create_stop_criteria(
                stop, self.tokenizer, self.model.device
            )

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.max_context_length,
                )

            s = generation_output.sequences
            s_no_input = s[:, input_ids.shape[1] :]
            output = self.tokenizer.batch_decode(s_no_input, skip_special_tokens=True)[
                0
            ]

        # Remove observations strings from output if generated
        for stop_word in STOP_WORDS + stop:
            output = output.replace(stop_word, "")

        return output.strip()

    @staticmethod
    def _extract_gpt_oss_final_message(raw: str) -> str:
        """Return only the final-channel content from GPT-OSS outputs."""
        if not raw:
            return raw

        final_marker = "<|channel|>final<|message|>"
        idx = raw.rfind(final_marker)
        if idx != -1:
            tail = raw[idx + len(final_marker) :]
            end_idx = tail.find("<|end|>")
            if end_idx != -1:
                tail = tail[:end_idx]
            tail = tail.strip()
            if tail:
                return tail

        start_marker = "<|start|>assistant"
        if start_marker in raw:
            tail = raw.split(start_marker)[-1]
            end_idx = tail.find("<|end|>")
            if end_idx != -1:
                tail = tail[:end_idx]
            tail = tail.strip()
            if tail:
                if tail.startswith(final_marker):
                    tail = tail[len(final_marker) :].strip()
                return tail

        matches = list(re.finditer(r"assistantfinal", raw))
        if matches:
            tail = raw[matches[-1].end() :].strip()
            if tail:
                return tail

        return raw

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }
