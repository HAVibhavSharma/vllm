"""End-to-end CacheBlend / KVLink benchmark driver (v1 vLLM).

Ported from the `epic` repo's benchmarks_ours/evals/e2e/main.py
(vLLM 0.8.5) to run on current vLLM (v1 engine, v1 attention backends).

Key differences vs. the 0.8.5 version:

* The attention backend is forced to XFORMERS via the
  ``attention_backend="XFORMERS"`` kwarg on ``LLM()``. This routes the
  ported XFormers v1 backend (which carries the CacheBlend / KVLink
  hooks). The old ``global_force_attn_backend`` helper no longer exists.

* The v1 engine does not expose ``driver_worker.model_runner.model``
  directly. We use ``llm.apply_model(fn)`` to reach into the model
  running inside the worker process. Utilities below hide that.

* The target model is Qwen3.5 (the dense ``Qwen3_5ForCausalLM`` class).
  Other models from the original list are left for reference but not
  exercised by default — the CacheBlend hooks have only been plumbed
  through ``Qwen3NextAttention`` / ``Qwen3NextDecoderLayer`` /
  ``Qwen3_5Model`` (which is what ``Qwen3_5ForCausalLM`` uses).
"""

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from tqdm.contrib import tenumerate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from benchmarks_ours.data_sets.data_set import Data_set
from benchmarks_ours.evals.utils import set_seed, str2class
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the inner language model (Qwen3_5Model) from whatever the
    LLM ended up wrapping.

    Handles both Qwen3_5ForCausalLM and Qwen3_5ForConditionalGeneration.
    """
    # Multimodal: Qwen3_5ForConditionalGeneration.language_model.model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    # Causal LM: Qwen3_5ForCausalLM.model
    if hasattr(model, "model"):
        return model.model
    raise AttributeError(
        "Could not locate the language model. Expected a Qwen3_5ForCausalLM "
        "or Qwen3_5ForConditionalGeneration wrapper."
    )


def set_cache_fuse_flags(llm: LLM, **updates) -> None:
    """Merge `updates` into model.cache_fuse_metadata on every worker."""

    def _apply(model: torch.nn.Module):
        lang_model = _get_language_model(model)
        lang_model.cache_fuse_metadata.update(updates)

    llm.apply_model(_apply)


def collect_hack_kv(llm: LLM) -> list[list[torch.Tensor]]:
    """Fetch each full-attention layer's last-seen (K, V) captured on the
    `collect=True` pass.

    Returns a list, one entry per full_attention layer, each a
    ``[key_tensor, value_tensor]`` pair already moved to CPU.
    """

    def _apply(model: torch.nn.Module):
        lang_model = _get_language_model(model)
        out: list[list[torch.Tensor]] = []
        for layer in lang_model.layers:
            if getattr(layer, "layer_type", None) != "full_attention":
                out.append([None, None])
                continue
            hk = getattr(layer.self_attn, "hack_kv", None)
            if not hk:
                out.append([None, None])
                continue
            # Move to CPU to minimize cross-process copy overhead.
            out.append([hk[0].detach().cpu(), hk[1].detach().cpu()])
        return out

    # Single-worker → first (and only) result.
    results = llm.apply_model(_apply)
    return results[0]


def set_old_kvs(llm: LLM, old_kvs: list[list[torch.Tensor]]) -> None:
    """Install `old_kvs` onto the language model on each worker.

    `old_kvs` is a list of [key, value] pairs indexed by layer.
    Tensors are moved back to the worker's device before install.
    """

    def _apply(model: torch.nn.Module):
        lang_model = _get_language_model(model)
        device = next(lang_model.parameters()).device
        dtype = next(lang_model.parameters()).dtype
        new_kvs: list[list[torch.Tensor | None]] = []
        for pair in old_kvs:
            if pair is None or pair[0] is None:
                new_kvs.append([None, None])
            else:
                new_kvs.append(
                    [
                        pair[0].to(device=device, dtype=dtype),
                        pair[1].to(device=device, dtype=dtype),
                    ]
                )
        lang_model.old_kvs = new_kvs

    llm.apply_model(_apply)


def reset_layer_hack_kv(llm: LLM) -> None:
    """Clear every layer's `hack_kv` and reset `old_kvs` to empty slots."""

    def _apply(model: torch.nn.Module):
        lang_model = _get_language_model(model)
        for layer in lang_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.hack_kv = []
        lang_model.old_kvs = [[None, None] for _ in range(len(lang_model.layers))]

    llm.apply_model(_apply)


@dataclass
class EvalConfigs:
    dataset: str
    model: str
    approach: str
    tot_num_data: int = 200
    all_datasets: List[str] = field(
        default_factory=lambda: [
            "2wikimqa",
            "musique",
            "samsum",
            "multi_news",
            "hotpotqa",
            "needle",
        ]
    )
    # For CacheBlend / KVLink hook plumbing we currently only ship the
    # Qwen3.5 path. Extending to other models requires porting the same
    # self_attn hooks into their respective model files.
    all_models: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-7B",
            # Original-stack models (not yet ported to v1 hooks):
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "01-ai/Yi-Coder-9B-Chat",
            "deepseek-ai/DeepSeek-V2-Lite",
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nvidia/Llama-3.1-70B-Instruct-FP8",
        ]
    )
    all_approaches: List[str] = field(
        default_factory=lambda: [
            "fr",
            "naive",
            "cacheblend-20",
            "cacheblend-15",
            "cacheblend-10",
            "cacheblend-5",
            "cacheblend-1",
            "kvlink-64",
            "kvlink-32",
            "kvlink-16",
            "kvlink-8",
            "kvlink-4",
            "kvlink-2",
            "kvlink-1",
        ]
    )

    model_config: AutoConfig = field(init=False)

    seed: int = 42
    result_path: str = "results"

    @classmethod
    def get_configs_from_cli_args(cls) -> "EvalConfigs":
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--approach", type=str, required=True)
        args = parser.parse_args()
        return cls(**vars(args))

    def __post_init__(self):
        self._verify_init_args()
        self.result_path = os.path.join(
            self.result_path, self.dataset, self.model.split("/")[-1]
        )
        os.makedirs(self.result_path, exist_ok=True)

        self.model_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True
        )

    def _verify_init_args(self):
        assert self.model in self.all_models, (
            f"{self.model} not in {self.all_models}"
        )
        assert self.dataset in self.all_datasets, (
            f"{self.dataset} not in {self.all_datasets}"
        )
        assert self.approach in self.all_approaches, (
            f"{self.approach} not in {self.all_approaches}"
        )


class EvalEngine:
    """Evaluate one approach on one model/dataset pair."""

    def __init__(self, configs: EvalConfigs) -> None:
        self.configs = configs

    def run(self):
        logger.info(
            f"Evaluate \033[32m{self.configs.approach}\033[0m on"
            f" \033[32m{self.configs.model}\033[0m and"
            f" \033[32m{self.configs.dataset}\033[0m"
        )
        logger.info(
            f"Save the results to \033[32m{self.configs.result_path}\033[0m"
        )

        set_seed(self.configs.seed)

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.configs.model, add_bos_token=True, trust_remote_code=True
        )
        self.dataset: Data_set = str2class[self.configs.dataset](
            tokenizer=self.tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        self.dataset.save_dataset(self.configs.result_path)

        # NOTE(port): force the ported XFormers v1 backend so the
        # CacheBlend / KVLink hooks in the xformers impl actually fire.
        # enforce_eager=True is required because the hooks take a
        # non-standard control-flow path that torch.compile cannot trace.
        self.model: AutoModelForCausalLM = LLM(
            model=self.configs.model,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=32768,
            attention_backend="XFORMERS",
            quantization=(
                "modelopt" if "FP8" in self.configs.model.upper() else None
            ),
        )

        # Run inference, record results into the dataset.
        self.dataset = self.run_inference(
            self.configs, self.dataset, self.tokenizer, self.model
        )

        self.generate_presentation()

    def run_inference(
        self,
        configs: EvalConfigs,
        dataset: Data_set,
        tokenizer: AutoTokenizer,
        llm: LLM,
    ) -> Data_set:
        logger.info(
            "Run the inference. This might take a long time... Good luck"
        )

        results = defaultdict(list)

        for _, (system_prompts, mod_prompts, free_form_prompt) in tenumerate(
            dataset, desc="dataset", leave=True
        ):
            # Convert the prompts to token ids.
            system_token_ids: List[int] = tokenizer.apply_chat_template(
                [{"role": "user", "content": system_prompts}]
            )
            if system_token_ids[-1] == tokenizer.eos_token_id:
                system_token_ids = system_token_ids[:-1]
            mod_token_ids: List[List[int]] = [
                tokenizer.encode(mod_prompt) for mod_prompt in mod_prompts
            ]
            free_form_token_ids: List[int] = tokenizer.encode(
                free_form_prompt
            ) + [tokenizer.eos_token_id]
            token_ids: List[List[int]] = (
                [system_token_ids] + mod_token_ids + [free_form_token_ids]
            )

            input_ids = [
                _token_ids[i]
                for _token_ids in token_ids
                for i in range(len(_token_ids))
            ]

            # --- Step 1: collect old KVs per chunk ---
            logger.debug("Collecting old kvs")
            set_cache_fuse_flags(
                llm,
                collect=True,
                check=False,
                kvlink=None,
            )
            sampling_params = SamplingParams(temperature=0, max_tokens=1)

            old_kvs: list[list[torch.Tensor]] = []
            for i in range(len(token_ids)):
                llm.generate(
                    sampling_params=sampling_params,
                    prompt_token_ids=[token_ids[i]],
                    use_tqdm=False,
                )
                per_layer_kv = collect_hack_kv(llm)

                for j, pair in enumerate(per_layer_kv):
                    if pair[0] is None:
                        # linear_attention layer — still append a
                        # placeholder so indexing stays aligned.
                        if i == 0:
                            old_kvs.append([None, None])
                        continue
                    temp_k = pair[0].clone()
                    temp_v = pair[1].clone()
                    if i == 0:
                        if j >= len(old_kvs):
                            old_kvs.append([temp_k, temp_v])
                        else:
                            old_kvs[j] = [temp_k, temp_v]
                    else:
                        if old_kvs[j][0] is None:
                            old_kvs[j] = [temp_k, temp_v]
                        else:
                            old_kvs[j][0] = torch.cat(
                                (old_kvs[j][0], temp_k), dim=0
                            )
                            old_kvs[j][1] = torch.cat(
                                (old_kvs[j][1], temp_v), dim=0
                            )

                set_old_kvs(llm, old_kvs)

            # --- Step 2: second inference for performance evaluation ---
            logger.debug("Second inference for performance evaluation")
            start_offset = [0]
            for _token_ids in token_ids:
                start_offset.append(start_offset[-1] + len(_token_ids))

            if configs.approach == "naive":
                # Only recompute the last token.
                set_cache_fuse_flags(
                    llm,
                    kvlink=[start_offset[-1] - 1],
                    check=False,
                    collect=False,
                )
            elif "kvlink" in configs.approach:
                recomp_num = int(configs.approach.split("-")[-1])
                temp: list[int] = []
                for i in range(1, len(start_offset) - 1):
                    if i == len(start_offset) - 2:
                        temp += list(
                            range(start_offset[i], start_offset[i + 1])
                        )
                    else:
                        temp += list(
                            range(
                                start_offset[i], start_offset[i] + recomp_num
                            )
                        )
                set_cache_fuse_flags(
                    llm,
                    kvlink=temp,
                    check=False,
                    collect=False,
                )
            elif "cacheblend" in configs.approach:
                recomp_ratio = int(configs.approach.split("-")[-1]) / 100
                set_cache_fuse_flags(
                    llm,
                    kvlink=None,
                    check=True,
                    collect=False,
                    recomp_ratio=recomp_ratio,
                )
            elif configs.approach == "fr":
                set_cache_fuse_flags(
                    llm,
                    kvlink=None,
                    check=False,
                    collect=False,
                )
            else:
                raise ValueError(f"Invalid approach: {configs.approach}")

            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1024 if configs.dataset == "multi_news" else 32,
                skip_special_tokens=True,
            )
            output = llm.generate(
                sampling_params=sampling_params,
                prompt_token_ids=[input_ids],
                use_tqdm=False,
            )
            generated_text = output[0].outputs[0].text

            # Chat models (e.g. Qwen2.5 / Qwen3.5) emit a new turn after the
            # prompt. Extract just the assistant answer.
            assistant_header = "assistant\n"
            if assistant_header in generated_text:
                generated_text = generated_text.split(assistant_header, 1)[1]
            for marker in ("<|im_end|>", "<|im_start|>", "</s>"):
                if marker in generated_text:
                    generated_text = generated_text[
                        : generated_text.index(marker)
                    ]

            generated_text = generated_text.strip()
            results[f"output_{configs.approach}"].append(generated_text)
            results[f"TTFT_{configs.approach}"].append(
                output[0].metrics.first_token_time
                - output[0].metrics.first_scheduled_time
            )

            # Free CUDA memory accumulated during this dataset item.
            reset_layer_hack_kv(llm)
            del old_kvs
            torch.cuda.empty_cache()

        dataset.update(results)
        dataset.save_dataset(configs.result_path)
        return dataset

    def generate_presentation(self):
        self.dataset.calc_accuracy(
            self.configs.dataset, self.configs.approach
        )
        self.dataset.save_dataset(self.configs.result_path)

        accuracy_avg = np.mean(
            self.dataset.data[f"score_{self.configs.approach}"]
        )
        ttft_avg = np.mean(
            self.dataset.data[f"TTFT_{self.configs.approach}"]
        )
        logger.info(
            f"Average accuracy of {self.configs.approach}: {accuracy_avg:.3f}"
        )
        logger.info(
            f"Average TTFT of {self.configs.approach}: {ttft_avg:.2f} s"
        )


if __name__ == "__main__":
    configs = EvalConfigs.get_configs_from_cli_args()
    eval_engine = EvalEngine(configs)
    eval_engine.run()
