# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> Optional[Dict[str, Any]]:
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}
    if len(videos) != 0:
        return {"video": videos}
    return None


class vLLMRollout(BaseRollout):
    """vLLM rollout: 先生成 m 个 <plan>，再对每个 plan 生成 n/m 个 response，保证总数为 n。"""

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)

        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")
        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # ---------- helpers ----------

    def _extract_plan(self, text: str) -> str:
        m = re.search(r"<plan>.*?</plan>", text, re.DOTALL)
        if m:  # 正常包含闭合标签
            return m.group(0)
        m = re.search(r"<plan>.*", text, re.DOTALL)  # stop 截掉了 </plan> 的情况
        return (m.group(0) + "</plan>") if m else ""

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # ---------- main ----------

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # tensors
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        B = input_ids.size(0)

        # non-tensors
        non_tensor_batch = prompts.non_tensor_batch
        raw_prompt_ids_list: List[List[int]] = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        # ---- m 与 n 的对齐 ----
        m = int(getattr(self.config, "m_plans", 1))
        n_expected = int(getattr(self.config, "n", 1))
        if m < 1:
            raise ValueError("m_plans must be >= 1")
        if n_expected is None:
            raise ValueError("rollout.n must be set in config/meta_info")
        if n_expected % m != 0:
            raise ValueError(f"rollout.n={n_expected} must be divisible by m_plans={m}")
        n_per_plan = n_expected // m
        if n_per_plan < 1:
            raise ValueError("n_per_plan must be >= 1")

        # step1: 生成 m 个 plan
        vllm_inputs = [{"prompt_token_ids": list(raw_ids)} for raw_ids in raw_prompt_ids_list]
        with self.update_sampling_params(n=m, detokenize=True, stop=["</plan>"]):
            plan_comps: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
        plans_per_sample: List[List[str]] = []
        for comp in plan_comps:
            texts = [o.text for o in comp.outputs]
            plans = [self._extract_plan(t) for t in texts]
            plans_per_sample.append(plans)

        # step2: 针对 B*m 条「原 prompt + plan」生成 response
        new_inputs: List[Dict[str, List[int]]] = []
        plan_ids_list: List[List[int]] = []
        for raw_ids, plans in zip(raw_prompt_ids_list, plans_per_sample):
            for plan in plans:
                prefix_ids = list(raw_ids)
                plan_ids = self.tokenizer.encode(plan, add_special_tokens=False) if plan else []
                prefix_ids += plan_ids
                new_inputs.append({"prompt_token_ids": prefix_ids})
                plan_ids_list.append(plan_ids)
        with self.update_sampling_params(n=n_per_plan, detokenize=False):
            resp_comps: List[RequestOutput] = self.inference_engine.generate(
                prompts=new_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
        response_token_ids = [o.token_ids for comp in resp_comps for o in comp.outputs]
        plan_ids_list = [p for p in plan_ids_list for _ in range(n_per_plan)]

        # step3: 拼接 plan_ids + response
        merged_ids = [p_ids + r_ids for p_ids, r_ids in zip(plan_ids_list, response_token_ids)]
        responses = VF.pad_2d_list_to_length(
            merged_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        total_per_sample = m * n_per_plan
        total = B * total_per_sample

        # step4: 展开 prompts/masks/pos_ids
        prompts_rep = _repeat_interleave(input_ids, total_per_sample)
        attn_mask_rep = _repeat_interleave(attention_mask, total_per_sample)
        pos_ids_rep = _repeat_interleave(position_ids, total_per_sample)
        if batch_multi_modal_data is not None:
            batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, total_per_sample)
        seq_ids = torch.cat([prompts_rep, responses], dim=-1)

        Lr = responses.size(1)
        delta_pos = torch.arange(1, Lr + 1, device=pos_ids_rep.device)
        if pos_ids_rep.dim() == 2:
            delta_pos = delta_pos.view(1, -1).expand(total, -1)
        elif pos_ids_rep.dim() == 3:
            delta_pos = delta_pos.view(total, 1, -1).expand(total, 3, -1)
        resp_pos_ids = pos_ids_rep[..., -1:] + delta_pos
        pos_ids_out = torch.cat([pos_ids_rep, resp_pos_ids], dim=-1)

        resp_mask = VF.get_response_mask(
            response_ids=responses, eos_token_id=eos_token_id, dtype=attn_mask_rep.dtype
        )
        attn_mask_out = torch.cat([attn_mask_rep, resp_mask], dim=-1)

        # ✅ 分组键：父组/子组/实际 group_key
        device = input_ids.device
        prompt_key = torch.arange(B, device=device).repeat_interleave(total_per_sample)
        subgroup_key = torch.arange(B * m, device=device).repeat_interleave(n_per_plan)
        group_key = subgroup_key  # 用 plan 子组作为 GRPO 分组键

        out_batch = TensorDict(
            {
                "prompts": prompts_rep,
                "responses": responses,
                "input_ids": seq_ids,
                "attention_mask": attn_mask_out,
                "response_mask": resp_mask,
                "position_ids": pos_ids_out,
                "prompt_key": prompt_key,
                "subgroup_key": subgroup_key,
                "group_key": group_key,
            },
            batch_size=total,
        )

        nt_out: Dict[str, Any] = {
            "plan_texts": [plan for plans in plans_per_sample for plan in plans for _ in range(n_per_plan)],
            "plan_index": [j for _i, plans in enumerate(plans_per_sample) for j in range(len(plans)) for _ in range(n_per_plan)],
        }
        if batch_multi_modal_data is not None:
            nt_out["multi_modal_data"] = batch_multi_modal_data

        return DataProto(batch=out_batch, non_tensor_batch=nt_out, meta_info=prompts.meta_info)
