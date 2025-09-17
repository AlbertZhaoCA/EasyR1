# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<plan></plan>.*<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0



def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    plan_weight: float = 0.1,
) -> List[Dict[str, float]]:
    """
    约定：每个 reward_input 可选携带以下任一键来标识“同一个 plan 的样本”：
      - 'plan_id'        : 任意可哈希对象（int/tuple/str等），代表唯一的plan键
      - 'subgroup_key'   : 训练侧 vLLM rollout 里给的子组键（B*m唯一）
      - ('prompt_key','plan_index') 二元组：同一prompt下的第几个plan

    overall = (1 - format_weight - plan_weight)*accuracy
              + format_weight*format
              + plan_weight*plan_acc_of_this_plan
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    if format_weight < 0 or plan_weight < 0 or (format_weight + plan_weight) > 1:
        raise ValueError("format_weight and plan_weight must be >=0 and sum <= 1.")

    fmt_list, acc_list, plan_keys = [], [], []
    cleaned_responses = []

    for ri in reward_inputs:
        resp = re.sub(r"\s*(<|>|/)\s*", r"\1", ri["response"])
        cleaned_responses.append(resp)

        fmt = format_reward(resp)
        acc = accuracy_reward(resp, ri["ground_truth"])
        fmt_list.append(fmt)
        acc_list.append(acc)

        plan_key = ri.get("plan_id", None)
        if plan_key is None:
            plan_key = ri.get("subgroup_key", None)
        if plan_key is None:
            pk = ri.get("prompt_key", None)
            pi = ri.get("plan_index", None)
            if pk is not None and pi is not None:
                plan_key = (int(pk), int(pi))

        plan_keys.append(plan_key)

    plan_sum = defaultdict(float)
    plan_cnt = defaultdict(int)
    for k, acc in zip(plan_keys, acc_list):
        if k is not None:
            plan_sum[k] += acc
            plan_cnt[k] += 1
    plan_acc_map = {k: (plan_sum[k] / plan_cnt[k]) for k in plan_sum.keys()}

    scores: List[Dict[str, float]] = []
    for fmt, acc, k in zip(fmt_list, acc_list, plan_keys):
        plan_acc = plan_acc_map.get(k, acc)
        overall = (1.0 - format_weight - plan_weight) * acc + format_weight * fmt + plan_weight * plan_acc
        scores.append(
            {
                "overall": float(overall),
                "format": float(fmt),
                "accuracy": float(acc),
                "plan": float(plan_acc), 
            }
        )
    return scores

