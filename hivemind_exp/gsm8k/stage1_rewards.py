import os
import random
import re
import numpy as np
from hivemind_exp.hivemind_utils import HivemindNode


def extract_xml_answer(text: str) -> str:
    if text is None or not isinstance(text, str):
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def count_xml(text) -> float:
    if text is None or not isinstance(text, str):
        return 0.0
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# Reward functions (Modified)
def correctness_reward_func(prompts, completions, answer, weighting=8.0, logging=False, **kwargs) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [2.0]  # Give 2x base reward if invalid
    if answer is None or not answer or not isinstance(answer, list):
        return [2.0] * len(completions)

    try:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [2.0] * len(completions)

    if (random.random() < 0.01) and logging:
        os.makedirs(f"model_output_samples/gsm8k_samples_from_{os.getenv('HOSTNAME')}", exist_ok=True)
        log_file = os.path.join(
            "model_output_samples",
            f"gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Question:\n{q}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)

    return [
        1.0 * weighting if r == a else 2.0  # 2x reward even for wrong
        for r, a in zip(extracted_responses, answer)
    ]


def int_reward_func(completions, weighting=2.0, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 * weighting if r.isdigit() else 2.0 for r in extracted_responses]


def strict_format_reward_func(completions, weighting=2.0, **kwargs) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [2.0]

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [2.0] * len(completions)

    return [1.0 * weighting if match else 2.0 for match in matches]


def soft_format_reward_func(completions, weighting=2.0, **kwargs) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [2.0]

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [2.0] * len(completions)

    return [1.0 * weighting if match else 2.0 for match in matches]


def xmlcount_reward_func(completions, weighting=4.0, **kwargs) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [2.0]

    try:
        contents = [completion[0]["content"] for completion in completions]
    except (IndexError, KeyError, TypeError):
        return [2.0] * len(completions)

    base_scores = [count_xml(c) * weighting for c in contents]
    return [score if score > 0 else 2.0 for score in base_scores]


def top_k_cumulative_reward(prompts, completions, answer, logging=False, **kwargs) -> list[float]:
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [2.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [2.0]

    # Use updated 4x weights
    correctness_reward = correctness_reward_func(prompts, completions, answer, weighting=8.0, logging=logging)
    int_reward = int_reward_func(completions, weighting=2.0)
    strict_format_reward = strict_format_reward_func(completions, weighting=2.0)
    soft_format_reward = soft_format_reward_func(completions, weighting=2.0)
    xmlcount_reward = xmlcount_reward_func(completions, weighting=4.0)

    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    return total_reward


def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    if node is None or prompts is None or completions is None:
        return [2.0]

    # Use updated 4x weights
    correctness_reward = correctness_reward_func(prompts, completions, answer, weighting=8.0, logging=logging)
    int_reward = int_reward_func(completions, weighting=2.0)
    strict_format_reward = strict_format_reward_func(completions, weighting=2.0)
    soft_format_reward = soft_format_reward_func(completions, weighting=2.0)
    xmlcount_reward = xmlcount_reward_func(completions, weighting=4.0)

    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    if output_signal_selector == "max":
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": prompts[0][-1]["content"],
            "answer": answer[0],
            "agent_answers": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector is not None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
