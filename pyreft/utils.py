import enum
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import evaluate
import numpy as np
import pyvene as pv
import torch
from pyvene.models.intervenable_base import IntervenableModelOutput
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
)

from .reft_model import ReftModel


@dataclass
class TokenSelectiveIntervenableModelOutput(IntervenableModelOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """

    original_outputs: Optional[Any] = None
    intervened_outputs: Optional[Any] = None
    collected_activations: Optional[Any] = None
    token_weights: Optional[torch.Tensor] = None


def compute_metrics_hf_train_loop(
    task: str,
    dataset_name: str,
    run_name: str,
    task_config: Dict,
    intervenable: pv.IntervenableModel,
    tokenizer: AutoTokenizer,
    dataloader,
    data_items: list,
    trigger_tokens: str,
    metric_key_prefix="eval",
    greedy_decoding=False,
    temperature=None,
    top_p=None,
    top_k=None,
) -> EvalLoopOutput:
    # switch the tokenizer mode first for generation tasks
    if task != "glue":
        tokenizer.padding_side = "left"  # switch padding side for collator
        num_beams = 4 if task in ["commonsense", "math"] and not greedy_decoding else 1

    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(dataloader, position=0, leave=True)
    all_preds = []
    all_labels = []
    losses = []

    device = intervenable.get_device()

    if "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        trigger_tokens = "assistant\n\n"

    with torch.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # [layers, batch_size, positions]
            if inputs["intervention_locations"].dim() == 3:
                intervention_locations = inputs["intervention_locations"].permute(
                    1, 0, 2
                )
            else:
                intervention_locations = None

            if task == "glue":
                _, cf_outputs = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    unit_locations={
                        "sources->base": (None, intervention_locations.tolist())
                    },
                    labels=inputs["labels"],
                )

                losses.append(cf_outputs.loss.detach())

                if dataset_name != "stsb":
                    preds = cf_outputs.logits.argmax(dim=-1)
                else:
                    preds = cf_outputs.logits.squeeze(dim=1)

                labels = inputs["labels"]
                all_preds += preds.tolist()
                all_labels += labels.tolist()

            else:
                # Handle left padding for generation tasks
                if intervention_locations is not None:
                    left_padding = (
                        inputs["input_ids"] == tokenizer.bos_token_id
                    ).nonzero(as_tuple=True)[1]
                    if left_padding.numel() > 0:
                        left_padding = left_padding.reshape(1, -1, 1).to(device)
                        intervention_locations += left_padding
                        intervention_locations -= 1
                    else:
                        print(
                            "Warning: No BOS token found, skipping left padding adjustment."
                        )

                    intervention_locations = intervention_locations.repeat_interleave(
                        num_beams, dim=1
                    ).tolist()
                else:
                    intervention_locations = 0

                generation_args = {
                    "base": {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    "unit_locations": {"sources->base": (None, intervention_locations)},
                    "intervene_on_prompt": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    "early_stopping": True,
                }

                if "generation_args" in task_config[task]:
                    generation_args.update(
                        task_config[task]["generation_args"][greedy_decoding]
                    )

                if "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path:
                    generation_args["eos_token_id"] = terminators

                if temperature is not None:
                    generation_args["temperature"] = temperature
                if top_p is not None:
                    generation_args["top_p"] = top_p
                if top_k is not None:
                    generation_args["top_k"] = top_k

                _, steered_response = intervenable.generate(**generation_args)
                actual_preds = tokenizer.batch_decode(
                    steered_response, skip_special_tokens=True
                )

                for id, pred in zip(inputs["id"].tolist(), actual_preds):
                    example = data_items[id]
                    try:
                        raw_generation = extract_output(pred, trigger_tokens)
                    except:
                        print("get not split based on trigger tokens: ", raw_generation)
                        raw_generation = "WRONG"

                    if task == "commonsense":
                        answer = example["answer"]
                        generation = raw_generation[:]
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    elif task == "math":
                        answer = example["answer"]
                        answer = answer.strip()
                        if not is_float(answer):
                            generation = extract_answer_letter(raw_generation)
                            if generation.strip() == answer.strip():
                                correct_count += 1
                        else:
                            generation = extract_answer_number(raw_generation)
                            if abs(float(answer) - generation) <= 0.001:
                                correct_count += 1
                    elif task == "gsm8k":
                        answer = example["answer"].split("####")[-1].strip()
                        generation = extract_answer_number(raw_generation)
                        if (
                            abs(float(extract_answer_number(answer)) - generation)
                            <= 0.001
                        ):
                            correct_count += 1

                    total_count += 1
                    if task not in [
                        "alpaca",
                        "instruct",
                        "ultrafeedback",
                        "ultrafeedback_pair",
                    ]:
                        metric_str = round(correct_count / total_count, 3)
                        eval_iterator.set_postfix({"em": metric_str})
                        instruction = (
                            example["question"]
                            if task == "gsm8k"
                            else example["instruction"]
                        )
                        generations += [
                            {
                                "instruction": instruction,
                                "raw_generation": raw_generation,
                                "generation": generation,
                                "answer": answer,
                            }
                        ]
                    else:
                        generations += [
                            {
                                "instruction": example["instruction"],
                                "output": raw_generation,
                                "dataset": dataset_name,
                                "generator": run_name,
                            }
                        ]

    # Compute final metrics
    metrics = {}
    if task == "glue":
        metric = evaluate.load("glue", dataset_name)
        metrics = metric.compute(predictions=all_preds, references=all_labels)
        if len(metrics) > 1:
            metrics["combined_score"] = np.mean(list(metrics.values())).item()
    elif task not in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"]:
        metrics = {f"{metric_key_prefix}/{dataset_name}": correct_count / total_count}

    if losses:
        metrics[f"{metric_key_prefix}_loss"] = torch.stack(losses).mean().item()

    return EvalLoopOutput(
        predictions=all_preds if task == "glue" else generations,
        label_ids=all_labels if task == "glue" else None,
        metrics=metrics,
        num_samples=len(all_preds) if task == "glue" else total_count,
    )


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance.

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


def extract_output(pred, trigger=""):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ""
    output = pred[start + len(trigger) :].lstrip()  # left strip any whitespaces
    return output


class ReftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in REFT.

    Supported REFT types:
    - LOREFT
    """

    LOREFT = "LOREFT"
    NLOREFT = "NOREFT"
    # Add yours here!


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by REFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - CAUSAL_LM: Causal language modeling.
    """

    SEQ_CLS = "SEQ_CLS"
    CAUSAL_LM = "CAUSAL_LM"


def get_reft_model(
    model,
    reft_config,
    set_device=True,
    disable_model_grads=True,
    instance_cls=ReftModel,
    **kwargs,
):
    """
    Create an instance of ReFT model.
    """
    reft_model = instance_cls(reft_config, model, **kwargs)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()
    return reft_model
