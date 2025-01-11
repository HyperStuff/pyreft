from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pyvene as pv
import torch
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollator,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import (
    EvalPrediction,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ReftDataCollator:
    """Collate examples for ReFT."""

    data_collator: DataCollator # type: ignore

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][
            ..., :max_seq_length,
        ]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn,
    )


@dataclass
class ReftTrainingArguments(TrainingArguments):
    token_sparsity_loss_weight: float = 0.0
    token_binary_loss_weight: float = 0.0
    # New arguments for evaluation
    task: str = "glue"
    dataset_name: str = "mnli"
    trigger_tokens: str = ""
    greedy_decoding: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    task_config: Optional[Dict] = field(default_factory=dict)


class ReftTrainer(Trainer):
    def save_model(self, output_dir: str, *, _internal_call: bool = False) -> None:
            if not Path(output_dir).exists():
                Path(output_dir).mkdir(parents=True)
            self.model.save_intervention(
                save_directory=f"{output_dir}/intervenable_model", include_model=True,
            )

    def _load_best_model(self):
        logger.warning(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).",
        )
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", include_model=True,
        )

    def compute_loss(
        self, intervenable: pv.IntervenableModel, inputs, return_outputs=False,
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations = {
                    "sources->base": (
                        None,
                        inputs["intervention_locations"].permute(
                            1, 0, 2).tolist(),
                    ),
                }
            else:
                # this is dummy for lora only baseline
                unit_locations = {"sources->base": (None, 0)}
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs
            else None,
        )
        # return
        output = cf_outputs
        if cf_outputs is None:
            output = base_outputs  # in case of lora only training

        return (output, output) if return_outputs else output.loss


class TokenSelectiveReftTrainer(ReftTrainer):
    def compute_loss(
        self, intervenable: pv.IntervenableModel, inputs, return_outputs=False,
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations = {
                    "sources->base": (
                        None,
                        inputs["intervention_locations"].permute(
                            1, 0, 2).tolist(),
                    ),
                }
            else:
                # this is dummy for lora only baseline
                unit_locations = {"sources->base": (None, 0)}
        _intervened_out = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs
            else None,
        )

        if len(_intervened_out) == 2:
            base_outputs, cf_outputs = _intervened_out
            token_weights = None
        else:
            base_outputs, cf_outputs, token_weights = _intervened_out

        if token_weights is not None:
            # Compute sparsity loss if enabled
            sparsity_loss = self.args.token_sparsity_loss_weight * torch.mean(
                torch.sum(token_weights, dim=-1),
            )
            # Compute binary loss if enabled
            binary_loss = (
                self.args.token_binary_loss_weight
                * (token_weights * (1 - token_weights)).mean()
            )

            # Log metrics about token weights
            self.log(
                {
                    "train/sparsity_loss": sparsity_loss.cpu().item(),
                    "train/binary_loss": binary_loss.cpu().item(),
                    "train/mean_token_weight": token_weights.mean().cpu().item(),
                    "train/token_weight_l0": token_weights.sum(dim=-1).float().mean().item(),
                    "train/token_weight_max": token_weights.max().item(),
                    "train/token_weight_min": token_weights.min().item(),
                    "train/token_weight_temperatures": intervenable.selection_module.temperature.item(),
                },
            )

        # return
        output = cf_outputs
        if cf_outputs is None:
            output = base_outputs  # in case of lora only training

        if token_weights is not None:
            output.loss += sparsity_loss + binary_loss

        return (output, output) if return_outputs else output.loss


class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(
            self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True,
        )


class TokenSelectiveReftTrainerForCausalLM(TokenSelectiveReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(
            self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True,
        )


class ReftTrainerForSequenceClassification(ReftTrainer):
    def compute_loss(
        self, intervenable: pv.IntervenableModel, inputs, return_outputs=False,
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations = {
                "sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                ),
            }

        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs
            else None,
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type

        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(),
                                labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model.model.num_labels), labels.view(-1),
            )
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return (loss, cf_outputs) if return_outputs else loss

    def evaluate(
        self,
        ignore_keys,
    ):
        # ensure everything is in eval mode
        self.model.model.eval()
        for k, v in self.model.interventions.items():
            _ = v[0].eval()

        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model

        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False,
        )

        logger.info("***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())

                # [layers, batch_size, positions]
                intervention_locations = (
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )
                _, cf_outputs = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    unit_locations={
                        "sources->base": (None, intervention_locations)},
                )

                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels),
        )
        metrics = denumpify_detensorize(metrics)

        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


class TokenSelectiveReftTrainerForSequenceClassification(TokenSelectiveReftTrainer, ReftTrainerForSequenceClassification):
    def compute_loss(
        self, intervenable: pv.IntervenableModel, inputs, return_outputs=False,
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations = {
                "sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                ),
            }

        _intervened_out = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs
            else None,
        )

        if len(_intervened_out) == 2:
            _, cf_outputs = _intervened_out
            token_weights = None
        else:
            _, cf_outputs, token_weights = _intervened_out

        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type

        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(),
                                labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model.model.num_labels), labels.view(-1),
            )
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if token_weights is not None:
            # Compute sparsity loss if enabled
            sparsity_loss = self.args.token_sparsity_loss_weight * torch.mean(
                torch.sum(token_weights, dim=-1),
            )
            # Compute binary loss if enabled
            binary_loss = (
                self.args.token_binary_loss_weight
                * (token_weights * (1 - token_weights)).mean()
            )

            # Log metrics about token weights
            self.log(
                {
                    "train/sparsity_loss": sparsity_loss.cpu().item(),
                    "train/binary_loss": binary_loss.cpu().item(),
                    "train/mean_token_weight": token_weights.mean().cpu().item(),
                    "train/token_weight_l0": token_weights.sum(dim=-1).float().mean().item(),
                    "train/token_weight_max": token_weights.max().item(),
                    "train/token_weight_min": token_weights.min().item(),
                    "train/token_weight_temperatures": intervenable.selection_module.temperature.item(),
                },
            )

            loss += sparsity_loss + binary_loss

        return (loss, cf_outputs) if return_outputs else loss

    def evaluate(
        self,
        ignore_keys,
    ):
        # ensure everything is in eval mode
        self.model.model.eval()
        for k, v in self.model.interventions.items():
            _ = v[0].eval()

        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model

        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False,
        )

        logger.info("***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())

                # [layers, batch_size, positions]
                intervention_locations = (
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )
                _, cf_outputs, token_weights = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    unit_locations={
                        "sources->base": (None, intervention_locations)},
                )

                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels),
        )
        metrics = denumpify_detensorize(metrics)

        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
