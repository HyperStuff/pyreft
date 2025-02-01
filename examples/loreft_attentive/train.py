import copy
import datetime
import json
import os

import evaluate
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pyreft.interventions import QuasiProjectiveReftIntervention
from task_config import task_config
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

from examples.loreft_attentive.compute_metrics import compute_metrics
from examples.loreft_attentive.dataset import (
    LoReftGLUEDataset,
    LoReftSupervisedDataset,
)
from pyreft import (
    ConsreftIntervention,  # constant bias only  # constant bias only
    DireftIntervention,  # direct edit reft  # direct edit reft
    LobireftIntervention,  # low-rank bitfit reft  # low-rank bitfit reft
    LoreftIntervention,
    NodireftIntervention,  # remove ortho + direct edit reft <- this is like LoRA on time-step  # remove ortho + direct edit reft <- this is like LoRA on time-step
    NoreftIntervention,  # remove ortho.  # remove ortho.
    ReftConfig,
    ReftDataCollator,
    TaskType,
    TokenSelectiveLoreftIntervention,
    get_reft_model,
)
from pyreft.reft_model import AutomatedReftModel
from pyreft.reft_trainer import (
    ReftTrainingArguments,
    TokenSelectiveReftTrainer,
    TokenSelectiveReftTrainerForSequenceClassification,
)

try:
    # This library is our indicator that the required installs
    # need to be done.
    import peft

    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output",
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}
intervention_mapping = {
    "NoreftIntervention": NoreftIntervention,
    "LoreftIntervention": LoreftIntervention,
    "ConsreftIntervention": ConsreftIntervention,
    "LobireftIntervention": LobireftIntervention,
    "DireftIntervention": DireftIntervention,
    "NodireftIntervention": NodireftIntervention,
    "TokenSelectiveLoreftIntervention": TokenSelectiveLoreftIntervention,
    "QuasiProjectiveIntervention": QuasiProjectiveReftIntervention,
}


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="loreft_attentive",
)
def finetune(cfg: DictConfig):
    """Generic Representation Finetuning."""  # noqa: D401
    assert cfg.task.name in {
        "commonsense",
        "math",
        "alpaca",
        "instruct",
        "ultrafeedback",
        "glue",
        "gsm8k",
        "ultrafeedback_pair",
    }

    dtype = dtype_mapping[cfg.model.dtype] if device != "mps" else torch.float

    # store/log run details
    print(
        f"task: {cfg.task.name}, model: {cfg.model.name}, "
        f"intervention_type: {cfg.intervention.type}, "
        f"layers: {cfg.intervention.layers}, rank: {cfg.intervention.low_rank_dimension}, "
        f"position: {cfg.intervention.position}, epoch: {cfg.training.epochs}, "
        f"train_on_inputs: {cfg.task.train_on_inputs}, "
        f"max_length: {cfg.model.max_length}, allow_cls_grad: {cfg.task.allow_cls_grad}",
    )

    set_seed(cfg.training.seed)

    model_name = cfg.model.name
    model_str = model_name.split("/")[-1]

    train_dataset_index = (
        cfg.task.train_dataset.index(cfg.training.train_dataset_key)
        if cfg.training.train_dataset_key
        else 0
    )
    train_dataset_str = cfg.task.train_dataset[train_dataset_index]

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if cfg.task.train_dataset is not None:
        run_name = f"{model_str}.{cfg.task.name}.{train_dataset_str}.{cfg.task.test_split}.{now}"
    else:
        run_name = f"{model_str}.{cfg.task.name}.{now}"

    # which layers to intervene on
    if cfg.intervention.layers.strip() == "":
        layers = []
    elif cfg.intervention.layers != "all":
        layers = [int(l) for l in cfg.intervention.layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model_name)
        layers = [l for l in range(temp_config.num_hidden_layers)]

    if cfg.lora.layers.strip() == "":
        lora_layers = []
    elif cfg.lora.layers != "all":
        lora_layers = [int(l) for l in cfg.lora.layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model_name)
        lora_layers = [l for l in range(temp_config.num_hidden_layers)]

    unique_layers = copy.deepcopy(layers)
    if "+" in cfg.intervention.position and not cfg.intervention.share_weights:
        layers += layers

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=cfg.model.max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        need_resize = False

    # load dataset splits
    assert cfg.task.name in task_config, f"Unrecognized task: {cfg.task.name}"
    if cfg.task.name == "glue":
        eval_datasets_names = cfg.task.train_dataset
    else:
        eval_datasets_names = cfg.task.eval_dataset

    ReftDataset = (
        LoReftGLUEDataset if cfg.task.name == "glue" else LoReftSupervisedDataset
    )

    train_dataset = ReftDataset(
        cfg.task.name,
        train_dataset_str
        if cfg.task.name == "glue" or cfg.task.name == "ultrafeedback_pair"
        else (
            os.path.join(cfg.task.data_dir, train_dataset_str)
            if cfg.task.data_dir is not None
            else train_dataset_str
        ),
        tokenizer,
        data_split="train",
        seed=cfg.training.seed,
        max_n_example=cfg.task.max_n_train_example,
        num_interventions=len(layers),
        position=cfg.intervention.position,
        share_weights=cfg.intervention.share_weights,
        test_split=cfg.task.test_split,
    )

    all_eval_datasets = {}
    for eval_dataset in eval_datasets_names:
        test_splits = cfg.task.test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = ReftDataset(
                cfg.task.name,
                eval_dataset
                if cfg.task.name == "glue"
                else os.path.join(cfg.task.data_dir, eval_dataset),
                tokenizer,
                data_split=split,
                seed=cfg.training.seed,
                max_n_example=cfg.task.max_n_eval_example,
                num_interventions=len(layers),
                position=cfg.intervention.position,
                share_weights=cfg.intervention.share_weights,
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    # Handle GLUE specific dataset splitting
    if cfg.task.name == "glue":
        to_split_eval_datasets = eval_datasets[train_dataset_str][cfg.task.test_split][
            0
        ]
        if len(to_split_eval_datasets) > 5000:
            in_train_n_eval_sample = 1000
        else:
            in_train_n_eval_sample = len(to_split_eval_datasets) // 2

        new_splits = torch.utils.data.random_split(
            to_split_eval_datasets,
            [
                len(to_split_eval_datasets) - in_train_n_eval_sample,
                in_train_n_eval_sample,
            ],
        )

        in_test_eval_datasets, in_train_eval_datasets = new_splits[0], new_splits[1]
        eval_datasets[train_dataset_str][cfg.task.test_split][0] = in_test_eval_datasets
        print("GLUE validation split (in training): ", len(in_train_eval_datasets))
        print(
            "GLUE validation split (testing): ",
            len(eval_datasets[train_dataset_str][cfg.task.test_split][0]),
        )

        is_regression = train_dataset_str == "stsb"
        metric = evaluate.load("glue", train_dataset_str)

        def in_training_compute_metrics(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

    else:
        pass

    # Initialize model based on task type
    if cfg.task.name in classification_tasks:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=train_dataset.num_labels,
            torch_dtype=dtype if dtype != "float8" else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if dtype != "float8" else None,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.padding_token = tokenizer.pad_token

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # Handle LoRA if enabled
    if cfg.lora.use_lora:
        if not is_peft_available:
            raise ModuleNotFoundError("peft is required for LoRA support")
        from peft import LoraConfig, get_peft_model

        print("WARNING: enabling lora for finetuning...")
        lora_modules = [m for m in cfg.lora.modules.split(";")]
        peft_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            target_modules=lora_modules,
            layers_to_transform=lora_layers,
            use_rslora=False,
            lora_dropout=cfg.training.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # Initialize ReFT config and model
    intervention_type = intervention_mapping[cfg.intervention.type]
    model_arch = model.config.architectures[0].lower()
    intervention_dtype = torch.bfloat16 if isinstance(dtype, str) else dtype

    intervention_obj = intervention_type(
        embed_dim=model.config.hidden_size,
        dropout=cfg.training.dropout,
        dtype=intervention_dtype,
        device=device,
        **OmegaConf.to_container(cfg.intervention),
    )

    if model_arch in residual_stream_component_mapping:
        representations = [
            {
                "component": residual_stream_component_mapping[model_arch] % l,
                "low_rank_dimension": cfg.intervention.low_rank_dimension,
                "intervention": intervention_obj,
            }
            for l in layers
        ]
        task_type = TaskType.SEQ_CLS
    else:
        representations = [
            {
                "layer": l,
                "component": f"base_model.model.model.layers[{l}].output"
                if cfg.lora.use_lora
                else "block_output",
                "low_rank_dimension": cfg.intervention.low_rank_dimension,
                "intervention": intervention_obj,
            }
            for l in layers
        ]
        task_type = TaskType.CAUSAL_LM

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(
        model.to(device),
        reft_config,
        set_device=True,
        instance_cls=AutomatedReftModel,
        do_token_selective_intervention=cfg.model.do_token_selection,
        embed_dim=cfg.model.embed_dim,
        start_temperature=cfg.model.start_temperature,
        end_temperature=cfg.model.end_temperature,
        max_steps=(cfg.training.epochs * len(train_dataset) // cfg.training.batch_size),
        dtype=dtype,
        scheduler=cfg.model.scheduler,
        discretization_strategy=cfg.model.discretization_strategy,
    )

    if cfg.lora.use_lora:
        reft_model.model.enable_adapter_layers()
    reft_model.print_trainable_parameters()

    # for GLUE tasks, we enable gradients on the classifier head.
    if cfg.task.name == "glue" and cfg.task.allow_cls_grad:
        for param in reft_model.model.classifier.parameters():
            param.requires_grad = True

    reft_model.model.train()
    n_params = reft_model.count_parameters(include_model=False)
    n_params_with_model = reft_model.count_parameters(include_model=True)

    # Create wandb run name with intervention type and base model
    base_model_name = model_name.split("/")[-1]
    intervention_type_name = cfg.intervention.type.replace("Intervention", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{intervention_type_name}_{base_model_name}_{timestamp}"

    # start wandb logging
    if cfg.logging.is_wandb:
        run = wandb.init(
            project=cfg.logging.wandb_proj,
            entity=cfg.logging.wandb_entity,
            name=run_name,
        )
        run.summary.update(OmegaConf.to_container(cfg, resolve=False))
        wandb.log(
            {
                "train/n_params": n_params,
                "train/n_params_with_model": n_params_with_model,
            },
        )

    # select collator based on the type
    if cfg.task.name in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest",
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    entropy_loss_mode = None
    if cfg.model.discretization_strategy == "binary_entropy":
        entropy_loss_mode = "binary"
    elif cfg.model.discretization_strategy == "single_entropy":
        entropy_loss_mode = "single"

    # training args
    training_args = ReftTrainingArguments(
        output_dir=f"{cfg.logging.output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        evaluation_strategy=cfg.training.eval_strategy
        if cfg.task.name == "glue"
        else "no",
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.eval_strategy if cfg.task.name == "glue" else "no",
        metric_for_best_model=cfg.task.metric_for_best_model
        if cfg.task.name == "glue"
        else None,
        load_best_model_at_end=True if cfg.task.name == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1,
        logging_steps=cfg.training.logging_steps,
        lr_scheduler_type=cfg.training.schedule,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        task_config=task_config,
        optim="adamw_torch",
        weight_decay=cfg.training.weight_decay,
        report_to="wandb" if cfg.logging.is_wandb else "none",
        use_cpu=False if device in ["cuda", "mps"] else True,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        entropy_loss_mode=entropy_loss_mode,
        entropy_loss_weight=cfg.training.entropy_loss_weight,
    )

    # make trainer
    trainer_cls = (
        TokenSelectiveReftTrainerForSequenceClassification
        if cfg.task.name in classification_tasks
        else TokenSelectiveReftTrainer
    )
    trainer = trainer_cls(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=in_train_eval_datasets if cfg.task.name == "glue" else None,
        data_collator=data_collator,
        compute_metrics=in_training_compute_metrics
        if cfg.task.name == "glue"
        else None,
    )
    trainer.train()

    # dump config
    config_dict = OmegaConf.to_container(cfg, resolve=False)
    config_dict = OmegaConf.to_container(cfg, resolve=False)
    config_dict["n_params"] = n_params
    json_file_name = f"{cfg.logging.output_dir}/{run_name}/config.json"
    with open(json_file_name, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    # save model
    if cfg.logging.save_model:
        reft_model.save(f"{cfg.logging.output_dir}/{run_name}")

    # ensure everything is in eval mode
    reft_model.model.eval()
    for v in reft_model.interventions.values():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            generations, stats = compute_metrics(
                cfg.task.name,
                dataset_name,
                reft_model,
                tokenizer,
                eval_dataset,
                data_items,
                train_dataset.trigger_tokens,
                run_name,
                cfg.training.eval_batch_size,
                data_collator if cfg.task.name in classification_tasks else None,
                split,
                cfg.generation.greedy_decoding,
                cfg.generation.temperature,
                cfg.generation.top_p,
                cfg.generation.top_k,
                device=device,
            )

            eval_results.update(stats)
            if cfg.logging.is_wandb:
                wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = f"{cfg.logging.output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, "w") as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{cfg.logging.output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, "w") as json_file:
        json.dump(eval_results, json_file, indent=4)

    if cfg.logging.is_wandb:
        wandb.finish()

    print(f"Training results can be found in {cfg.logging.output_dir}/{run_name}")


if __name__ == "__main__":
    finetune()
