import copy
import datetime
import os

import hydra
import torch
import wandb
from compute_metrics import compute_metrics
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from omegaconf import DictConfig
from pyreft.interventions import TokenSelectiveLoreftIntervention
from task_config import task_config
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from pyreft import (
    ConsreftIntervention,  # constant bias only
    DireftIntervention,  # direct edit reft
    LobireftIntervention,  # low-rank bitfit reft
    LoreftIntervention,
    NodireftIntervention,  # remove ortho + direct edit reft <- this is like LoRA on time-step
    NoreftIntervention,  # remove ortho.
    ReftConfig,
    ReftDataCollator,
    ReftTrainerForCausalLM,
    ReftTrainerForSequenceClassification,
    get_reft_model,
)

try:
    # This library is our indicator that the required installs
    # need to be done.
    import peft

    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
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
}


@hydra.main(version_base=None, config_path="conf", config_name="loreft_attentive")
def finetune(cfg: DictConfig):
    """
    Generic Representation Finetuning.
    """
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

    dtype = dtype_mapping[cfg.model.dtype]

    # store/log run details
    print(
        f"task: {cfg.task.name}, model: {cfg.model.name}, "
        f"intervention_type: {cfg.intervention.type}, "
        f"layers: {cfg.intervention.layers}, rank: {cfg.intervention.rank}, "
        f"position: {cfg.intervention.position}, epoch: {cfg.training.epochs}, "
        f"train_on_inputs: {cfg.task.train_on_inputs}, "
        f"max_length: {cfg.model.max_length}, allow_cls_grad: {cfg.task.allow_cls_grad}"
    )

    set_seed(cfg.training.seed)

    model_name = cfg.model.name
    model_str = model_name.split("/")[-1]
    train_dataset_str = cfg.task.train_dataset
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=cfg.model.max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load dataset splits
    assert cfg.task.name in task_config, f"Unrecognized task: {cfg.task.name}"
    train_datasets = (
        task_config[cfg.task.name]["train_datasets"]
        if cfg.task.train_dataset is None
        else [cfg.task.train_dataset]
    )
    if cfg.task.name == "glue":
        eval_datasets = [cfg.task.train_dataset]
    else:
        eval_datasets = (
            task_config[cfg.task.name]["eval_datasets"]
            if cfg.task.eval_dataset is None
            else [cfg.task.eval_dataset]
        )

    ReftDataset = (
        LoReftGLUEDataset if cfg.task.name == "glue" else LoReftSupervisedDataset
    )
    train_dataset = ReftDataset(
        cfg.task.name,
        train_datasets[0]
        if cfg.task.name == "glue" or cfg.task.name == "ultrafeedback_pair"
        else (
            os.path.join(cfg.task.data_dir, train_datasets[0])
            if cfg.task.data_dir is not None
            else train_datasets[0]
        ),
        tokenizer,
        data_split="train",
        seed=cfg.training.seed,
        max_n_example=cfg.task.max_n_train_example,
        **{
            "num_interventions": len(layers),
            "position": cfg.intervention.position,
            "share_weights": cfg.intervention.share_weights,
            "test_split": cfg.task.test_split,
        },
    )

    eval_dataset = ReftDataset(
        cfg.task.name,
        eval_datasets[0]
        if cfg.task.name == "glue" or cfg.task.name == "ultrafeedback_pair"
        else (
            os.path.join(cfg.task.data_dir, eval_datasets[0])
            if cfg.task.data_dir is not None
            else eval_datasets[0]
        ),
        tokenizer,
        data_split=cfg.task.test_split,
        seed=cfg.training.seed,
        max_n_example=cfg.task.max_n_eval_example,
        **{
            "num_interventions": len(layers),
            "position": cfg.intervention.position,
            "share_weights": cfg.intervention.share_weights,
            "test_split": cfg.task.test_split,
        },
    )

    # initialize model
    model_config = AutoConfig.from_pretrained(model_name)
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

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # initialize reft config
    reft_config = ReftConfig(
        intervention_type=intervention_mapping[cfg.intervention.type],
        layers=layers,
        rank=cfg.intervention.rank,
        position=cfg.intervention.position,
        act_fn=cfg.intervention.act_fn,
        add_bias=cfg.intervention.add_bias,
        share_weights=cfg.intervention.share_weights,
        use_lora=cfg.lora.use_lora,
        disable_reft=cfg.lora.disable_reft,
        lora_rank=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_modules=cfg.lora.modules,
        lora_layers=lora_layers,
    )

    # initialize reft model
    model = get_reft_model(model, reft_config)
    model = model.to(device)

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # initialize data collator
    if cfg.task.name in classification_tasks:
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        data_collator = ReftDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True,
        )

    # initialize training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.logging.output_dir, run_name),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=1.0,
        num_train_epochs=cfg.training.epochs,
        lr_scheduler_type=cfg.training.schedule,
        warmup_ratio=cfg.training.warmup_ratio,
        log_level="error",
        logging_strategy="steps",
        logging_steps=cfg.training.logging_steps,
        save_strategy="no",
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        metric_for_best_model=cfg.task.metric_for_best_model,
        greater_is_better=True,
        report_to="wandb" if cfg.logging.is_wandb else "none",
        run_name=run_name if cfg.logging.is_wandb else None,
    )

    # initialize trainer
    if cfg.task.name in classification_tasks:
        trainer = ReftTrainerForSequenceClassification(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )
    else:
        trainer = ReftTrainerForCausalLM(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )

    # train
    if cfg.logging.is_wandb:
        wandb.init(
            project=cfg.logging.wandb_proj,
            name=cfg.logging.wandb_name,
            dir=cfg.logging.wandb_dir,
            config={
                "model": cfg.model.name,
                "task": cfg.task.name,
                "intervention_type": cfg.intervention.type,
                "layers": cfg.intervention.layers,
                "rank": cfg.intervention.rank,
                "position": cfg.intervention.position,
                "act_fn": cfg.intervention.act_fn,
                "add_bias": cfg.intervention.add_bias,
                "share_weights": cfg.intervention.share_weights,
                "train_dataset": cfg.task.train_dataset,
                "eval_dataset": cfg.task.eval_dataset,
                "test_split": cfg.task.test_split,
                "train_on_inputs": cfg.task.train_on_inputs,
                "max_length": cfg.model.max_length,
                "allow_cls_grad": cfg.task.allow_cls_grad,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "epochs": cfg.training.epochs,
                "warmup_ratio": cfg.training.warmup_ratio,
                "weight_decay": cfg.training.weight_decay,
                "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            },
        )

    trainer.train()

    if cfg.logging.save_model:
        trainer.save_model()

    if cfg.logging.is_wandb:
        wandb.finish()


if __name__ == "__main__":
    finetune()
