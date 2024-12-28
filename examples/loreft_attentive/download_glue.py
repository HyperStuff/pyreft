import json
import os
from pathlib import Path

from datasets import load_dataset

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def process_and_save_split(dataset, split, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Convert to list and save to JSON
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w") as f:
        json.dump(dataset[split].to_list(), f, indent=2)
    print(f"Saved {split} split to {output_path}")


# Process each split
for subtask in glue_task_to_keys.keys():
    dataset = load_dataset("glue", subtask)
    output_directory = Path(f"../dataset/{subtask}")
    output_directory.mkdir(exist_ok=True, parents=True)
    for split in dataset.keys():
        process_and_save_split(dataset, split, output_directory)
