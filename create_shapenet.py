#!/usr/bin/env python
# coding=utf-8

import json
import logging
import os
import sys
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_shapenet_data(input_file, sep_token="[SEP]"):
    """
    Process the ShapeNet dataset from JSON to the format expected by LlamaFactory.
    Formats data using conversation format with user/assistant roles that LlamaFactory expects.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    # Convert to the format expected by LlamaFactory
    processed_data = []
    for item in data:
        # Format the octree as a string with special tokens for each value
        formatted_octree = ""
        for i, level in enumerate(item.get("octree", [])):
            if i > 0:
                formatted_octree += f"{sep_token}"  # Separator between levels

            # Convert each element to a special token
            level_tokens = []
            for element in level:
                # Convert element to integer and ensure it's in the valid range
                try:
                    value = int(element)
                    if 0 <= value <= 255:
                        level_tokens.append(f"<octree_{value:03d}>")
                    else:
                        # Handle out-of-range values
                        level_tokens.append(f"<octree_000>")
                        logger.warning(f"Out of range octree value: {value}")
                except ValueError:
                    # Handle non-integer values
                    level_tokens.append(f"<octree_000>")
                    logger.warning(f"Non-integer octree value: {element}")

            formatted_octree += "".join(level_tokens)

        # Create conversation with user and assistant messages
        processed_data.append(
            {
                "conversations": [
                    {
                        "role": "user",
                        "content": f"Convert this 3D shape description to an octree representation: {item.get('caption', '')}",
                    },
                    {"role": "assistant", "content": formatted_octree},
                ]
            }
        )

    return processed_data


def create_dataset(shapenet_path, output_dir, sep_token="[SEP]"):
    """Create datasets from ShapeNet data and save to jsonl files for LlamaFactory."""
    # Process the data
    logger.info(f"Processing ShapeNet data from {shapenet_path}")
    processed_data = process_shapenet_data(shapenet_path, sep_token)

    # Create the dataset directory structure LlamaFactory expects
    dataset_dir = os.path.join(output_dir, "shapenet_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Calculate validation set size (10% or at most 100 samples)
    val_size = min(len(processed_data) // 10, 100)

    if val_size > 0:
        # Split the data into train and validation sets
        # Take validation samples from the end to ensure no overlap
        train_data = processed_data[:-val_size]
        val_data = processed_data[-val_size:]
    else:
        train_data = processed_data
        val_data = []

    # Write training data
    train_path = os.path.join(dataset_dir, "train.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    # Write validation data if we have any
    if val_data:
        val_path = os.path.join(dataset_dir, "validation.json")
        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

    logger.info(f"Dataset saved to {dataset_dir}")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    return dataset_dir


def create_yaml_config(model_name, dataset_dir, output_dir, model_nickname):
    """Create a YAML configuration file for LlamaFactory."""
    config_path = os.path.join(output_dir, "shapenet_config.yaml")

    with open(config_path, "w") as f:
        f.write(
            f"""### model
model_name_or_path: {model_name}
trust_remote_code: true

### method
stage: sft
do_train: true
# Use full fine-tuning instead of LoRA to properly train the new token embeddings
finetuning_type: full
# Alternatively, if you still want to use LoRA but need to train the embeddings:
# finetuning_type: lora
# lora_rank: 8
# lora_alpha: 32
# lora_dropout: 0.1
# lora_target: all
# train_embedding: true  # This is crucial for training the new token embeddings

### dataset
dataset: shapenet_train
eval_dataset: shapenet_val
template: chatml
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: saves/{model_nickname}-shapenet-full
logging_steps: 10
save_steps: 500
eval_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
evaluation_strategy: steps
report_to: wandb
"""
        )

    logger.info(f"YAML config created at {config_path}")
    return config_path


def create_dataset_info(info_path="data/dataset_info.json"):
    """Update data/dataset_info.json file to register the dataset with LlamaFactory."""

    # Read existing dataset_info.json
    with open(info_path, "r") as f:
        dataset_info = json.load(f)

    # Update/add shapenet dataset definitions
    dataset_info["shapenet_train"] = {
        "file_name": "shapenet_dataset/train.json",
        "formatting": "sharegpt",  # Use sharegpt format for conversations
        "columns": {"messages": "conversations"},  # Map conversations field to messages
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    dataset_info["shapenet_val"] = {
        "file_name": "shapenet_dataset/validation.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    # Save the updated file
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2, sort_keys=True)

    logger.info(f"Updated dataset info file at {info_path}")
    return info_path


def main():
    """Main function to prepare ShapeNet data for LlamaFactory training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare ShapeNet data for LlamaFactory"
    )
    parser.add_argument(
        "--shapenet_path",
        type=str,
        default="shapenet_octrees_list.json",
        help="Path to ShapeNet dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path for tokenizer",
    )
    parser.add_argument(
        "--sep_token", type=str, default="[SEP]", help="Token to separate octree levels"
    )
    parser.add_argument(
        "--info_path",
        type=str,
        default="data/dataset_info.json",
        help="Path to dataset_info.json file",
    )

    args = parser.parse_args()

    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)

    # Process and save the dataset in LlamaFactory format
    dataset_dir = create_dataset(args.shapenet_path, args.output_dir, args.sep_token)

    # Update the dataset_info.json file
    create_dataset_info(args.info_path)

    # Create local model directory
    local_model_dir = "./local_model"
    os.makedirs(local_model_dir, exist_ok=True)

    # Download the model and tokenizer
    logger.info(f"Downloading model and tokenizer from {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Add the SEP token
    special_token = args.sep_token

    # Create 256 octree tokens
    octree_tokens = [f"<octree_{i:03d}>" for i in range(256)]

    # Add all special tokens
    special_tokens_dict = {"additional_special_tokens": [special_token] + octree_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens to tokenizer")

    # Save the model and updated tokenizer to local directory
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    logger.info(f"Saved model and tokenizer with special tokens to {local_model_dir}")

    logger.info(f"Added SEP token: {special_token}")
    logger.info(f"Added 256 octree tokens: <octree_000> to <octree_255>")

    # Update the YAML config to use the local model directory
    model_nickname = args.model_name.split("/")[-1].replace(".", "p")
    config_path = create_yaml_config(
        local_model_dir, dataset_dir, args.output_dir, model_nickname
    )

    # Print instructions
    logger.info("\nAll preprocessing completed successfully!")
    logger.info("\nTo start training with LlamaFactory, run:")
    logger.info(f"llamafactory-cli train {config_path}")


if __name__ == "__main__":
    main()
