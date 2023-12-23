# coding=utf-8
# Copyright 2023 mail.anindya@gmail.com. All rights reserved.
#
# Modified from: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
#
# This is a generic class to fine tune any chat model. One can start with a 
# base model trained on code on trained on instruction.
# The dataset must have a "text" column that would contain the formatted prompt.
import os
from typing import List, Optional
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments
)

from trl import SFTTrainer, is_xpu_available


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer/peft_config we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="codellama/CodeLlama-7b-Instruct-hf", 
        metadata={"help":"the model name"},
    )
    trust_remote_code: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enable `trust_remote_code`"}
    )

@dataclass
class QuantizationArguments:
    """
    Arguments pertaining to what quantization we are going to use.
    """

    load_in_8bit: Optional[bool] = field(
        default=False, 
        metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, 
        metadata={"help": "load the model in 4 bits precision"}
    )

@dataclass
class PeftArguments:
    """
    Arguments pertaining to what peft configuration we are going to use.
    """

    use_peft: Optional[bool] = field(
        default=False, 
        metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, 
        metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, 
        metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "Target modules for LoRA adapters"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", 
        metadata={"help": "The name of the dataset to use (via the hf datasets library)."}
    )
    validation_fraction: Optional[float] = field(
        default=0.2, 
        metadata={"help": "Fraction of data to use for training validation"}
    )
    seq_length: Optional[int] = field(
        default=512, 
        metadata={"help": "Input sequence length"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", 
        metadata={"help": "the text field of the dataset"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, QuantizationArguments, PeftArguments, DataArguments, TrainingArguments))
    model_args, quantz_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses(look_for_args_file=True)
    
    # Step 1: Load the model
    if quantz_args.load_in_8bit and quantz_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif quantz_args.load_in_8bit or quantz_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantz_args.load_in_8bit, 
            load_in_4bit=quantz_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=training_args.hub_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=training_args.hub_token,)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side="right"

    # Step 2: Load the dataset
    dataset = load_dataset(data_args.dataset_name, split="train")

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim = training_args.optim,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        fp16=not training_args.bf16,
        bf16=training_args.bf16,
        max_grad_norm = training_args.max_grad_norm,
        warmup_ratio = training_args.warmup_ratio,
        lr_scheduler_type = training_args.lr_scheduler_type,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        report_to=training_args.report_to,
        save_steps=training_args.save_steps,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        push_to_hub=training_args.push_to_hub,
        hub_model_id=training_args.hub_model_id,
        hub_token=training_args.hub_token,
        gradient_checkpointing=training_args.gradient_checkpointing,
        #gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        save_safetensors=True,
        resume_from_checkpoint=True,
        ddp_find_unused_parameters=False,
    )

    # Step 4: Define the LoraConfig
    if peft_args.use_peft:
        peft_config = LoraConfig(
            r=peft_args.peft_lora_r,
            lora_alpha=peft_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=peft_args.target_modules,
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=data_args.seq_length,
        train_dataset=dataset,
        dataset_text_field=data_args.dataset_text_field,
        peft_config=peft_config,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(os.path.join(training_args.output_dir, "final"))

if __name__ == '__main__':
    main()
