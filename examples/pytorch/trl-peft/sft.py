# coding=utf-8
# Copyright 2023 mail.anindya@gmail.com. All rights reserved.
#
# Modified from: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
#
# This is a generic class to fine tune any chat model. One can start with a 
# base model trained on code on trained on instruction.
# The dataset must have a "text" column that would contain the formatted prompt.
import os
import sys
import math
import logging
from typing import List, Optional

from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments,
    pipeline
)

from transformers.trainer_utils import (
    set_seed, 
    get_last_checkpoint,
    is_torch_tpu_available
)

from peft import PeftModel, PeftConfig, LoraConfig

from trl import SFTTrainer, is_xpu_available

import evaluate

from tqdm import tqdm
tqdm.pandas()

logger = logging.getLogger(__name__)

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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def train(
        model_args: ModelArguments, 
        quantz_args: QuantizationArguments, 
        peft_args: PeftArguments, 
        data_args: DataArguments, 
        training_args: TrainingArguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    logger.info(f"Detecting last checkpoint")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Step 1: Load the dataset
    logger.info(f"Step 1: Load the dataset")
    
    if data_args.dataset_name:
        dataset = load_dataset(data_args.dataset_name)
        dataset['validation'] = dataset.pop('test')
    else:
        raise ValueError(
            f"Dataset name not provided. Cannot train a model without a dataset."
        )

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Step 2: Load the model and tokenizer
    logger.info(f"Load the model and tokenizer")
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

    # Step 3: Define the training arguments
    # override any arguments passed from the sys.argv
    training_args.fp16 = not training_args.bf16

    # Step 4: Define the LoraConfig
    logger.info(f"Define the LoraConfig")
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
    logger.info(f"Define the Trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=data_args.seq_length,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        dataset_text_field=data_args.dataset_text_field,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        peft_config=peft_config,
    )

    # Step 6: Training
    if training_args.do_train:
        logger.info(f"Start Training")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Finished Training")

    # Step 7: Evaluation
    if training_args.do_eval:
        logger.info(f"Start Evaluation")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info(f"Finished Evaluation")

    # Step 8: Save the model
    output_dir = os.path.join(training_args.output_dir, "final")
    logger.info(f"Save model to {output_dir}")
    trainer.save_model(output_dir)

def test(training_args: TrainingArguments):
    
    peft_model_id = os.path.join(training_args.output_dir, "final")
    hub_token = training_args.hub_token

    peft_config = PeftConfig.from_pretrained(peft_model_id, token=hub_token)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, token=hub_token)
    merged_model = PeftModel.from_pretrained(base_model, peft_model_id, token=hub_token)

    tokenizer = AutoTokenizer.from_pretrained(
        peft_model_id, 
        token=hub_token
    )

    prompt = "Who is Leonardo Da Vinci?"
    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, QuantizationArguments, PeftArguments, DataArguments, TrainingArguments))
    model_args, quantz_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    train(model_args, quantz_args, peft_args, data_args, training_args)

    test(training_args)
