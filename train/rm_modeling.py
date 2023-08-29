from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import sentencepiece
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


if __name__ == "__main__":
    

    @dataclass
    class TrainingArguments(transformers.TrainingArguments):
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="adamw_torch")
        model_max_length: int = field(
            default=1024,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
        )
    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


    @dataclass
    class DataArguments:
        train_data_path: str = field(default="fiveflow/cot_ranking", metadata={"help": "Path to the training data."})
        valid_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
        reference_path: str = field(default=None, metadata={"help": "Path to the reference representation tensor."})
    
    num_proc = 24

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    train_dataset = load_dataset(data_args.train_data_path, split="train")
    eval_dataset = load_dataset(data_args.train_data_path, split="test")
    
    original_columns = train_dataset.column_names
    training_args.label_names=[]
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                               cache_dir=training_args.cache_dir, num_labels=1),AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=training_args.model_max_length)    
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["j_proj", "k_proj"]
    )
    
    model = get_peft_model(rank_model, peft_config)
    
    def preprocess_function(examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

        return new_examples


    # preprocess the dataset and filter out QAs that are longer than 512
    max_length = 1024
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids_j"]) <= max_length and len(x["input_ids_k"]) <= max_length)

    eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids_j"]) <= max_length and len(x["input_ids_k"]) <= max_length)

    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = 1024
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            features_j = []
            features_k = []
            for feature in features:
                features_j.append(
                    {
                        "input_ids": feature["input_ids_j"],
                        "attention_mask": feature["attention_mask_j"],
                    }
                )
                features_k.append(
                    {
                        "input_ids": feature["input_ids_k"],
                        "attention_mask": feature["attention_mask_k"],
                    }
                )
            batch_j = self.tokenizer.pad(
                features_j,
                padding='max_length',
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch_k = self.tokenizer.pad(
                features_k,
                padding='max_length',
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch = {
                "input_ids_j": batch_j["input_ids"],
                "attention_mask_j": batch_j["attention_mask"],
                "input_ids_k": batch_k["input_ids"],
                "attention_mask_k": batch_k["attention_mask"],
                "return_loss": True,
            }
            return batch


    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape, dtype=int)
        return accuracy.compute(predictions=predictions, references=labels)


    from transformers import Trainer
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
            rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=512),
    )

    trainer.train()