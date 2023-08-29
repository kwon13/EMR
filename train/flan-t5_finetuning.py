import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from datasets import load_dataset
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100

PROMPT_DICT={
    "AQUA":("Q. First train 108 m long moving at a speed of 50 km/hr crosses a second train 112 m long coming from opposite direction in 6 seconds. Find the speed of the second train? \n"
            "Let’s think step by step.\n"
            "speed of the second train consider as x\nrelative speed = (x+50)*5/18 m/sec\n(250+5x/18) m/sec\ndistance = (108 + 112) = 220 m\n==> 220/(250+5x/18) = 6\n250 + 5x = 660\n=>x = 82 km/hr \nTherefore, the answer is 82 km/hr \n\n"
            "Q. In a basketball game, Tim scored 20 points more than Joe, but only half as many points as Ken. If the three players scored a combined total of 100 points, how many points did Tim score? \n"
            "Let’s think step by step.\n"
            "Let Joe scored point = x\nThen Tim Scored = x+20\nKen Scored = 2*(x+20) = 2x+40\nAs given, x+x+20+2x+40 = 100 points\n4x+60 = 100\nx = 100-60/4 = 10\nSo Tim Scored = x +20 i.e) 10+20 = 30 \nTherefore, the answer is 30 \n\n"
            "Q. ABCDEFGHI is a regular polygon with nine sides. What is the measure in degrees of angle ACB? \n"
            "Let’s think step by step.\n"
            "the formula of sum of angles in a polygon of sides n=(n-2)*180\nso sum of angles=7*180..\ntherefore each angle ,angle ABC=7*180/9=140..\nwhen we join A and C to make a triangle ABC, it becomes an isosceles triangle with AB and BC equal..\nso angle ACB=angleCAB=(180-140)/2=20.. \nTherefore, the answer is 20 \n\n"
            "Q. {QUESTION} \n"
            "Let’s think step by step.\n"
            ),
    "GSM8K":("Q. It takes Jason 30 minutes to cut 1 lawn in his neighborhood.  If he cuts 8 yards on both Saturday and Sunday, how many hours does he spend cutting grass? \n"
            "Let’s think step by step. \n"
            "He cuts 8 yards on both Saturday and Sunday so that’s 2*8 = 16 yards \nIt takes him 30 minutes to cut 1 yard so it takes him 30*16 = 480 minutes \nThere are 60 minutes in 1 hour and it takes him 480 minutes so that’s a total of 480/60 = 8 hours \nTherefore, the answer is 8 \n\n"
            "Q. A church has 120 members. 40% are adults. The rest are children. How many children more children are there than adults? \n"
            "Let’s think step by step. \n"
            "There are 48 adults because 120 x .4 = 48 \n60% of members are children because 100 - 40 = 60 \nThere are 72 children because 120 x .6 = 72 \nThere are 24 more children than adults because 72 - 48 = 24 \nTherefore, the answer is 24 \n\n"
            "Q. A book costs $4 more than a CD. The CD is 30% cheaper than a $20 album. How much does the book cost? \n"
            "Let’s think step by step. \n"
            "The CD is 30/100 * 20 = $6 cheaper than the album.\nSo the cost of the CD is 20 - 6 = $14.\nThe book is $4 more expensive than the CD, which means it cost 14 + 4 = $18. \nTherefore, the answer is 18 \n\n"
            "Q. {QUESTION} \n"
            "Let’s think step by step.\n"
            ),
    "ZERO":(
            "Q. {QUESTION} \n"
            "Let’s think step by step.\n"
            ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        for text in sources
    ]
    
    labels = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        for text in targets
    ]
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dict, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = data_dict
        
        logging.warning("Formatting inputs...")
        sources = [PROMPT_DICT["GSM8K"].replace("{QUESTION}", question.strip()) for question in list_data_dict['question']]
        targets = list_data_dict['answer']

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_data_dict = load_dataset('csv', data_files='gsm8k_question_answer_ranking_correct_train.csv', split='train')
    valid_data_dict = load_dataset('csv', data_files='gsm8k_question_answer_ranking_correct_test.csv', split='train')
    
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_dict=train_data_dict)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_dict=valid_data_dict)
    # eval_dataset=None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
    
    
    
