import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from datasets import load_dataset
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
        targets = list_data_dict['gold']
        wins = list_data_dict['win']
        loses = list_data_dict['lose']

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        win_data_dict = preprocess(sources, wins, tokenizer)
        lose_data_dict = preprocess(sources, loses, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.win_input_ids = win_data_dict["input_ids"]     
        self.win_labels = win_data_dict["labels"]     
        self.lose_input_ids = lose_data_dict["input_ids"]     
        self.lose_labels = lose_data_dict["labels"] 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i], 
            labels=self.labels[i],     
            win_input_ids=self.win_input_ids[i], 
            win_labels = self.win_labels[i],      
            lose_input_ids=self.lose_input_ids[i], 
            lose_labels = self.lose_labels[i],
            )     
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, win_input_ids, win_labels, lose_input_ids, lose_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", 
                                                                                                                        "labels", 
                                                                                                                        "win_input_ids", 
                                                                                                                        "win_labels", 
                                                                                                                        "lose_input_ids", 
                                                                                                                        "lose_labels", 
                                                                                                                            # "ref_key"
                                                                                                                            ))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        win_input_ids = torch.nn.utils.rnn.pad_sequence(
            win_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        win_labels = torch.nn.utils.rnn.pad_sequence(win_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        lose_input_ids = torch.nn.utils.rnn.pad_sequence(
            lose_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        lose_labels = torch.nn.utils.rnn.pad_sequence(lose_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            
            win_input_ids=win_input_ids,
            win_labels = win_labels,
            win_attention_mask=win_input_ids.ne(self.tokenizer.pad_token_id),
            
            lose_input_ids=lose_input_ids,
            lose_labels = lose_labels,
            lose_attention_mask=lose_input_ids.ne(self.tokenizer.pad_token_id),
            
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_data_dict = load_dataset('csv', data_files='only_5_answer_train.csv', split='train').shuffle(42)
    valid_data_dict = load_dataset('csv', data_files='ranking_cot_with_gold_v.csv', split='train')
    
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_dict=train_data_dict)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_dict=valid_data_dict)
    # eval_dataset=None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class SATTrainer(Trainer):
    """Supervised Alignment Training Trainer"""
   
    def _shift_right(self, input_ids):
        decoder_start_token_id = 0 # self.config.decoder_start_token_id
        pad_token_id = 0 # self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
                "See T5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    
    def normalize(self, embds):
        embds = embds.div(torch.norm(embds, dim=-1).unsqueeze(-1))
        return embds
    
    def bert_score(self, ref_rep, rep_2):
        norm_rep_1 = self.normalize(ref_rep)
        norm_rep_2 = self.normalize(rep_2)
        pairwise_cosine = torch.matmul(norm_rep_1, norm_rep_2.transpose(1,2))
        P_max, _ = pairwise_cosine.max(dim=1)
        R_max, _ = pairwise_cosine.max(dim=2)
        P = P_max.mean(dim=1)
        R = R_max.mean(dim=1)
        F_score = 2 * (R * P) / (R + P)
        return F_score
        
    def compute_loss(self, model, inputs, return_outputs=False):
        alpha = 1
        
        ref_decoder_input_ids = self._shift_right(inputs["labels"])
        ref = model(input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"])
        ref_rep = ref['decoder_hidden_states'][-1]
        ref_loss = ref.loss
        
        win_decoder_input_ids = self._shift_right(inputs["win_labels"])
        win = model(input_ids=inputs["win_input_ids"], 
                    attention_mask=inputs["win_attention_mask"],
                    decoder_input_ids=win_decoder_input_ids)
        win_last_hs = win['decoder_hidden_states'][-1]
        
        lose_decoder_input_ids = self._shift_right(inputs["lose_labels"])
        lose = model(input_ids=inputs["lose_input_ids"], 
                     attention_mask=inputs["lose_attention_mask"],
                     decoder_input_ids=lose_decoder_input_ids)
        lose_last_hs = lose['decoder_hidden_states'][-1]
        
        win_bert_score = self.bert_score(ref_rep, win_last_hs)
        lose_bert_score = self.bert_score(ref_rep, lose_last_hs)
        
        ranking_loss = -nn.functional.logsigmoid(win_bert_score - lose_bert_score).mean()
        
        loss = ref_loss + ranking_loss * alpha
        
        if return_outputs:
            return loss, {"logits": ref.logits}
        return loss



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        output_hidden_states=True,
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

    trainer = SATTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
    
    
    
"""
## SAT small
deepspeed --num_gpus 2 flan-t5_SAT.py \
    --model_max_length 1536 \
    --model_name_or_path "flan-t5-small-gsm8k" \
    --output_dir 'flan-t5-samll_SAT' \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 6 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_steps=500 \
    --save_strategy="steps" \
    --save_steps=500 \
    --remove_unused_columns False \
    --evaluation_strategy='steps' \
    --deepspeed ds_flan_t5_z3_config_bf16.json
    
## SAT base
deepspeed --num_gpus 2 flan-t5_SAT.py \
    --model_max_length 1536 \
    --model_name_or_path "flan-t5-base-gsm8k" \
    --output_dir 'flan-t5-base_SAT' \
    --num_train_epochs 4 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 6 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_steps=500 \
    --save_strategy="steps" \
    --save_steps=500 \
    --remove_unused_columns False \
    --evaluation_strategy='steps' \
    --deepspeed ds_flan_t5_z3_config_bf16.json
    
## SAT large
deepspeed --num_gpus 2 flan-t5_SAT.py \
    --model_max_length 1536 \
    --model_name_or_path "flan-t5-large-gsm8k" \
    --output_dir 'flan-t5-large_sat_baselines' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_steps=500 \
    --save_strategy="steps" \
    --save_steps=500 \
    --remove_unused_columns False \
    --evaluation_strategy='steps' \
    --deepspeed ds_flan_t5_z3_config_bf16.json
"""