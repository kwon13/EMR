import os
from typing import List

import torch
from datasets import load_dataset
# from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from accelerate import Accelerator
from transformers import  Adafactor, AutoTokenizer, HfArgumentParser, pipeline, AdamW, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModelForSequenceClassification
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

SFT_MODEL_PATH = 'gpt2-small-sat'


config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=50,
        total_steps=100000,
        batch_size=8,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="GPT-small_PPO",
        save_best = False,
    ),
    model=ModelConfig(
        model_path=SFT_MODEL_PATH,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=SFT_MODEL_PATH,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=8,
        ppo_epochs=12,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
        },
    ),
)

PROMPT_DICT={
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

if __name__ == "__main__":
    # Load the pre-trained reward model
    current_device = Accelerator().local_process_index
    reward_model_name = "opt-iml-max_gsm8k_peft_stack-exchange-paired_rmts__100000_5e-05/checkpoint-4000"
    peft_model_id= "opt-iml-max_gsm8k_peft"
    rw_tokenizer = AutoTokenizer.from_pretrained("opt-iml-max_gsm8k")
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, num_labels=1)
    peft_reward_model = PeftModelForSequenceClassification.from_pretrained(reward_model, peft_model_id,
                                        device_map={"": current_device},
                                        model_kwargs={"load_in_8bit": True})

    sent_kwargs = {"top_k":None, "function_to_apply": "none", "batch_size": 8, "truncation": True}

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=peft_reward_model,
        device_map={"": current_device},
        model_kwargs={"load_in_8bit": True},
        tokenizer=rw_tokenizer,
    )

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    PROMPT_DICT["GSM8K"].replace("{QUESTION}", prompts[i]),
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts
        

    def reward_fn(samples: List[str], **kwargs):
        
        
        keyward = "Let’s think step by step."
        rewards=[]
        samples = [PROMPT_DICT["GSM8K"].replace("{QUESTION}", sample) for sample in samples]    
        samples = ["Q." + sample.split("\n\nQ.")[-1].replace(tokenizer.eos_token, "").strip().rstrip('.') for sample in samples] 
        pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
        norms_scores = torch.tensor([output[0]["score"] for output in pipe_outputs])
        return norms_scores

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    dataset = load_dataset('csv',data_files='gsm8k_ppo.csv', split='train')

    # Store data into prompt and label pairs
    train_set = [(sample["question"], sample["answer"], sample["answer_genrated_cot"]) for sample in dataset]

    # Split contents into summaries and labels
    train_posts, train_summaries, train_cots = zip(*train_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i].strip()] = eval(train_cots[i])
    
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        # eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
    