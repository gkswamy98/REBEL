import pandas as pd
from datasets import load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from dataclasses import field
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
import deepspeed
from torch import optim


from rich.console import Console
from rich.pretty import pprint
from rich.table import Table

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
import warnings
import numpy as np

from vllm import LLM, SamplingParams
import os

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
)
import gc

warnings.filterwarnings("ignore")

def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b")
    parser.add_argument("--reward_model", type=str, default="/data/user_data/gswamy/models/models/rm_sft_tldr_pythia_1.4b_1")
    parser.add_argument("--local_reward_model", type=str, default="/data/user_data/gswamy/models/models/local_rm_H_lowlr_tldr_pythia_1.4b_1")
    parser.add_argument("--eval_df_path", type=str, default="eval_df.csv")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--output_repo", type=str, default="gswamy/pythia-1.4B-tldr-ws-iter-")
    parser.add_argument("--prompts", type=str, default="cleanrl/summarize_from_feedback_oai_preprocessing_1705009345")
    parser.add_argument("--branch", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=53)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--use_ref", type=bool, default=False)
    
    return parser.parse_args()

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def truncate_response(tokenizer, responses):
    trunc_idxs = first_true_indices(responses == tokenizer.eos_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [53]
    idxs = torch.arange(53, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward

def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )

def main():

    set_seed()
    # init
    args = parse_arguments()

    accelerator = Accelerator(gradient_accumulation_steps=8)

    # dataset = load_dataset(args.prompts, split=args.branch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True, add_eos_token=True)
    left_tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True, add_eos_token=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    left_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    local_rm = LLM(
        model=args.local_reward_model,
        tensor_parallel_size=1,
        distributed_executor_backend='mp',
        disable_custom_all_reduce=True,
    )

    eval_df = pd.read_csv(args.eval_df_path, converters={'postprocessed_responses': pd.eval})
    all_query = eval_df["query"].tolist()
    all_query_token = left_tokenizer(all_query, padding=True, max_length=512)
    all_query_token = torch.tensor(all_query_token["input_ids"]).to(accelerator.device)
    
    all_responses = eval_df["postprocessed_responses"].tolist()
    print(all_responses[:10])
    bon_rewards = []
    N = 25

    for n in tqdm(range(N)):
        all_response = [x[n] for x in all_responses]
        all_response_token = tokenizer(all_response, padding=True, max_length=args.maxlen, truncation=True)
        all_response_token = torch.tensor(all_response_token["input_ids"]).to(accelerator.device)

        contain_pad_token = torch.any(all_response_token == tokenizer.pad_token_id, dim=-1)
        
        all_query_response = [query + response for query, response in zip(all_query, all_response)]
        all_rewards = []

        sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
        outputs = local_rm.generate(prompts=all_query_response, sampling_params=sampling_params)
        for output in outputs:
            logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
            reward = torch.sum(torch.tensor(logprobs))
            # reward = torch.tensor(logprobs[-1]) # last token
            all_rewards.append(reward)
        all_rewards = torch.tensor(all_rewards).to(accelerator.device)

        all_rewards = torch.where(contain_pad_token, all_rewards, torch.full_like(all_rewards, -1e6))
        bon_rewards.append(all_rewards)

    bon_rewards = torch.stack(bon_rewards, dim=1)

    print("destroying model parallel")
    destroy_model_parallel()
    del local_rm.llm_engine.model_executor.driver_worker
    del local_rm.llm_engine.model_executor
    del local_rm
    gc.collect()
    torch.cuda.empty_cache()

    if args.use_ref:
        print("trying to create model parallel")
        ref_model = LLM(
                model=args.model,
                tensor_parallel_size=1,
                distributed_executor_backend='mp',
                disable_custom_all_reduce=True,
        )
        
        bon_rewards_ref = []
        for n in tqdm(range(N)):
            all_response = [x[n] for x in all_responses]
            all_response_token = tokenizer(all_response, padding=True, max_length=args.maxlen, truncation=True)
            all_response_token = torch.tensor(all_response_token["input_ids"]).to(accelerator.device)

            contain_pad_token = torch.any(all_response_token == tokenizer.pad_token_id, dim=-1)
            
            all_query_response = [query + response for query, response in zip(all_query, all_response)]
            all_rewards = []

            sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
            outputs = ref_model.generate(prompts=all_query_response, sampling_params=sampling_params)
            for output in outputs:
                logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
                reward = torch.sum(torch.tensor(logprobs))
                # reward = torch.tensor(logprobs[-1]) # last token
                all_rewards.append(reward)
            all_rewards = torch.tensor(all_rewards).to(accelerator.device)

            all_rewards = torch.where(contain_pad_token, all_rewards, torch.full_like(all_rewards, -1e6))
            bon_rewards_ref.append(all_rewards)

        bon_rewards_ref = torch.stack(bon_rewards_ref, dim=1)
        bon_rewards = bon_rewards - bon_rewards_ref # subtracting the reference rewards


    gm = args.eval_df_path.split("/")[-2]
    rm = args.local_reward_model.split("/")[-1]
    os.makedirs(f"/data/user_data/gswamy/eval_bon/{gm}", exist_ok=True)
    os.makedirs(f"/data/user_data/gswamy/eval_bon/{gm}/{rm}", exist_ok=True)
    
    for n in tqdm([1, 2, 5, 10, 25]):
        best_idx = torch.argmax(bon_rewards[:, :n], dim=1)
        bon = []
        for i in range(len(eval_df)):
            bon.append(all_responses[i][best_idx[i]])
        eval_df_n = eval_df.copy()
        del eval_df_n["postprocessed_responses"]
        eval_df_n["postprocessed_response"] = bon
        eval_df_n.to_csv(f"/data/user_data/gswamy/eval_bon/{gm}/{rm}/table_{n}.csv")


    # dataset.push_to_hub(args.output_repo + str(args.iter))


if __name__ == "__main__":
    main()