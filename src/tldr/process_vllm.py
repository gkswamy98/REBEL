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

warnings.filterwarnings("ignore")

def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sft_tldr_pythia_1.4b")
    parser.add_argument("--reward_model", type=str, default="models/gpt_rm_sft_tldr_pythia_1_4b") # TODO: change this to the correct model
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--output_repo", type=str, default="gswamy/pythia-1.4B-tldr-ws-iter-")
    parser.add_argument("--prompts", type=str, default="cleanrl/summarize_from_feedback_oai_preprocessing_1705009345")
    parser.add_argument("--branch", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=53)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
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

    dataset = load_dataset(args.prompts, split=args.branch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True, add_eos_token=True)
    left_tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True, add_eos_token=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    left_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = dataset.with_format("torch", columns=["query_token"])

    reward_model: PreTrainedModel = ScalarModel.from_pretrained(args.reward_model, trust_remote_code=True,)
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = 8
    eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
    }
    accelerator.print(f"{eval_ds_config=}")
    reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
    reward_model.eval()

    for p in range(args.p, args.p + 1):
        print(f"Processing shard {p}")
        all_query = torch.load(f"./temp_datastore/all_query_iter_{args.iter}_samp_{p}_vllm_temp.pt")
        all_response = torch.load(f"./temp_datastore/all_response_iter_{args.iter}_samp_{p}_vllm_temp.pt")
       
        all_query_token = left_tokenizer(all_query, padding=True, max_length=512)
        all_query_token = torch.tensor(all_query_token["input_ids"]).to(accelerator.device)
        all_response_token = tokenizer(all_response, padding=True, max_length=args.maxlen, truncation=True)
        all_response_token = torch.tensor(all_response_token["input_ids"]).to(accelerator.device)

        contain_pad_token = torch.any(all_response_token == tokenizer.pad_token_id, dim=-1)
        
        all_query_response = torch.cat((all_query_token, all_response_token), dim=1)
        all_masks = all_query_response.clone()
        all_masks[:, :512] = tokenizer.pad_token_id
        all_rewards = []

        for batch in tqdm(DataLoader(
            torch.utils.data.TensorDataset(all_query_response),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )):
            query_responses = batch[0]
            with torch.no_grad():
                _, scores, _ = get_reward(reward_model, query_responses, tokenizer, 512)
            all_rewards.append(scores)
        all_rewards = torch.cat(all_rewards)
        all_rewards = torch.where(contain_pad_token, all_rewards, torch.full_like(all_rewards, -1e6))

        torch.save(all_query_response.detach().cpu(), f"./temp_datastore/all_query_responses_iter_{args.iter}_samp_{p}_vllm_temp.pt")
        torch.save(all_masks.detach().cpu(), f"./temp_datastore/all_masks_iter_{args.iter}_samp_{p}_vllm_temp.pt")
        torch.save(all_rewards.detach().cpu(), f"./temp_datastore/all_rewards_iter_{args.iter}_samp_{p}_vllm_temp.pt")

        # _, scores, _ = get_reward(reward_model, all_query_response, tokenizer, 512)



        # all_query_responses = []
        # decoded = tokenizer.batch_decode(postprocessed_query_responses)

        # all_query_responses = torch.load(f"./temp_datastore/all_query_responses_iter_{args.iter}_samp_{p}_ws.pt").tolist()
        # all_masks = torch.load(f"./temp_datastore/all_masks_iter_{args.iter}_samp_{p}_ws.pt").tolist()
        # all_decoded = torch.load(f"./temp_datastore/all_decoded_iter_{args.iter}_samp_{p}_ws.pt")
        # all_rewards = torch.load(f"./temp_datastore/all_rewards_iter_{args.iter}_samp_{p}_ws.pt").tolist()
        # all_logprobs = torch.load(f"./temp_datastore/all_logprobs_{p}.pt").tolist()

        # print(p, len(all_query_responses), len(all_rewards), len(all_logprobs))

        # dataset = dataset.add_column(f"iter_{args.iter}_query_response_{p}", all_query_responses)
        # dataset = dataset.add_column(f"iter_{args.iter}_mask_{p}", all_masks)
        # dataset = dataset.add_column(f"iter_{args.iter}_decoded_{p}", all_decoded)
        # dataset = dataset.add_column(f"iter_{args.iter}_reward_{p}", all_rewards)
        # dataset = dataset.add_column(f"iter_{args.iter}_logprob_{p}", all_logprobs)

    # dataset.push_to_hub(args.output_repo + str(args.iter))


if __name__ == "__main__":
    main()