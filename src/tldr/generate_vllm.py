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
from vllm import LLM, SamplingParams


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
    parser.add_argument("--reward_model", type=str, default="models/rm_sft_tldr_pythia_1_4b")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--output_repo", type=str, default="gswamy/pythia-1.4B-tldr")
    parser.add_argument("--prompts", type=str, default="cleanrl/summarize_from_feedback_oai_preprocessing_1705009345")
    parser.add_argument("--branch", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=53)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()

def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)

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

def get_local_reward(model, query_responses, masks, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    masks = masks[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (masks != tokenizer.pad_token_id)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=masks.unsqueeze(2)).squeeze(2)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    return all_logps

def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

def main():
    args = parse_arguments()

    set_seed(args.p * 50)
    print("Model", args.model)
    print("Iter", args.iter)
    print("p", args.p)

    dataset = load_dataset(args.prompts, split=args.branch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True,)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    sft_dataset = load_dataset("cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162", split="train")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
    )

    prompts = [row['query'] for row in tqdm(dataset)]
    
    # sft_prompts = [row['query'] for row in tqdm(sft_dataset)]

    # prompts = prompts + sft_prompts  # Kiante's suggestion

    sampling_params = [SamplingParams(temperature=0.1, top_p=1.0, max_tokens=53, seed=((args.p + 1) * 50) + i) for i in range(len(prompts))]

    outputs = llm.generate(prompts, sampling_params)

    l1 = []
    l2 = []

    for output in outputs:
        prompt = output.prompt
        l1.append(prompt)
        generated_text = output.outputs[0].text
        l2.append(generated_text)
    
    torch.save(l1, f"./temp_datastore/all_query_iter_{args.iter}_samp_{args.p}_vllm_temp.pt")
    torch.save(l2, f"./temp_datastore/all_response_iter_{args.iter}_samp_{args.p}_vllm_temp.pt")

if __name__ == "__main__":
    main()