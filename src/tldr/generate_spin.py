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
    parser.add_argument("--reward_model", type=str, default="models/rm_sft_tldr_pythia_1_4b")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--output_repo", type=str, default="gswamy/pythia-1.4B-tldr")
    parser.add_argument("--prompts", type=str, default="cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162")
    parser.add_argument("--branch", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=53)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
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

# TODO (gswamy): fix generation code ... somehow, what we're sampling is not leading to good policy optimization.
def main():

    set_seed()
    # init
    args = parse_arguments()

    print("Iter", args.iter)

    accelerator = Accelerator(gradient_accumulation_steps=8)

    dataset = load_dataset(args.prompts, split=args.branch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True,)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_config = AutoConfig.from_pretrained(args.model)
    dropout_layer_keys = ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    configure_dropout(model_config, dropout_layer_keys, 0.0)
    if accelerator.is_main_process:
        pprint(model_config)
    
    policy = AutoModelForCausalLM.from_pretrained(args.model, config=model_config, trust_remote_code=True)
    accelerator.print(policy)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

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

    dataset = dataset.with_format("torch", columns=["query_token"])
    dataloader = DataLoader(dataset, batch_size=28, shuffle=False, drop_last=False)

    optimizer = optim.Adam(policy.parameters())
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    generation_config = GenerationConfig(
        max_new_tokens=53,
        min_new_tokens=53,
        temperature=(0.7 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    for p in range(args.p, args.p + 1):
        set_seed(p * 50)
        all_query_responses = []
        all_masks = []
        all_rewards = []
        # all_logprobs = []
        all_decoded = []
        i = 0
        print(len(dataset))
        for data in tqdm(dataloader):
            i += 1
            queries = data["query_token"]
            context_length = queries.shape[1]
            
            # 1. generate responses
            query_responses = generate(
                    accelerator.unwrap_model(policy),
                    queries,
                    tokenizer,
                    generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(tokenizer, responses) # remove everything after the first EOS token
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            masks = postprocessed_query_responses.clone()
            masks[:, :context_length] = tokenizer.pad_token_id
            decoded = tokenizer.batch_decode(postprocessed_query_responses)

            all_query_responses.append(postprocessed_query_responses.detach().cpu())
            all_masks.append(masks.detach().cpu())
            all_decoded.extend(decoded)

            del query_responses, responses, decoded, queries
            torch.cuda.empty_cache()

            # 2. get log probs
            # output = forward(policy, query_responses, tokenizer)
            # logits = output.logits[:, context_length - 1 : -1]
            # logits /= generation_config.temperature
            # all_logprob = F.log_softmax(logits, dim=-1)
            # logprobs = torch.gather(all_logprob, 2, responses.unsqueeze(-1)).squeeze(-1)
            # del output, logits, all_logprob, query_responses
            # torch.cuda.empty_cache()
            # all_logprobs.append(logprobs.detach().cpu())
            
            # 3. get rewards
            # 3.1 query reward model
            # postprocessed_responses = truncate_response(tokenizer, responses)
            # postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            # _, scores, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer, context_length)
            # scores = get_local_reward(reward_model, postprocessed_query_responses, masks, tokenizer)
            # print(scores)
            # all_rewards.append(rewards.detach().cpu())
            # del rewards, postprocessed_query_responses
            # torch.cuda.empty_cache()

            # del postprocessed_query_responses
            # torch.cuda.empty_cache()
            # # 3.2 penalize for length
            # contain_pad_token = torch.any(postprocessed_responses == tokenizer.pad_token_id, dim=-1)
            # rewards = torch.where(contain_pad_token, scores, torch.full_like(scores, -1e6)) # penalize for length
            # all_rewards.append(rewards.detach().cpu())
            # if contain_pad_token.sum() / len(contain_pad_token) < 1.0:
            #     accelerator.print(f"{(contain_pad_token.sum() / len(contain_pad_token))=}")
            del postprocessed_responses, postprocessed_query_responses, masks
            torch.cuda.empty_cache()
            
            # # 3.3 calculate KL penalty
            # sequence_lengths = first_true_indices(postprocessed_responses == tokenizer.pad_token_id) - 1
            # seq_mask = torch.arange(responses.size(1), device=policy.device).unsqueeze(0).expand_as(responses) <= sequence_lengths.unsqueeze(1)
            # logprobs = (logprobs * seq_mask).sum(-1)
            # ref_logprobs = (ref_logprobs * seq_mask).sum(-1)
            # kl = logprobs - ref_logprobs
            # # 3.4 add in KL penalty
            # non_score_reward = -0.05 * kl
            # rewards = non_score_reward + scores
            # del scores, kl, non_score_reward, postprocessed_responses, sequence_lengths, seq_mask, logprobs, ref_logprobs
            # torch.cuda.empty_cache()
            
            # decoded = tokenizer.batch_decode(postprocessed_query_responses)
            # all_rewards.append(rewards.detach().cpu())
            # all_decoded.append(decoded)
            # del rewards, postprocessed_query_responses
            # torch.cuda.empty_cache()
            
            # if i > 5:
            #     break

        all_query_responses = torch.cat(all_query_responses, 0)
        all_masks = torch.cat(all_masks, 0)
        # all_rewards = torch.cat(all_rewards, 0)
        # all_logprobs = torch.cat(all_logprobs, 0)

        torch.save(all_query_responses, f"./temp_datastore/all_query_responses_iter_{args.iter}_samp_{p}_spin.pt")
        torch.save(all_masks, f"./temp_datastore/all_masks_iter_{args.iter}_samp_{p}_spin.pt")
        torch.save(all_decoded, f"./temp_datastore/all_decoded_iter_{args.iter}_samp_{p}_spin.pt")
        # torch.save(all_rewards, f"./temp_datastore/all_rewards_iter_{args.iter}_samp_{p}_local.pt")
        # torch.save(all_logprobs, f"./temp_datastore/all_logprobs_{p}.pt")

        # print(p, len(all_query_responses), len(all_rewards), len(all_logprobs))

        # all_query_responses.extend([all_query_responses[-1] for _ in range(len(dataset) - len(all_query_responses))])
        # all_rewards.extend([all_rewards[-1] for _ in range(len(dataset) - len(all_rewards))])
        # all_logprobs.extend([all_logprobs[-1] for _ in range(len(dataset) - len(all_logprobs))])
        # all_decoded.extend([all_decoded[-1] for _ in range(len(dataset) - len(all_decoded))])

        # dataset = dataset.add_column(f"iter_{args.iter}_query_response_{p}", all_query_responses)
        # dataset = dataset.add_column(f"iter_{args.iter}_reward_{p}", all_rewards)
        # dataset = dataset.add_column(f"iter_{args.iter}_logprob_{p}", all_logprobs)
        # dataset = dataset.add_column(f"iter_{args.iter}_decoded_{p}", all_decoded)

    # dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()