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
    parser.add_argument("--output_repo", type=str, default="gswamy/pythia-1.4B-tldr-spin-iter-")
    parser.add_argument("--prompts", type=str, default="cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162")
    parser.add_argument("--branch", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=53)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    return parser.parse_args()


def main():

    set_seed()
    # init
    args = parse_arguments()

    accelerator = Accelerator(gradient_accumulation_steps=8)

    dataset = load_dataset(args.prompts, split=args.branch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True,)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = dataset.with_format("torch", columns=["query_token"])

    for p in range(0, 1):
        all_query_responses = torch.load(f"./temp_datastore/all_query_responses_iter_{args.iter}_samp_{p}_spin.pt").tolist()
        all_masks = torch.load(f"./temp_datastore/all_masks_iter_{args.iter}_samp_{p}_spin.pt").tolist()
        all_decoded = torch.load(f"./temp_datastore/all_decoded_iter_{args.iter}_samp_{p}_spin.pt")
        # all_rewards = torch.load(f"./temp_datastore/all_rewards_iter_{args.iter}_samp_{p}_local.pt").tolist()
        # all_logprobs = torch.load(f"./temp_datastore/all_logprobs_{p}.pt").tolist()

        # print(p, len(all_query_responses), len(all_rewards), len(all_logprobs))

        dataset = dataset.add_column(f"iter_{args.iter}_query_response_{p}", all_query_responses)
        dataset = dataset.add_column(f"iter_{args.iter}_mask_{p}", all_masks)
        dataset = dataset.add_column(f"iter_{args.iter}_decoded_{p}", all_decoded)
        # dataset = dataset.add_column(f"iter_{args.iter}_reward_{p}", all_rewards)
        # dataset = dataset.add_column(f"iter_{args.iter}_logprob_{p}", all_logprobs)

    dataset.push_to_hub(args.output_repo + str(args.iter))


if __name__ == "__main__":
    main()