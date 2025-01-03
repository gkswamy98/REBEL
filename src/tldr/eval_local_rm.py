import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import gc
import contextlib



@dataclass
class LabelHParams:
    type: Optional[str] = None
    num_train: int = 92832
    num_labels: int = 2
    source: Optional[str] = None


# a patch
@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162"

    query_format_str: Optional[str] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    query_truncate_field: Optional[str] = "post"
    query_truncate_text: Optional[str] = "\n"
    query_padding: Optional[str] = None  # defaults to repeated spaces
    query_pad_side: Optional[str] = "left"

    # Response params
    response_length: int = 53

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.01

@dataclass
class Args:
    # common args
    exp_name: str = "pythia_1.4_refless_dpo_lora_rm"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize_pythia"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    world_size: Optional[int] = 8
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 4
    """The number of gradient accumulation steps"""
    local_micro_batch_size: int = 2
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = 16
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = 8
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = 64
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""

    # other args
    #base_model: str = "EleutherAI/pythia-160m"
    base_model: str = "/data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b"
    """the name of the pretrained model to use"""
    reward_model: str = "/data/user_data/gswamy/models/models/rm_tldr_pythia_1.4b"
    """the name of the reward model to use"""
    local_reward_model: str = ""
    """the name of the local reward model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/refless_dpo_policy_model_1_4b"
    """Where to save the model"""
    label_dataset: str = "cleanrl/summarize_from_feedback_oai_preprocessing_1705009345"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    ipo: bool = False
    """Whether to use IPO loss https://arxiv.org/abs/2310.12036"""
    label_smoothing: float = 0.0
    """Label smoothing for DPO (Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf))"""
    beta: float = 0.05
    """The beta value for DPO"""
    use_ref: bool = False
    """Whether to use the reference prob in the reward model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    label: LabelHParams = field(default_factory=LabelHParams)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def forward(model, query_responses, labels, mb_best, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    labels = labels[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (labels != tokenizer.pad_token_id)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    masked = per_token_logps * loss_mask
    all_logps = masked.sum(-1)
    # last_logps = torch.zeros_like(all_logps)
    # for i in range(masked.size(0)):
    #     for j in range(masked.size(1)):
    #         if loss_mask[i, j]:
    #             last_logps[i] = masked[i, j] # last token version
    chosen_logps = all_logps.view(-1, args.label.num_labels).gather(1, mb_best.view(-1, 1)).view(-1)
    rejected_logps = all_logps.view(-1, args.label.num_labels).gather(1, (1 - mb_best).view(-1, 1)).view(-1)
    return chosen_logps, rejected_logps


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
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


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def evaluate_rm(args: Args, accelerator, tokenizer, model, dataset):
    # model.eval()
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)

    query_response_0 = dataset["query_response0"]
    outputs_0 = model.generate(prompts=query_response_0, sampling_params=sampling_params)
    all_rewards_0 = []
    for output in outputs_0:
        logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
        reward_all = torch.sum(torch.tensor(logprobs))
        # reward_last = torch.tensor(logprobs[-1]) # last token
        all_rewards_0.append(reward_all)
    all_rewards_0 = torch.tensor(all_rewards_0).to(accelerator.device)

    query_response_1 = dataset["query_response1"]
    outputs_1 = model.generate(prompts=query_response_1, sampling_params=sampling_params)
    all_rewards_1 = []
    for output in outputs_1:
        logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
        reward_all = torch.sum(torch.tensor(logprobs))
        # reward_last = torch.tensor(logprobs[-1]) # last token
        all_rewards_1.append(reward_all)
    all_rewards_1 = torch.tensor(all_rewards_1).to(accelerator.device)

    print("destroying model parallel")
    destroy_model_parallel()
    del model.llm_engine.model_executor.driver_worker
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if args.use_ref:
        print("trying to create model parallel")
        ref_model = LLM(
                model=args.base_model,
                tensor_parallel_size=1,
                distributed_executor_backend='mp',
                disable_custom_all_reduce=True,
        )    
        query_response_0_ref = dataset["query_response0"]
        outputs_0_ref = ref_model.generate(prompts=query_response_0_ref, sampling_params=sampling_params)
        all_rewards_0_ref = []
        for output in outputs_0_ref:
            logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
            reward_all = torch.sum(torch.tensor(logprobs))
            # reward_last = torch.tensor(logprobs[-1]) # last token
            all_rewards_0_ref.append(reward_all)
        all_rewards_0_ref = torch.tensor(all_rewards_0_ref).to(accelerator.device)

        query_response_1_ref = dataset["query_response1"]
        outputs_1_ref = ref_model.generate(prompts=query_response_1_ref, sampling_params=sampling_params)
        all_rewards_1_ref = []
        for output in outputs_1_ref:
            logprobs = [list(x.values())[0].logprob for x in output.prompt_logprobs[1:]]
            reward_all = torch.sum(torch.tensor(logprobs))
            # reward_last = torch.tensor(logprobs[-1]) # last token
            all_rewards_1_ref.append(reward_all)
        all_rewards_1_ref = torch.tensor(all_rewards_1_ref).to(accelerator.device)

    choice = dataset["choice"].to(accelerator.device)
    chosen_rewards = all_rewards_0 * (1 - choice) + all_rewards_1 * choice
    rejected_rewards = all_rewards_0 * choice + all_rewards_1 * (1 - choice)
    
    if args.use_ref:
        chosen_rewards_ref = all_rewards_0_ref * (1 - choice) + all_rewards_1_ref * choice
        rejected_rewards_ref = all_rewards_0_ref * choice + all_rewards_1_ref * (1 - choice)
        chosen_rewards = chosen_rewards - chosen_rewards_ref
        rejected_rewards = rejected_rewards - rejected_rewards_ref

    # chosen_rewards = all_rewards_0
    # rejected_rewards = all_rewards_1
    accuracy = (chosen_rewards > rejected_rewards)
    odds = F.sigmoid((chosen_rewards - rejected_rewards) * args.beta)
    return pd.DataFrame(
        {
            # "query": dataset["query"],
            # "response0": dataset["response0"],
            # "response1": dataset["response1"],
            # "batch": dataset["batch"],
            # "split": dataset["split"],
            # "confidence": dataset["extra.confidence"],
            # "choice": dataset["choice"],
            "accuracy": accuracy.cpu().numpy(),
            "odds": odds.cpu().numpy(),
        }
    )



if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id

    local_rm_name = args.local_reward_model.split("/")[-1]
    print(f"local_rm: {local_rm_name}, beta: {args.beta}, use_ref: {args.use_ref}")

    # load dataset
    dataset = load_dataset(args.label_dataset, split="train")
    dataset = dataset.shuffle(seed=local_seed)
    dataset = dataset.select(range(args.label.num_train))
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "choice",
            "response0_token",
            "query_response0_token",
            "query_response0_token_response_label",
            "response1_token",
            "query_response1_token",
            "query_response1_token_response_label",
            "batch",
            "split",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation"]:
        validation_dataset = load_dataset(args.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                # "query_token",
                "choice",
                # "response0_token",
                # "query_response0_token",
                # "query_response0_token_response_label",
                # "response1_token",
                # "query_response1_token",
                # "query_response1_token_response_label",
                # "batch",
                # "split",
                # "extra.confidence",
                "query_response0",
                "query_response1",
                # "response0_policy",
                # "response1_policy",
                # "policies",
            ],
        )
    # for split in ["validation", "validation_cnndm"]:
    #     validation_dataset = load_dataset("gswamy/pythia-1.4B-tldr-gpt-val2", split=split).flatten()
    #     validation_dataset = validation_dataset.with_format(
    #         "torch",
    #         columns=[
    #             # "query_token",
    #             # "choice",
    #             # "response0_token",
    #             # "query_response0_token",
    #             "iter_1_best_query_response",
    #             "iter_1_worst_query_response",
    #             # "response1_token",
    #             # "query_response1_token",
    #             # "batch",
    #             # "split",
    #             # "extra.confidence",
    #             # "chosen_policy",
    #             # "rejected_policy",
    #             # "policies",
    #         ],
    #     )
        eval_datasets.append(validation_dataset)
        eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.local_eval_batch_size)
        accelerator.print("The number of samples in validation_dataset", len(validation_dataset))
    accelerator.print("The number of samples in dataset", len(dataset))

    sft_validation_dataset = load_dataset(args.task.query_dataset, split="validation")
    sft_validation_dataset = sft_validation_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    sft_validation_dataloader = DataLoader(sft_validation_dataset, batch_size=args.local_eval_batch_size)

    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            # file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            # wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"/data/user_data/gswamy/eval_rm/{local_rm_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    sft_validation_dataloader = accelerator.prepare(sft_validation_dataloader)

    local_rm = LLM(
        model=args.local_reward_model,
        tensor_parallel_size=1,
        distributed_executor_backend='mp',
        disable_custom_all_reduce=True,
    )

    global_step = 0
    update = 0
    if args.run_eval:
        for idx, eval_split in enumerate(["validation"]):
            print("EVAL RM")
            evaluate_df = evaluate_rm(args, accelerator, tokenizer, local_rm, eval_datasets[idx])
            print("FINISH EVAL")
            # for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
            #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], global_step)
            #     accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
            # for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
            #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
            #     accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
            # for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
            #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
            #     accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
            for lb in np.arange(0, 1.0, 0.1):
                ub = lb + 0.1
                mask = (evaluate_df["odds"] >= lb) & (evaluate_df["odds"] < ub)
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/odds/{lb}-{ub}", evaluate_df[mask]["accuracy"].mean(), update)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/odds/{lb}-{ub}: {evaluate_df[mask]['accuracy'].mean()}")
            writer.add_scalar(f"eval/rm/{eval_split}/likelihood", evaluate_df["odds"].mean(), global_step)
            accelerator.print(f"eval/rm/{eval_split}/likelihood: {evaluate_df['odds'].mean()}")
            writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), global_step)
            accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
            
            if accelerator.is_main_process:
                os.makedirs(f"/data/user_data/gswamy/eval_rm/{local_rm_name}/", exist_ok=True)
                evaluate_df.to_csv(f"/data/user_data/gswamy/eval_rm/{local_rm_name}/eval_{eval_split}_{update}.csv")
                if eval_split != "validation_cnndm":
                    np.save(f"/data/user_data/gswamy/eval_rm/{local_rm_name}/validation_odds.npy", evaluate_df["odds"].to_numpy())
                    np.save(f"/data/user_data/gswamy/eval_rm/{local_rm_name}/validation_acc.npy", evaluate_df["accuracy"].to_numpy())
                    likelihood = evaluate_df["odds"].mean()
                    with open(f"/data/user_data/gswamy/eval_rm/{local_rm_name}/likelihood.txt", "w") as f:
                        f.write(str(likelihood))
                    print(f"local_rm: {local_rm_name}, likelihood: {likelihood}")
                if args.track:
                    wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
            del evaluate_df
            torch.cuda.empty_cache()