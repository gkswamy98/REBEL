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
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model


# a patch
@dataclass
class TaskHParams:
    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    num_train: int = 92832
    """number of training samples"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """number of training samples"""


@dataclass
class Args:
    # common args
    exp_name: str = "pythia_RM"
    """the name of this experiment"""
    seed: int = 66613
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize_pythia"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
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

    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 4
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 4
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 2
    """per rank eval batch size"""

    # optional args filled while running
    world_size: Optional[int] = 2
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    batch_size: Optional[int] = 128
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""

    # other args
    base_model: str = "/data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b"
    """the name of the pretrained model to use"""
    output_dir: str = "/data/user_data/gswamy/models/models/rm_sft_tldr_pythia_1.4b"
    """Where to save the model"""
    lora: bool = False
    """Whether to use lora"""
    lora_rank: int = 1024
    """the rank of the lora matrix"""
    lora_alpha: int = 2048
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    reward_model_path: str = ""
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    task: TaskHParams = field(default_factory=TaskHParams)


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


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

    def __init__(self, args: Args, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        if args.lora:
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            self.lm_backbone = get_peft_model(self.lm_backbone, peft_config=self.peft_config)
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = (torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(query_responses.device)
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths]


def evaluate(args: Args, accelerator, tokenizer, model, dataloader):
    model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float()
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i]))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["confidence"].append(data["extra.confidence"][i].item())
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
    model.train()
    return pd.DataFrame(items)


if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # load dataset
    dataset = load_dataset(args.task.label_dataset, split="train")
    dataset = dataset.select(range(args.task.num_train))
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "chosen_token",
            "query_chosen_token",
            "rejected_token",
            "query_rejected_token",
            "batch",
            "split",
            # "iter_1_worst_query_response",
            # "iter_1_worst_mask",
            # "iter_1_best_query_response",
            # "iter_1_best_mask",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True)

    # eval_datasets = []
    # eval_dataloaders = {}
    # for split in ["validation", "validation_cnndm"]:
    #     validation_dataset = load_dataset(args.task.label_dataset, split=split).flatten()
    #     validation_dataset = validation_dataset.with_format(
    #         "torch",
    #         columns=[
    #             "query_token",
    #             "choice",
    #             "chosen_token",
    #             "query_chosen_token",
    #             "rejected_token",
    #             "query_rejected_token",
    #             "batch",
    #             "split",
    #             "extra.confidence",
    #             "chosen_policy",
    #             "rejected_policy",
    #             "policies",
    #         ],
    #     )
    #     eval_datasets.append(validation_dataset)
    #     eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)
    
    accelerator.print('num_batches:', len(dataloader))
    # accelerator.print("The number of samples in validation_dataset", len(eval_datasets[0])+len(eval_datasets[1]))
    accelerator.print("The number of samples in dataset", len(dataset))
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{args.output_dir.split('/')[1]}"
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
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.base_model)
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if len(args.reward_model_path) == 0:
        model: PreTrainedModel = ScalarModel(args, scalar_model_config)
    else:
        model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    disable_dropout(model)

    if accelerator.is_main_process:
        pprint(model_config)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    if args.deepspeed:
        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}

    accelerator.print("===training model===")
    losses = torch.zeros(args.gradient_accumulation_steps, device=device)
    accuracies = torch.zeros(args.gradient_accumulation_steps, device=device)
    reward_preferreds = torch.zeros(args.gradient_accumulation_steps, device=device)
    reward_rejecteds = torch.zeros(args.gradient_accumulation_steps, device=device)

    model.train()
    gradient_accumulation_idx = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            # query_responses = torch.cat((data["iter_1_best_query_response"], data["iter_1_worst_query_response"]), dim=0)

            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)

                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                # chosen_rewards = predicted_reward[:data['iter_1_best_query_response'].shape[0]]
                # rejected_rewards = predicted_reward[data['iter_1_best_query_response'].shape[0]:]

                accuracy = (chosen_rewards > rejected_rewards).float().mean()

                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            losses[gradient_accumulation_idx] = loss
            accuracies[gradient_accumulation_idx] = accuracy
            reward_preferreds[gradient_accumulation_idx] = chosen_rewards.mean()
            reward_rejecteds[gradient_accumulation_idx] = rejected_rewards.mean()

            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if gradient_accumulation_idx == 0:
                update += 1
                scheduler.step()
                train_accuracy = accelerator.gather(accuracies).mean().item()

                writer.add_scalar("train/rm/loss", accelerator.gather(losses).mean().item(), update)
                writer.add_scalar("train/rm/accuracy", train_accuracy, update)
                writer.add_scalar("train/rm/chosen_rewards", accelerator.gather(reward_preferreds).mean().item(), update)
                writer.add_scalar("train/rm/rejected_rewards", accelerator.gather(reward_rejecteds).mean().item(), update)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], update)
                accelerator.print(f"{train_accuracy=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}")

    # if args.run_eval:
    #     accelerator.print('evaluting model')
    #     for eval_split in eval_dataloaders:
    #         evaluate_df = evaluate(args, accelerator, tokenizer, model, eval_dataloaders[eval_split])
    #         for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], update)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
    #         for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], update)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
    #         for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], update)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
    #         writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), update)
    #         accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
    #         if accelerator.is_main_process:
    #             os.makedirs(f"eval_tables/{run_name}", exist_ok=True)
    #             evaluate_df.to_csv(f"eval_tables/{run_name}/eval_{eval_split}_{update}.csv")
    #             if args.track:
    #                 wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
    #         del evaluate_df
    #         torch.cuda.empty_cache()

    norm_dataset = load_dataset(args.task.query_dataset, split="train")
    norm_dataset = norm_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    norm_dataloader = DataLoader(norm_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True)
    items = defaultdict(list)
    norm_dataloader = accelerator.prepare(norm_dataloader)
    with torch.no_grad():
        for data in tqdm(norm_dataloader):
            reference_responses = data["reference_response_token"].to(device, non_blocking=True)
            queries = data["query_token"].to(device, non_blocking=True)
            query_responses = torch.cat((queries, reference_responses), dim=1)
            predicted_reward = get_reward(model, query_responses, tokenizer)
            predicted_reward = accelerator.gather(predicted_reward)
            queries = accelerator.gather(queries)
            reference_responses = accelerator.gather(reference_responses)
            for i in range(len(predicted_reward)):
                items["query"].append(tokenizer.decode(queries[i], skip_special_tokens=True))
                items["reference_response"].append(tokenizer.decode(reference_responses[i]))
                items["predicted_reward"].append(predicted_reward[i].item())

    if accelerator.is_main_process:
        norm_df = pd.DataFrame(items)
        os.makedirs(f"eval_tables/{run_name}", exist_ok=True)
        norm_df.to_csv(f"eval_tables/{run_name}/eval_{update}_normalized.csv")
        if args.track:
            wandb.log({"samples/normalized": wandb.Table(dataframe=norm_df)}, step=update)
        stats = {
            "mean": norm_df["predicted_reward"].mean(),
            "std": norm_df["predicted_reward"].std(),
            "max": norm_df["predicted_reward"].max(),
            "min": norm_df["predicted_reward"].min(),
        }
        for stat_name, stat_value in stats.items():
            writer.add_scalar(f"eval/rm/normalized_{stat_name}", stat_value, update)
            accelerator.print(f"Normalized Reward {stat_name.capitalize()}: {stat_value}")

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        accelerator.print('saving model')
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.config.bias = norm_df["predicted_reward"].mean()
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
