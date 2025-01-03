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
    label_dataset: str = "cleanrl/summarize_from_feedback_oai_preprocessing_1705009345"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    num_train: int = 92832
    """number of training samples"""
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162"
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

    num_train_epochs: int = 0
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 4
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 2
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 2
    """per rank eval batch size"""

    # optional args filled while running
    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    batch_size: Optional[int] = 64
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""

    # other args
    base_model: str = "/data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b"
    """the name of the pretrained model to use"""
    output_dir: str = "models/trash"
    """Where to save the model"""
    lora: bool = True
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
            choice = data["choice"].reshape(-1, 1)
            chosen_token = data["query_response0_token"] * (1 - choice) + data["query_response1_token"] * choice
            rejected_token = data["query_response1_token"] * (1 - choice) + data["query_response0_token"] * choice
            # chosen_token = data["iter_1_best_query_response"]
            # rejected_token = data["iter_1_worst_query_response"]
            
            query_responses = torch.cat((chosen_token, rejected_token), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:chosen_token.shape[0]]
                rejected_rewards = predicted_reward[chosen_token.shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float()
                odds = F.sigmoid(chosen_rewards - rejected_rewards)
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                # items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                # items["chosen"].append(tokenizer.decode(chosen_token[i]))
                # items["rejected"].append(tokenizer.decode(rejected_token[i]))
                # items["batch"].append(data["batch"][i])
                # items["split"].append(data["split"][i])
                # items["confidence"].append(data["extra.confidence"][i].item())
                # items["choice"].append(data["choice"][i].item())
                # items["policies"].append(data["policies"][i])
                # items["chosen_policy"].append(data["chosen_policy"][i])
                # items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
                items["odds"].append(odds[i].item())
    model.train()
    return pd.DataFrame(items)


if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    rm_name = args.reward_model_path.split("/")[-1]

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
            "choice",
            "response0_token",
            "query_response0_token",
            "response1_token",
            "query_response1_token",
            "batch",
            "split",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True)

    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation"]:
        validation_dataset = load_dataset(args.task.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "choice",
                "response0_token",
                "query_response0_token",
                # "iter_1_best_query_response",
                # "iter_1_worst_query_response",
                "response1_token",
                "query_response1_token",
                # "batch",
                # "split",
                # "extra.confidence",
                # "chosen_policy",
                # "rejected_policy",
                # "policies",
            ],
        )
        eval_datasets.append(validation_dataset)
        eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)
    
    accelerator.print('num_batches:', len(dataloader))
    accelerator.print("The number of samples in validation_dataset", len(eval_datasets[0]))
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
        writer = SummaryWriter(f"/data/user_data/gswamy/eval_rm/{rm_name}")
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
        model: PreTrainedModel = ScalarModel.from_pretrained(args.reward_model_path, trust_remote_code=True)
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
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}


    update = 0
    accelerator.print('evaluting model')
    for eval_split in eval_dataloaders:
        evaluate_df = evaluate(args, accelerator, tokenizer, model, eval_dataloaders[eval_split])
        # for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
        #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], update)
        #     accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
        # for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
        #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], update)
        #     accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
        # for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
        #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], update)
        #     accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
        
        for lb in np.arange(0, 1.0, 0.1):
            ub = lb + 0.1
            mask = (evaluate_df["odds"] >= lb) & (evaluate_df["odds"] < ub)
            writer.add_scalar(f"eval/rm/{eval_split}/accuracy/odds/{lb}-{ub}", evaluate_df[mask]["accuracy"].mean(), update)
            accelerator.print(f"eval/rm/{eval_split}/accuracy/odds/{lb}-{ub}: {evaluate_df[mask]['accuracy'].mean()}")
        writer.add_scalar(f"eval/rm/{eval_split}/likelihood", evaluate_df["odds"].mean(), update)
        accelerator.print(f"eval/rm/{eval_split}/likelihood: {evaluate_df['odds'].mean()}")
        writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), update)
        accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")

        if accelerator.is_main_process:
            os.makedirs(f"/data/user_data/gswamy/eval_rm/{rm_name}/", exist_ok=True)
            evaluate_df.to_csv(f"/data/user_data/gswamy/eval_rm/{rm_name}/eval_{eval_split}_{update}.csv")

            if eval_split != "validation_cnndm":
                np.save(f"/data/user_data/gswamy/eval_rm/{rm_name}/validation_odds.npy", evaluate_df["odds"].to_numpy())
                np.save(f"/data/user_data/gswamy/eval_rm/{rm_name}/validation_acc.npy", evaluate_df["accuracy"].to_numpy())
                likelihood = evaluate_df["odds"].mean()
                with open(f"/data/user_data/gswamy/eval_rm/{rm_name}/likelihood.txt", "w") as f:
                    f.write(str(likelihood))
                print(f"rm: {rm_name}, likelihood: {likelihood}")

                
            accelerator.print(f"saved evaluation table to eval_tables/{run_name}/eval_{eval_split}_{update}.csv")
            if args.track:
                wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
        del evaluate_df
        torch.cuda.empty_cache()
