import copy
import logging
import torch
import transformers
import os
import re
import shutil
import sys
import numpy as np
import time
import os
import math
import datasets
import torch.nn as nn
from functools import partial
from prompter import Prompter
from pathlib import Path
import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import Trainer
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import BaseModelOutput

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from typing import Any, Dict, List, Optional, Union

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    get_scheduler,
)
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#%%

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    student_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained LLM model or model identifier from huggingface.co/models"
        }
    )
    teacher_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained teacher model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    loss_type: str = field(
        default='KD_loss',
        metadata={
            "help": (
                "The type of supervised or unsupervised KD loss to be used during training "
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    prompt_template_name: str = field(
        default="alpaca",
        metadata={"help": "prompt_template_name"},
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in each dataset when loading multiple datasets with streaming mode. "
            "Not required when using one dataset or non-streaming mode. The sample values provide the sampling "
            "probability for each dataset. Setting them equal to the number of sample values ensures that every "
            "sample from every dataset is used once per epoch."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the "
            "training dataset config name if unspecified."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing if using non-streaming mode."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    val_set_size: int = field(
        default=2000,
        metadata={"help": "val_set_size"},
    )

    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set will pad the target sequence to a multiple of the provided"
                " value. This is important to avoid triggering recompilations on TPU."
                " If unspecified, will default to padding the targets to max length."
            )
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={"help": ("Whether to only do data preprocessing and skip training.")},
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={
            "help": "Whether to use Datasets' streaming mode to load and pre-process the data."
        },
    )
    timestamp_probability: float = field(
        default=0.2,
        metadata={
            "help": "Probability for training on timestamped tokens if the data contains it."
        },
    )
    condition_on_prev_probability: float = field(
        default=0.2,
        metadata={"help": "Probability for conditioning on the previous text example."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to predict timestamps in the generation step."
        },
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_decoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze some layers of decoder model. Only recommended when the entire decoder has been "
                "copied from the teacher model."
            )
        },
    )
    temperature: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "Temperature to anneal the logits when computing the softmax."
        },
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    mode: Optional[str] = field(
        default="min",
        metadata={
            "help": (
                "mode for learning rate scheduler"
            )
        },
    )

def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def get_layers_to_supervise(student_layers: int, teacher_layers: int) -> Dict:
    layer_intervals = np.linspace(
        teacher_layers // student_layers - 1,
        teacher_layers - 1,
        student_layers,
        dtype=int,
    )
    layer_intervals[-1] = teacher_layers - 1
    layer_map = {}

    for student_layer, teacher_layer in enumerate(layer_intervals):
        layer_map[student_layer] = teacher_layer

    return layer_map


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [
        str(x)
        for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
        if os.path.isdir(x)
    ]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(
    save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint"
) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(
        output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
    )
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
        )
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])),
    )


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, DistillationTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(project_name=data_args.wandb_project)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(
                repo_name, exist_ok=True, token=training_args.hub_token
            ).repo_id
            # Clone repo locally
            repo = Repository(
                training_args.output_dir,
                clone_from=repo_id,
                token=training_args.hub_token,
            )

            with open(
                os.path.join(training_args.output_dir, ".gitignore"), "w+"
            ) as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # set seed for determinism
    set_seed(training_args.seed)
    layer_list = []
    prompter = Prompter()

    student_model = LlamaForCausalLM.from_pretrained(
        model_args.student_model_name_or_path, output_hidden_states=True, output_attentions=True
    )

    teacher_model = LlamaForCausalLM.from_pretrained(
        model_args.teacher_model_name_or_path, output_hidden_states=True, output_attentions=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.student_model_name_or_path,
        model_max_length=model_args.model_max_length,
        use_fast=False,
    )
    
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if "<sosp>" not in tokenizer.get_vocab():
        units_size = 1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + [
            "<sosp>",
            "<eosp>",
            "[Human]",
            "[SynthLM]",
            "<eoh>",
            "<eos>",
        ]
        tokenizer.add_tokens(new_tokens)
    for token in ["<sosp>", "<eosp>", "[Human]", "[SynthLM]", "<eoh>", "<eos>"]:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])

    embedding_size = student_model.get_input_embeddings().weight.shape[0]
    logger.info(f"Resizing student model embedding")
    if len(tokenizer) > embedding_size:
        student_model.resize_token_embeddings(len(tokenizer))

    embedding_size = teacher_model.get_input_embeddings().weight.shape[0]
    logger.info(f"Resizing teacher model embedding")
    if len(tokenizer) > embedding_size:
        teacher_model.resize_token_embeddings(len(tokenizer))

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        """
        moss-style instructions
        """
        full_prompt = prompter.generate_prompt(
            data_point["prefix"],
            data_point["plain_text"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(data_point["prefix"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_args.train_dataset_name.endswith(
        ".json"
    ) or data_args.train_dataset_name.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.train_dataset_name)
    else:
        data = load_dataset(data_args.train_dataset_name)

    tokenized_cache_file_names = {
        "train": os.path.join(
            data_args.dataset_cache_dir, "tokenized", "train", "processed_train.arrow"
        ),
        "test": os.path.join(
            data_args.dataset_cache_dir, "tokenized", "valid", "processed_valid.arrow"
        ),
    }

    if data_args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=data_args.val_set_size, shuffle=True, seed=42
        )
        train_val_data = train_val.map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            cache_file_names=tokenized_cache_file_names,
            desc=f"generate_and_tokenize_prompt",
        )
        train_data = train_val_data["train"]
        val_data = train_val_data["test"]

    else:
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            cache_file_names=tokenized_cache_file_names,
            desc=f"generate_and_tokenize_prompt",
        )
        val_data = None

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError(
            "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=data_args.pad_target_to_multiple_of,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=student_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=val_data if training_args.do_eval else None,
        data_collator=data_collator,
    )

    share_hidden_states = (
        training_args.freeze_decoder
        and student_model.config.d_model == teacher_model.config.d_model
    )

    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    if training_args.freeze_decoder:
        student_model.freeze_decoder()
        student_model.model.gradient_checkpointing = False

    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if not data_args.streaming and training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(train_data) // (
            train_batch_size * gradient_accumulation_steps
        )
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info(
            "max_steps is given, it will override any value given in num_train_epochs"
        )
        total_train_steps = int(training_args.max_steps)
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps
    else:
        raise ValueError(
            "max_steps must be specified when training with a streaming (iterable) dataset"
        )

    if training_args.eval_steps is None:
        logger.info(
            f"eval_steps is not set, evaluating at the end of {'each epoch' if not data_args.streaming else 'training'}"
        )
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module=[student_model.decoder] if training_args.freeze_decoder else None,
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for name, param in student_model.named_parameters()
                if name in decay_parameters
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                param
                for name, param in student_model.named_parameters()
                if name not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(student_model.generation_config, "num_beams", 1)
    )

    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence
    
    def unsupervised_C_kl_divergence(teacher_distribution, student_distribution, labels):
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        b, s, d = teacher_distribution.size(0), teacher_distribution.size(1), teacher_distribution.size(-1)
        Lco = F.cosine_similarity(student_distribution, teacher_distribution)
        loss_Lco = -torch.nn.LogSoftmax(-1)(Lco).diag()
        divergence = loss_Lco.sum() / (b*s)
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    def unsupervised_kl_divergence(teacher_distribution, student_distribution, labels, beta):
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        b, s, d = teacher_distribution.size(0), teacher_distribution.size(1), teacher_distribution.size(-1)
        Lco = F.cosine_similarity(student_distribution, teacher_distribution)
        Lss = F.cosine_similarity(student_distribution.permute(2, 1, 0), teacher_distribution.permute(2, 1, 0))
        loss_Lco = -torch.nn.LogSoftmax(-1)(Lco).diag()
        loss_Lss = -torch.nn.LogSoftmax(0)(Lss).diag()
        divergence = beta * loss_Lco.sum() / (b*s)  + (1-beta) * loss_Lss.sum() / d
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence
    
    def train_step(
        batch,
        temperature=1.0,
    ):
        student_model.train()
        teacher_model.eval()
        
        student_outputs = student_model(**batch)
        with torch.no_grad():
            # do the full forward pass for the teacher model
            teacher_outputs = teacher_model(**batch)
        ce_loss = student_outputs.loss
        if model_args.loss_type=='KD_loss':
            teacher_distribution = nn.functional.softmax(
                teacher_outputs.logits / temperature, dim=-1
            )
            student_distribution = nn.functional.log_softmax(
                student_outputs.logits / temperature, dim=-1
            )
            kl_loss = (
                kl_divergence(teacher_distribution, student_distribution, batch["labels"])
                * temperature**2
            )
            loss = (1 - training_args.kl_weight) * ce_loss + training_args.kl_weight * kl_loss
        
        elif model_args.loss_type=='F_B_KD_loss':
            teacher_distribution_TS = nn.functional.softmax(
                teacher_outputs.logits / temperature, dim=-1
            )
            student_distribution_TS = nn.functional.log_softmax(
                student_outputs.logits / temperature, dim=-1
            )
            
            teacher_distribution_ST = nn.functional.log_softmax(
                teacher_outputs.logits / temperature, dim=-1
            )
            student_distribution_ST = nn.functional.softmax(
                student_outputs.logits / temperature, dim=-1
            )
            kl_loss_TS = (
                kl_divergence(teacher_distribution_TS, student_distribution_TS, batch["labels"])
                * temperature**2
            )
            kl_loss_ST = (
                kl_divergence(student_distribution_ST, teacher_distribution_ST, batch["labels"])
                * temperature**2
            )
            kl_loss = 0.5 * kl_loss_ST + 0.5 * kl_loss_TS
            loss = 0.3 * ce_loss + training_args.kl_weight * kl_loss
            
        elif model_args.loss_type=='C_KD_loss':
            teacher_distribution = teacher_outputs.hidden_states[-1]
            student_distribution = student_outputs.hidden_states[-1]
            kl_loss = (
                unsupervised_C_kl_divergence(teacher_distribution, student_distribution, batch["labels"])
            )
            loss = 0.5 * ce_loss + training_args.kl_weight * kl_loss  
              
        elif model_args.loss_type=='S_KD_loss':
            
            teacher_distribution_TS = nn.functional.softmax(
                teacher_outputs.logits / temperature, dim=-1
            )
            student_distribution_TS = nn.functional.log_softmax(
                student_outputs.logits / temperature, dim=-1
            )
            
            teacher_distribution_ST = nn.functional.log_softmax(
                teacher_outputs.logits / temperature, dim=-1
            )
            student_distribution_ST = nn.functional.softmax(
                student_outputs.logits / temperature, dim=-1
            )
            
            teacher_hidden_state = teacher_outputs.hidden_states[-1]
            student_hidden_state = student_outputs.hidden_states[-1]
            
            kl_loss_TS = (
                kl_divergence(teacher_distribution_TS, student_distribution_TS, batch["labels"])
                * temperature**2
            )
            kl_loss_ST = (
                kl_divergence(student_distribution_ST, teacher_distribution_ST, batch["labels"])
                * temperature**2
            )
            kl_loss_S = (
                unsupervised_kl_divergence(teacher_hidden_state, student_hidden_state, batch["labels"], 0.5)
            )
            logit_teacher = teacher_outputs.logits
            logit_student = student_outputs.logits
            kl_loss = 0.5 * kl_loss_ST + 0.5 * kl_loss_TS
            loss = 0.3 * ce_loss + training_args.kl_weight * kl_loss + 0.3 * kl_loss_S
        
        elif model_args.loss_type=='U_KD_loss':
            teacher_distribution = teacher_outputs.hidden_states[-1]
            student_distribution = student_outputs.hidden_states[-1]
            kl_loss = (
                unsupervised_kl_divergence(teacher_distribution, student_distribution, batch["labels"], 0.5)
            )
            loss = 0.5 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    def eval_step(batch):
        student_model.eval()
        teacher_model.eval()

        with torch.no_grad():
            student_outputs = student_model(**batch)
            teacher_outputs = teacher_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss

        # log softmax / softmax for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
        # temperature is always 1 for eval
        kl_loss = kl_divergence(
            teacher_distribution, student_distribution, batch["labels"]
        )

        
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return metrics

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm.tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")
    
        steps_trained_progress_bar.update(cur_step)
        
        for epoch in range(0, epochs_trained):
            train_data = train_data.shuffle(training_args.seed)

        if not data_args.streaming and training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            resume_step = None
            train_data = train_data.shuffle(training_args.seed)
    else:
        resume_step = None
    
    for epoch in range(epochs_trained, num_epochs):
        train_data = train_data.shuffle(training_args.seed)
        train_dataloader = trainer.get_train_dataloader()
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None
        
        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step(loss)
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1
                # For ReduceLROnPlateau optimizer.param_groups[0]["lr"]
                # else lr_scheduler.get_last_lr()[0]
                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss'].item(), train_metric['kl_loss'].item(), train_metric['ce_loss'].item()}, Learning Rate:"
                        f" {optimizer.param_groups[0]['lr']})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )  
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)

                        if cur_step == total_train_steps:
                            student_model = accelerator.unwrap_model(student_model)
                            student_model.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )       
                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_preds = []
                    eval_labels = []
                    eval_start = time.time()
                    validation_dataloader = trainer.get_eval_dataloader()
                    validation_dataloader = accelerator.prepare(validation_dataloader)
                    for batch in tqdm.tqdm(
                            validation_dataloader,
                            desc=f"Evaluating ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                        eval_metric = eval_step(batch)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)
                    eval_time = time.time() - eval_start
                    eval_metrics = {
                            key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                        }
                    steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            )
                        
                    log_metric(
                            accelerator,
                            metrics=eval_metrics,
                            train_time=eval_time,
                            step=cur_step,
                            epoch=epoch,
                            prefix='eval',
                        )

                    # flush the train metrics
                    train_start = time.time()
                if cur_step == total_train_steps:
                    continue_training = False
                    break
        if not continue_training:
            break
    accelerator.end_training()

if __name__ == "__main__":
    main()

# %%
