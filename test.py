import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
from torch.utils.data import DataLoader
import transformers
import inspect
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, RemoveColumnsCollator
from prompter import Prompter
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import sys
import math
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
block_size_10MB = 10<<20
_signature_columns = None
remove_unused_columns = True
#%%
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    freeze_layers: str = field(
        default=None,
        metadata={"help": "If set freeze selected layers"},
    )
    remove_layers: str = field(
        default=None,
        metadata={"help": "If set remove selected layers"},
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    prompt_template_name: str = field(
        default="alpaca",
        metadata={"help": "prompt_template_name"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    val_set_size: int = field(
        default=2000,
        metadata={"help": "val_set_size"},
    )
    preprocessing_num_workers: int = field(
        default=100,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "learning_rate"},
    )
    output_dir: str = field(
        default="",
        metadata={"help": "output_dir"},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    initial_global_step: int = field(
        default=0,
        metadata={"help": "initial_global_step"}
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    layer_list = []
    prompter = Prompter()

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    layer_list = model.base_model.layers
    
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    freeze_layers = model_args.freeze_layers
    if freeze_layers is not "":
        layer_indexes = [int(x) for x in freeze_layers.split(",")]
        for layer_idx in layer_indexes:
            for param in list(layer_list[layer_idx].parameters()):
                param.requires_grad = False
            print ("Froze Layer: ", layer_idx)
    
    remove_layers = model_args.remove_layers
    if remove_layers is not "":
            layer_indexes = [int(x) for x in remove_layers.split(",")]
            layer_indexes.sort(reverse=True)
            for layer_idx in layer_indexes:
                if layer_idx < 0:
                    print ("Only positive indices allowed")
                    sys.exit(1)
                del(layer_list[layer_idx])
                print ("Removed Layer: ", layer_idx)
    
    model.config.num_hidden_layers = len(layer_list)   
    print(model.config)                 
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    #Extend vocab for speech units
    if '<sosp>' not in tokenizer.get_vocab():
        units_size=1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>','<eosp>','[Human]','[SynthLM]','<eoh>','<eos>']
        tokenizer.add_tokens(new_tokens)
    for token in ['<sosp>','<eosp>','[Human]','[SynthLM]','<eoh>','<eos>']:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])

    #resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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
        '''
        moss-style instructions
        '''
        full_prompt = prompter.generate_prompt(
            data_point["prefix"],
            data_point["plain_text"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["prefix"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt
    
    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    tokenized_cache_file_names = {
        "train":os.path.join(training_args.cache_dir, 'tokenized', 'train', 'processed_train.arrow'),
        "test":os.path.join(training_args.cache_dir, 'tokenized', 'valid', 'processed_valid.arrow'),
        }

    if training_args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_val_data = (
            train_val.map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=True,
                cache_file_names=tokenized_cache_file_names,
                desc=f"generate_and_tokenize_prompt",
                )
        )
        train_data = train_val_data["train"]
        val_data = train_val_data["test"]
    else:
        train_data = data["train"].map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=True,
                cache_file_names=tokenized_cache_file_names,
                desc=f"generate_and_tokenize_prompt",
                )
        val_data = None

    train_data = train_data.shuffle(1234)
    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_data if training_args.do_train else None, 
        eval_dataset=val_data if training_args.do_eval else None, 
        data_collator=data_collator
        )
    for batch in trainer.get_train_dataloader():
        model(**batch)
train()
# %%
