import argparse
import copy
import logging

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq, LlamaModel


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialise a student model from a teacher model, copying the relevant layer weights and adjusting as necessary."
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="The HF Hub ID of the teacher checkpoint.",
    )
    parser.add_argument(
        "--num_of_layers",
        type=str,
        default=None,
        help="Number of encoder layers to use in the student model. Defaults to all layers from the teacher.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Where to save the student weights and processor.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        required=False,
        default=False,
        help="Whether to push the student weights and processor to the Hub.",
    )

    args = parser.parse_args()
    return args


def init_student_model_from_teacher(
    teacher_checkpoint,
    num_of_layers=None,
    save_dir=None,
    push_to_hub=None
):
    
    teacher_model = LlamaForCausalLM.from_pretrained(
        teacher_checkpoint
    )
    
    layer_list = teacher_model.base_model.layers
    teacher_config = teacher_model.config
    teacher_encoder_layers = teacher_config.num_hidden_layers

    student_config = copy.deepcopy(teacher_config)
    remove_layers = num_of_layers
    if remove_layers is not "":
        layer_indexes = [int(x) for x in remove_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            if layer_idx < 0:
                print ("Only positive indices allowed")
                sys.exit(1)
            del(layer_list[layer_idx])
            print ("Removed Layer: ", layer_idx)
                
    student_config.update(
        {
            "num_hidden_layers": len(layer_list)
        }
    )

    encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, student_config.num_hidden_layers, dtype=int)
    encoder_mapping[-1] = teacher_encoder_layers - 1

    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[teacher_layer] = student_layer

    student_model = LlamaModel(student_config)
    
    # remove the teacher params and model
    del teacher_model

    # save the converted weights and model
    if save_dir is not None:
        student_model.save_pretrained(save_dir)
        # we also need to correctly save the processor and generation config

    # check we can do a forward pass with the saved model - first load the weights
    logger.info("Checking we can load the saved model...")
    student_model = LlamaForCausalLM.from_pretrained(
        save_dir
    )

    logger.info("Conversion successful!")

    if push_to_hub:
        student_model.push_to_hub(save_dir)
        processor.push_to_hub(save_dir)
        generation_config.push_to_hub(save_dir)


if __name__ == "__main__":
    args = parse_args()

    init_student_model_from_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        num_of_layers=args.num_of_layers,
        save_dir=args.save_dir,
        push_to_hub=args.push_to_hub
    )