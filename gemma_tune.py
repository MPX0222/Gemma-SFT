# -*- coding: gbk -*-

import sys

path_root = "gemma_model"
print(path_root)
sys.path.append(path_root)

import os
import argparse
import functools

CPU_NUMS = "10"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

ID_PAD = 0
ID_BOS = 2
ID_EOS = 1
ID_UNK = 3
ID_MASK = 4
ID_SOT = 106
ID_EOT = 107
ID_BR = 108
ID_USER = 1645
ID_MODEL = 2516

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import GenerationConfig, BitsAndBytesConfig, TrainingArguments, Trainer

from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm
import transformers
import torch

from gemma_sft.models.gemma.tokenization_gemma import GemmaTokenizer as LLMTokenizer
from gemma_sft.models.gemma.configuration_gemma import GemmaConfig as LLMConfig
from gemma_sft.models.gemma.modeling_gemma import GemmaForCausalLM as LLMModel


parser = argparse.ArgumentParser()
# Model Loading
parser.add_argument("--" +"base_model", type=str, default='', help="Gemma Weight Path")
# Data Loading
parser.add_argument("--" +"data_path", type=str, default='', help="Dataset Weight Path")
parser.add_argument("--" +"save_path", type=str, default='', help="Saving Path")
# Training Setting
parser.add_argument("--" +"batch_size",                   type=int, default=1, help="")
parser.add_argument("--" +"gradient_accumulation_steps",  type=int, default=2, help="")
parser.add_argument("--" +"learning_rate",                type=float, default=2e-4, help="")
parser.add_argument("--" +"lr_scheduler_type",            type=str, default="constant", help="[constant, cosine]")
parser.add_argument("--" +"optimizer",                    type=str, default="adamw_torch", help="[adamw_torch, adamw_hf]")
parser.add_argument("--" +"epochs",                       type=int, default=3, help="")
parser.add_argument("--" +"save_steps",                   type=int, default=10, help="")
parser.add_argument("--" +"warmup_steps",                 type=int, default=100, help="")
parser.add_argument("--" +"log_steps",                    type=int, default=10, help="")
parser.add_argument("--" +"max_source_length",            type=int, default=1023, help="")
parser.add_argument("--" +"max_target_length",            type=int, default=1023, help="")
parser.add_argument("--" +"max_length",                   type=int, default=2048, help="max_length = max_source_length + max_target_length - 2")
parser.add_argument("--" +"save_limit",                   type=int, default=5, help="")
# Other Setting
parser.add_argument("--" +"is_parallelizable",      type=bool, default=True, help="")
parser.add_argument("--" +"is_model_parallel",      type=bool, default=True, help="")
parser.add_argument("--" +"is_use_cache",           type=bool, default=True, help="")
parser.add_argument("--" +"gradient_checkpointing", type=bool, default=True, help="")
# Lora Setting
parser.add_argument("--" +"is_fp16",      type=bool, default=True, help="")
parser.add_argument("--" +"lora_r",       type=int, default=64, help="Lora_r")
parser.add_argument("--" +"lora_alpha",   type=int, default=16, help="Lora_alpha")
parser.add_argument("--" +"lora_dropout", type=float, default=0.1, help="Lora_dropout")

args = parser.parse_args(args=[])


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.safetensors"):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # save config
    if config:
        config.save_pretrained(model_save_dir)
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    
    torch.save(grad_params_dict, path_model)
    print("****** model_save_path is {} ******".format(path_model))

def print_named_parameters(model, use_print_data=False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print('='*150)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    print('='*150)

def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_headҲ��Ҫ��tf.float32(���һ��)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model

world_size = int(os.environ.get("WORLD_SIZE", 1))
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
ddp = world_size != 1

print('Loading model weights...')
model = LLMModel.from_pretrained(args.base_model)

print('Preparing model for kbit training...')

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = args.is_parallelizable
model.model_parallel = args.is_model_parallel
model.config.use_cache = args.is_use_cache

lora_modules = ["q_proj", "k_proj", "v_proj"]

print('Geting Lora modules...')
config = LoraConfig(target_modules=lora_modules,
                    lora_dropout=args.lora_dropout,
                    lora_alpha=args.lora_alpha,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=args.lora_r
                    )

model = get_peft_model(model, config)
model = model.cuda()

tokenizer = LLMTokenizer.from_pretrained(args.base_model, add_eos_token=True)
tokenizer.pad_token_id = ID_EOS
tokenizer.eos_token_id = ID_EOS
tokenizer.padding_side = "left"

def generate_prompt(example):

    instruction, text_input, text_output = example["instruction"], example["text_input"], example["text_output"]

    text_input = instruction + '\t' + text_input
    text_out = text_output

    prompt_text_1 = "<start_of_turn>user\n{}<end_of_turn>\n"
    prompt_text_2 = "<start_of_turn>model\n{}<end_of_turn>"
    
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out.strip())
    x = tokenizer.encode(text_1, add_special_tokens=False)
    y = tokenizer.encode(text_2, add_special_tokens=False)

    if len(x) + len(y) > (args.max_source_length + args.max_target_length):
        x = x[:args.max_source_length]
        y = y[:args.max_target_length]

    x = [ID_BOS] + x
    y = y + [ID_EOS]
    out = {"input_ids": x, "labels": y}

    return out

def data_collator(batch):
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) for i in range(len(batch))]
    len_max_batch = min(args.max_length, max(len_max_batch))

    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []

    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)

        labels = [-100] * len(x) + y + [-100] * len_padding
        input_ids = x + y + [ID_PAD] * len_padding
        attention_mask = [0] * (len(x)+len(y)) + [1] * len_padding

        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)

    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    
    input_dict = {"attention_mask": batch_attention_mask,  # no use
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    
    return input_dict

dataset = load_dataset("json", data_files=args.data_path)
dataset = dataset.map(generate_prompt, num_proc=10)

class GemmaTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """  newest loss """
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)  # if contain labels, will calculate loss

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    
train_args = TrainingArguments(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_grad_norm=1.0,
            logging_steps=args.log_steps,
            warmup_steps=args.warmup_steps,
            evaluation_strategy="no",
            lr_scheduler_type=args.lr_scheduler_type,
            logging_first_step=False,
            save_strategy="steps",
            save_total_limit=args.save_limit,
            save_steps=args.save_steps,
            ddp_find_unused_parameters=False if ddp else None,
            gradient_checkpointing=args.gradient_checkpointing,
            output_dir=args.save_path,
            report_to=["tensorboard"],
            optim=args.optimizer,
            fp16=args.is_fp16,
        )

print('=' * 90)
model.print_trainable_parameters()
print('=' * 90)

trainer = GemmaTrainer(
        data_collator=data_collator,
        train_dataset=dataset['train'],
        model=model,
        args=train_args
    )

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


trainer.train(resume_from_checkpoint=None)
save_model_state(model=model, config=config, model_save_dir=args.save_path)
# print_named_parameters(model, use_print_data=True)  # �鿴LoRA��Ȩ���ǲ���ΪNAN���
