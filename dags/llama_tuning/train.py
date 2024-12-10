"""Train Flyte Llama."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import math
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from dataclasses_json import dataclass_json
from dataclasses_json import DataClassJsonMixin
from peft import get_peft_model
from peft import LoraConfig
from peft import prepare_model_for_kbit_training
from llama_tuning.dataloader import get_dataset
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
import argparse

transformers.logging.set_verbosity_debug()


@dataclass
class HuggingFaceModelCard(DataClassJsonMixin):
    language: List[str]
    license: str  # valid licenses can be found at https://hf.co/docs/hub/repositories-licenses
    tags: List[str]


@dataclass
class PublishConfig(DataClassJsonMixin):
    repo_id: str
    readme: Optional[str] = None
    model_card: Optional[HuggingFaceModelCard] = None


@dataclass
class TrainerConfig(DataClassJsonMixin):
    model_path: str = "llama/Llama-3.1-8B-Instruct"
    data_dir: str = "./data"
    output_dir: str = "./output"
    checkpoint_dir: Optional[str] = None
    num_epochs: int = 20
    max_steps: int = -1
    batch_size: int = 8
    test_size: float = 0.01
    model_max_length: int = 1024
    seed: int = 41
    report_to: str = "none"
    device_map: Optional[str] = None
    gradient_accumulation_steps: int = 8
    padding: str = "right"
    dataloader_num_proc: int = 1
    use_fp16: bool = False
    use_4bit: bool = False
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj"])
    lora_dropout: float = 0.05
    debug: bool = False
    publish_config: Optional[PublishConfig] = field(default=None)


def train(
    config: TrainerConfig,
    pretrained_adapter: Optional[Path] = None,
    hf_auth_token: Optional[str] = None,
    **kwargs,
):
    print("Training model...")
    os.environ["HF_TOKEN"] = hf_auth_token

    # Setup distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            print(f"Training with {world_size} GPUs")

    # Only log to wandb from the main process
    if local_rank != 0:
        config.report_to = "none"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
        token=hf_auth_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Modify model loading parameters based on distributed setup
    load_model_params = {
        **kwargs,
        "token": hf_auth_token,
        "torch_dtype": dtype,
    }
    
    if is_distributed:
        load_model_params["device_map"] = {"": local_rank}
    else:
        load_model_params["device_map"] = "auto"

    if config.use_4bit:
        load_model_params["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_model_params,
    )

    optim = "adamw_torch"
    if config.use_qlora:
        optim = "paged_adamw_8bit"
        # Enable gradient checkpointing before DDP wrapping
        model.config.use_cache = False  # Required for gradient checkpointing
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        if local_rank == 0:
            print("LORA Config:")
            print(lora_config)
            model.print_trainable_parameters()

    # Move model to device and wrap with DDP if needed
    if is_distributed:
        model = model.to(local_rank)
        torch.distributed.barrier()
        model = DDP(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
        torch.distributed.barrier()

    def tokenize(examples):
        tokens = tokenizer(
            [f"{t}{tokenizer.eos_token}" for t in examples['text']],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        return tokens

    limit = 5 if config.debug else None
    dataset = (get_dataset(
        Path(config.data_dir).expanduser(),
        num_proc=config.dataloader_num_proc,
        limit=limit,
        block_size=config.model_max_length,
        skip_by=config.model_max_length,
    ).map(tokenize, batched=True, num_proc=config.dataloader_num_proc))

    # Remove unnecessary columns
    columns_to_remove = [
        col for col in dataset.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    print(f"Dataset size: {len(dataset)}")
    dataset_splits = dataset.train_test_split(test_size=config.test_size,
                                              seed=config.seed)

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    # Create a custom subclass of Trainer to prevent automatic gradient checkpointing
    class CustomTrainer(Trainer):
        def _wrap_model(self, model, training=True, dataloader=None):
            if is_distributed:
                return model
            return super()._wrap_model(model, training, dataloader)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",
        eval_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=config.use_fp16,
        half_precision_backend="auto",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=4,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        logging_steps=1,
        optim=optim,
        report_to=config.report_to,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        remove_unused_columns=False,
        local_rank=local_rank,
        ddp_backend="nccl",
        # Disable gradient checkpointing in training args since we handle it manually
        gradient_checkpointing=False,
        run_name=f"llama_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        data_collator=data_collator,
    )

    def get_latest_checkpoint(checkpoints_dir) -> Optional[str]:
        # Ensure the directory exists
        if not os.path.exists(checkpoints_dir):
            return None

        # List all subdirectories that match the checkpoint naming pattern
        subdirs = [
            os.path.join(checkpoints_dir, d)
            for d in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, d)) and
            d.startswith("checkpoint-")
        ]

        if not subdirs:
            return None

        # Check if checkpoint files exist
        for checkpoint_dir in sorted(subdirs, key=os.path.getmtime, reverse=True):
            if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
                return checkpoint_dir
        
        return None

    # Only the main process should check for checkpoints
    if local_rank == 0:
        checkpoint_dir = get_latest_checkpoint(config.output_dir)
        if checkpoint_dir:
            print(f"Found checkpoint: {checkpoint_dir}")
            # Verify checkpoint files exist
            if not (os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(checkpoint_dir, "model.safetensors"))):
                print("Warning: Checkpoint directory exists but model files are missing")
                checkpoint_dir = None
    else:
        checkpoint_dir = None

    # Ensure all processes use the same checkpoint decision
    if is_distributed:
        checkpoint_dir = torch.tensor(checkpoint_dir is not None, device=local_rank)
        torch.distributed.broadcast(checkpoint_dir, src=0)
        checkpoint_dir = "./sky_llama/output/checkpoint-376" if checkpoint_dir.item() else None

    if checkpoint_dir:
        print(f"Resuming from checkpoint: {checkpoint_dir}")
    
    # Start training
    trainer.train(resume_from_checkpoint=None)  # Disable automatic checkpoint loading
    if checkpoint_dir:
        # Load checkpoint manually
        state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), 
                              map_location=f"cuda:{local_rank}")
        model.load_state_dict(state_dict)

    eval_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model(training_args.output_dir)
def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='Train Llama model')
    parser.add_argument('--config', type=str, required=True, help='Training config as JSON string')
    parser.add_argument('--hf_auth_token', type=str, required=True, help='HuggingFace auth token')
    parser.add_argument('--pretrained_adapter', type=str, default=None, help='Path to pretrained adapter')
    
    try:
        args = parser.parse_args()
        print(f"Parsed arguments: {args}")
        
        config = TrainerConfig(**json.loads(args.config))
        print(f"Loaded config: {config}")
        
        pretrained_adapter = Path(args.pretrained_adapter) if args.pretrained_adapter else None
        
        print("Starting training...")
        train(
            config=config,
            pretrained_adapter=pretrained_adapter,
            hf_auth_token=args.hf_auth_token
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
