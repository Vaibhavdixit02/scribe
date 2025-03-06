import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig, get_scheduler
import wandb
from accelerate import Accelerator

from data.mind2web_dataset import create_mind2web_dataloader
from models.cross_modal_model import CrossModalWebAgent
from utils.visual_processor import VisualProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Train a cross-modal web agent")
    parser.add_argument("--config", type=str, default="configs/multimodal_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    return parser.parse_args()



def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
    )
    
    # Set up logging
    if accelerator.is_main_process:
        if config["training"].get("use_wandb", False):
            wandb.init(
                project=config["project_name"],
                name=config["run_name"],
                config=config
            )
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Set up TensorBoard
        tb_writer = SummaryWriter(os.path.join(config["output_dir"], "logs"))
    
    # Set device
    device = accelerator.device
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["text_model_name"],
        use_fast=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize visual processor
    visual_processor = VisualProcessor(
        vision_model_name=config["model"]["vision_model_name"]
    )
    
    # Create dataloaders
    train_dataloader = create_mind2web_dataloader(
        data_dir=config["data"]["data_dir"],
        split="train",
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        max_length=config["model"]["max_length"]
    )
    
    val_dataloader = create_mind2web_dataloader(
        data_dir=config["data"]["data_dir"],
        split="test_task",  # Using test_task as validation
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        max_length=config["model"]["max_length"]
    )
    
    # Initialize model
    model = CrossModalWebAgent(
        text_model_name=config["model"]["text_model_name"],
        vision_model_name=config["model"]["vision_model_name"],
        use_lora=config["model"]["use_lora"],
        lora_rank=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"]
    )
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Set up learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // config["training"]["gradient_accumulation_steps"]
    max_train_steps = config["training"]["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=config["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=max_train_steps
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                labels=batch["labels"]
            )
            
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / config["training"]["gradient_accumulation_steps"]
            accelerator.backward(loss)
            
            # Update weights
            if (step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log metrics
                if accelerator.is_main_process and global_step % config["training"]["logging_steps"] == 0:
                    train_loss += loss.item() * config["training"]["gradient_accumulation_steps"]
                    current_lr = optimizer.param_groups[0]["lr"]
                    
                    # Log to TensorBoard
                    tb_writer.add_scalar("train/loss", train_loss / (step + 1), global_step)
                    tb_writer.add_scalar("train/lr", current_lr, global_step)
                    
                    # Log to wandb
                    if config["training"].get("use_wandb", False):
                        wandb.log({
                            "train/loss": train_loss / (step + 1),
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step
                        })
                    
                    print(f"Epoch: {epoch}, Step: {global_step}, Loss: {train_loss / (step + 1):.4f}, LR: {current_lr:.6f}")
            
            # Evaluate
            if global_step % config["training"]["eval_steps"] == 0:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_dataloader):
                        val_outputs = model(
                            input_ids=val_batch["input_ids"],
                            attention_mask=val_batch["attention_mask"],
                            visual_features=val_batch["visual_features"],
                            labels=val_batch["labels"]
                        )
                        
                        val_loss += val_outputs["loss"].item()
                
                val_loss /= len(val_dataloader)
                
                # Log validation metrics
                if accelerator.is_main_process:
                    # Log to TensorBoard
                    tb_writer.add_scalar("val/loss", val_loss, global_step)
                    
                    # Log to wandb
                    if config["training"].get("use_wandb", False):
                        wandb.log({
                            "val/loss": val_loss,
                            "val/epoch": epoch,
                            "val/global_step": global_step
                        })
                    
                    print(f"Validation - Epoch: {epoch}, Step: {global_step}, Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                        # Save checkpoint
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Save model
                        model_path = os.path.join(config["output_dir"], "best_model")
                        os.makedirs(model_path, exist_ok=True)
                        unwrapped_model.save_pretrained(model_path)
                        
                        # Save tokenizer
                        tokenizer.save_pretrained(model_path)
                        
                        print(f"New best model saved with val_loss: {val_loss:.4f}")
                
                model.train()
        
        # Save checkpoint at end of epoch
        if accelerator.is_main_process:
            # Save checkpoint
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save model
            model_path = os.path.join(config["output_dir"], f"checkpoint-epoch-{epoch}")
            os.makedirs(model_path, exist_ok=True)
            unwrapped_model.save_pretrained(model_path)
            
            # Save tokenizer
            tokenizer.save_pretrained(model_path)
            
            print(f"Checkpoint saved at epoch {epoch}")
    
    # Save final model
    if accelerator.is_main_process:
        # Save checkpoint
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save model
        model_path = os.path.join(config["output_dir"], "final_model")
        os.makedirs(model_path, exist_ok=True)
        unwrapped_model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(model_path)
        
        print(f"Final model saved")
        
        # Close TensorBoard writer
        tb_writer.close()
        
        # Close wandb
        if config["training"].get("use_wandb", False):
            wandb.finish()

if __name__ == "__main__":
    main()