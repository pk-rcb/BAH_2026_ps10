"""
train_controlnet.py — Train a ControlNet adapter for Stable Diffusion (Phase 2)
================================================================================
This script trains ONLY the ControlNet pathway, freezing the base SD backbone.
It uses Canny edges (edge_100m.npy) as the spatial condition to guide the
diffusion process towards reconstructing the real RGB image.

Memory footprint: ~10-12GB VRAM (batch size 1 + gradient accumulation)
Expected time: ~6-12 hours on an A100 for 10-20k steps.

Usage:
    python train_controlnet.py \
        --model_id "runwayml/stable-diffusion-v1-5" \
        --patches_dir output/patches \
        --output_dir weights/controlnet \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --max_train_steps 15000
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, random_split

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import TIRDataset


def collate_fn(examples):
    # dataset returns (edge [0,1], rgb [0,1])
    edges = torch.stack([ex[0] for ex in examples])
    rgbs  = torch.stack([ex[1] for ex in examples])
    return {"edges": edges, "rgbs": rgbs}


def train_controlnet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    # task='controlnet' returns (edge_100m in [0,1], rgb_100m in [0,1])
    dataset = TIRDataset(
        patches_dir=args.patches_dir,
        task='controlnet',
        stats_file=os.path.join(args.patches_dir, 'global_stats_color.json')
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    train_dataset, _, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )
    print(f"Training on {len(train_dataset)} samples.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.num_workers,
        pin_memory=True
    )

    # ── 2. Load Models ────────────────────────────────────────────────────────
    print(f"Loading backbone: {args.model_id}")
    
    # Load tokenizer and text encoder (we use empty prompts, but required by pipeline)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")

    # Initialize ControlNet from the UNet
    controlnet = ControlNetModel.from_unet(unet)

    # Freeze VAE, UNet, and text_encoder
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Move to device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    controlnet.to(device)

    # Optional: Enable xformers for memory efficient attention if installed
    # try:
    #     import xformers
    #     unet.enable_xformers_memory_efficient_attention()
    #     controlnet.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass

    # Optional: Gradient checkpointing for memory savings
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # ── 3. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_train_steps,
    )

    # ── 4. Training Loop ──────────────────────────────────────────────────────
    print("Starting training...")
    global_step = 0
    
    # Pre-tokenize empty prompt once
    empty_prompt_tokens = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(empty_prompt_tokens)[0]

    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate epochs based on max_train_steps
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            # VAE expects input in [-1, 1], dataset provides [0, 1]
            rgbs = batch["rgbs"].to(device)
            rgbs_norm = (rgbs * 2.0) - 1.0
            
            with torch.no_grad():
                latents = vae.encode(rgbs_norm).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            # Add noise to latents (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ControlNet conditioning (edges are [0,1] from dataset)
            controlnet_image = batch["edges"].to(device)

            # Expand text embeddings for batch size
            encoder_hidden_states_batch = encoder_hidden_states.repeat(bsz, 1, 1)

            # Forward pass ControlNet
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states_batch,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states_batch,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Loss
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{global_step}/{max_train_steps}] "
                          f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} "
                          f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step >= max_train_steps:
                    break
        
        if global_step >= max_train_steps:
            break

    # ── 5. Save Model ─────────────────────────────────────────────────────────
    print(f"Saving ControlNet to {args.output_dir}")
    controlnet.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ControlNet for IR colorization")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--patches_dir", type=str, default="output/patches")
    parser.add_argument("--output_dir", type=str, default="weights/controlnet")
    parser.add_argument("--train_batch_size", type=int, default=1, 
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Enable gradient checkpointing to save memory.")
    
    args = parser.parse_args()
    train_controlnet(args)
