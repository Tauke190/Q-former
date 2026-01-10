"""
BLIP-2 Q-Former Training Script

Trains the Q-Former model on Flickr8k dataset with three objectives:
- Image-Text Contrastive Learning (ITC)
- Image-Text Matching (ITM)
- Image-grounded Text Generation (ITG)
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.blip2_qformer import Blip2Qformer
from dataset.flickr import Flickr8kDataset

def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """Collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_names = [item['image_name'] for item in batch]

    return {
        'image': images,
        'caption': captions,
        'image_name': image_names,
    }

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: Blip2Qformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_loss_itc = 0.0
    total_loss_itm = 0.0
    total_loss_itg = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        captions = batch['caption']

        # Forward pass
        outputs = model(images, captions)

        loss = outputs['loss'] / gradient_accumulation_steps
        loss_itc = outputs['loss_itc'].item()
        loss_itm = outputs['loss_itm'].item()
        loss_itg = outputs['loss_itg'].item()

        # Backward pass
        loss.backward()

        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Accumulate losses
        total_loss += outputs['loss'].item()
        total_loss_itc += loss_itc
        total_loss_itm += loss_itm
        total_loss_itg += loss_itg
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{outputs['loss'].item():.4f}",
            'itc': f"{loss_itc:.4f}",
            'itm': f"{loss_itm:.4f}",
            'itg': f"{loss_itg:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return {
        'loss': total_loss / num_batches,
        'loss_itc': total_loss_itc / num_batches,
        'loss_itm': total_loss_itm / num_batches,
        'loss_itg': total_loss_itg / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: Blip2Qformer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on validation set"""
    model.eval()

    total_loss = 0.0
    total_loss_itc = 0.0
    total_loss_itm = 0.0
    total_loss_itg = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        images = batch['image'].to(device)
        captions = batch['caption']

        outputs = model(images, captions)

        total_loss += outputs['loss'].item()
        total_loss_itc += outputs['loss_itc'].item()
        total_loss_itm += outputs['loss_itm'].item()
        total_loss_itg += outputs['loss_itg'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'loss_itc': total_loss_itc / num_batches,
        'loss_itm': total_loss_itm / num_batches,
        'loss_itg': total_loss_itg / num_batches,
    }


def save_checkpoint(
    model: Blip2Qformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: Blip2Qformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
) -> int:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logging.info(f"Loaded checkpoint from {checkpoint_path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


# ============================================================================
# Main Training Script
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 Q-Former on Flickr8k")

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Flickr8k_Dataset",
        help="Path to Flickr8k dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="CLIP model variant to use",
    )
    parser.add_argument(
        "--num_query_tokens",
        type=int,
        default=32,
        help="Number of query tokens for Q-Former",
    )
    parser.add_argument(
        "--qformer_hidden_size",
        type=int,
        default=768,
        help="Q-Former hidden dimension",
    )
    parser.add_argument(
        "--qformer_num_layers",
        type=int,
        default=12,
        help="Number of Q-Former layers",
    )
    parser.add_argument(
        "--max_txt_len",
        type=int,
        default=32,
        help="Maximum text sequence length",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )

    # Other arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size (for debugging)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "training.log"),
        ],
    )


    

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model
    logging.info("Initializing model...")
    model = Blip2Qformer(
        clip_model_name=args.clip_model,
        num_query_tokens=args.num_query_tokens,
        qformer_hidden_size=args.qformer_hidden_size,
        qformer_num_layers=args.qformer_num_layers,
        max_txt_len=args.max_txt_len,
        device=device,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    logging.info("Loading datasets...")
    train_dataset = Flickr8kDataset(
        data_root=args.data_root,
        split="train",
        transform=model.preprocess,
        max_samples=args.max_samples,
    )

    val_dataset = Flickr8kDataset(
        data_root=args.data_root,
        split="val",
        transform=model.preprocess,
        max_samples=args.max_samples // 10 if args.max_samples else None,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    # Initialize optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.gradient_accumulation_steps

    # Learning rate scheduler: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)

    # Training loop
    logging.info("Intialize/Test Validation...")

    val_metrics = evaluate(model, val_loader, device)

    logging.info(
        f"Val - Loss: {val_metrics['loss']:.4f}, "
        f"ITC: {val_metrics['loss_itc']:.4f}, "
        f"ITM: {val_metrics['loss_itm']:.4f}, "
        f"ITG: {val_metrics['loss_itg']:.4f}"
    )



    logging.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )

        logging.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"ITC: {train_metrics['loss_itc']:.4f}, "
            f"ITM: {train_metrics['loss_itm']:.4f}, "
            f"ITG: {train_metrics['loss_itg']:.4f}"
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device)

            logging.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"ITC: {val_metrics['loss_itc']:.4f}, "
                f"ITM: {val_metrics['loss_itm']:.4f}, "
                f"ITG: {val_metrics['loss_itg']:.4f}"
            )

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    save_path=os.path.join(args.output_dir, "best_model.pt"),
                )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=train_metrics,
                save_path=os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        metrics=train_metrics,
        save_path=os.path.join(args.output_dir, "final_model.pt"),
    )

    logging.info("\nTraining complete!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
