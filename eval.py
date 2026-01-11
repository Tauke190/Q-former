"""
BLIP-2 Q-Former Evaluation Script

Evaluates the trained Q-Former model on the Flickr8k test set with:
- Loss metrics (ITC, ITM, ITG)
- Image-to-Text Retrieval (Recall@1, R@5, R@10)
- Text-to-Image Retrieval (Recall@1, R@5, R@10)
- ITM Accuracy

Usage:
    python eval.py --checkpoint checkpoints/best_model.pt --data_root data/Flickr8k_Dataset
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np

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


@torch.no_grad()
def compute_losses(
    model: Blip2Qformer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute average losses on the dataset"""
    model.eval()

    total_loss = 0.0
    total_loss_itc = 0.0
    total_loss_itm = 0.0
    total_loss_itg = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Computing Losses")

    for batch in pbar:
        images = batch['image'].to(device)
        captions = batch['caption']

        outputs = model(images, captions)

        total_loss += outputs['loss'].item()
        total_loss_itc += outputs['loss_itc'].item()
        total_loss_itm += outputs['loss_itm'].item()
        total_loss_itg += outputs['loss_itg'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{outputs['loss'].item():.4f}",
        })

    return {
        'loss': total_loss / num_batches,
        'loss_itc': total_loss_itc / num_batches,
        'loss_itm': total_loss_itm / num_batches,
        'loss_itg': total_loss_itg / num_batches,
    }


@torch.no_grad()
def extract_features(
    model: Blip2Qformer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract image and text features for retrieval evaluation"""
    model.eval()

    all_image_feats = []
    all_text_feats = []
    all_image_names = []

    pbar = tqdm(dataloader, desc="Extracting Features")

    for batch in pbar:
        images = batch['image'].to(device)
        captions = batch['caption']
        image_names = batch['image_name']

        # Encode images
        image_embeds, image_atts = model.encode_image(images)
        batch_size = image_embeds.size(0)
        query_tokens = model.query_tokens.expand(batch_size, -1, -1)

        # Get image features through Q-Former
        query_output = model.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            model.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # Tokenize and get text features
        text_tokens = model.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = model.Qformer(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feats = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        all_image_feats.append(image_feats.cpu())
        all_text_feats.append(text_feats.cpu())
        all_image_names.extend(image_names)

    return {
        'image_feats': torch.cat(all_image_feats, dim=0),
        'text_feats': torch.cat(all_text_feats, dim=0),
        'image_names': all_image_names,
    }


def compute_retrieval_metrics(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute image-text and text-image retrieval metrics"""
    num_samples = image_feats.size(0)

    # Compute similarity matrix
    sim_matrix = image_feats @ text_feats.t()  # [N, N]

    # Image-to-Text Retrieval (i2t)
    # For each image, find the rank of its corresponding text
    i2t_ranks = []
    for i in range(num_samples):
        sim_scores = sim_matrix[i]
        sorted_indices = torch.argsort(sim_scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)

    # Text-to-Image Retrieval (t2i)
    t2i_ranks = []
    for i in range(num_samples):
        sim_scores = sim_matrix[:, i]
        sorted_indices = torch.argsort(sim_scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)

    t2i_ranks = np.array(t2i_ranks)

    metrics = {}

    # Compute Recall@K for Image-to-Text
    for k in k_values:
        recall = (i2t_ranks <= k).sum() / num_samples * 100
        metrics[f'i2t_recall@{k}'] = recall

    # Compute Recall@K for Text-to-Image
    for k in k_values:
        recall = (t2i_ranks <= k).sum() / num_samples * 100
        metrics[f't2i_recall@{k}'] = recall

    # Mean Rank and Median Rank
    metrics['i2t_mean_rank'] = i2t_ranks.mean()
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_mean_rank'] = t2i_ranks.mean()
    metrics['t2i_median_rank'] = np.median(t2i_ranks)

    return metrics


@torch.no_grad()
def compute_itm_accuracy(
    model: Blip2Qformer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute ITM accuracy on positive pairs"""
    model.eval()

    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Computing ITM Accuracy")

    for batch in pbar:
        images = batch['image'].to(device)
        captions = batch['caption']
        batch_size = images.size(0)

        # Encode images
        image_embeds, image_atts = model.encode_image(images)
        query_tokens = model.query_tokens.expand(batch_size, -1, -1)

        # Tokenize text
        text_tokens = model.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # Forward through Q-Former with both image and text (positive pairs)
        pos_output = model.Qformer(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Get ITM predictions
        itm_logits = model.itm_head(pos_output.last_hidden_state[:, 0, :])
        predictions = itm_logits.argmax(dim=1)

        # For positive pairs, label should be 1
        correct += (predictions == 1).sum().item()
        total += batch_size

    return {
        'itm_accuracy': correct / total * 100,
        'itm_correct': correct,
        'itm_total': total,
    }


def load_model(
    checkpoint_path: str,
    device: torch.device,
    clip_model: str = "ViT-L/14",
    num_query_tokens: int = 32,
    qformer_hidden_size: int = 768,
    qformer_num_layers: int = 12,
    max_txt_len: int = 32,
) -> Blip2Qformer:
    """Load trained model from checkpoint"""
    logging.info(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = Blip2Qformer(
        clip_model_name=clip_model,
        num_query_tokens=num_query_tokens,
        qformer_hidden_size=qformer_hidden_size,
        qformer_num_layers=qformer_num_layers,
        max_txt_len=max_txt_len,
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            logging.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP-2 Q-Former on Flickr8k test set")

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Flickr8k_Dataset",
        help="Path to Flickr8k dataset root",
    )

    # Model arguments
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="CLIP model variant",
    )
    parser.add_argument(
        "--num_query_tokens",
        type=int,
        default=32,
        help="Number of query tokens",
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

    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--skip_retrieval",
        action="store_true",
        help="Skip retrieval metrics computation (faster)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        clip_model=args.clip_model,
        num_query_tokens=args.num_query_tokens,
        qformer_hidden_size=args.qformer_hidden_size,
        qformer_num_layers=args.qformer_num_layers,
        max_txt_len=args.max_txt_len,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Load test dataset
    logging.info(f"Loading {args.split} dataset...")
    test_dataset = Flickr8kDataset(
        data_root=args.data_root,
        split=args.split,
        transform=model.preprocess,
        max_samples=args.max_samples,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logging.info(f"Test samples: {len(test_dataset)}")

    # Initialize results
    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(test_dataset),
    }

    # Compute losses
    print("\n" + "=" * 60)
    print("COMPUTING LOSSES")
    print("=" * 60)
    loss_metrics = compute_losses(model, test_loader, device)
    results.update(loss_metrics)

    print(f"\nTotal Loss: {loss_metrics['loss']:.4f}")
    print(f"  ITC Loss: {loss_metrics['loss_itc']:.4f}")
    print(f"  ITM Loss: {loss_metrics['loss_itm']:.4f}")
    print(f"  ITG Loss: {loss_metrics['loss_itg']:.4f}")

    # Compute ITM accuracy
    print("\n" + "=" * 60)
    print("COMPUTING ITM ACCURACY")
    print("=" * 60)
    itm_metrics = compute_itm_accuracy(model, test_loader, device)
    results.update(itm_metrics)

    print(f"\nITM Accuracy: {itm_metrics['itm_accuracy']:.2f}%")
    print(f"  Correct: {itm_metrics['itm_correct']}/{itm_metrics['itm_total']}")

    # Compute retrieval metrics
    if not args.skip_retrieval:
        print("\n" + "=" * 60)
        print("COMPUTING RETRIEVAL METRICS")
        print("=" * 60)

        features = extract_features(model, test_loader, device)
        retrieval_metrics = compute_retrieval_metrics(
            features['image_feats'],
            features['text_feats'],
            k_values=[1, 5, 10],
        )
        results.update(retrieval_metrics)

        print("\nImage-to-Text Retrieval:")
        print(f"  Recall@1:  {retrieval_metrics['i2t_recall@1']:.2f}%")
        print(f"  Recall@5:  {retrieval_metrics['i2t_recall@5']:.2f}%")
        print(f"  Recall@10: {retrieval_metrics['i2t_recall@10']:.2f}%")
        print(f"  Mean Rank: {retrieval_metrics['i2t_mean_rank']:.2f}")
        print(f"  Median Rank: {retrieval_metrics['i2t_median_rank']:.2f}")

        print("\nText-to-Image Retrieval:")
        print(f"  Recall@1:  {retrieval_metrics['t2i_recall@1']:.2f}%")
        print(f"  Recall@5:  {retrieval_metrics['t2i_recall@5']:.2f}%")
        print(f"  Recall@10: {retrieval_metrics['t2i_recall@10']:.2f}%")
        print(f"  Mean Rank: {retrieval_metrics['t2i_mean_rank']:.2f}")
        print(f"  Median Rank: {retrieval_metrics['t2i_median_rank']:.2f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(test_dataset)}")
    print(f"Total Loss: {loss_metrics['loss']:.4f}")
    print(f"ITM Accuracy: {itm_metrics['itm_accuracy']:.2f}%")
    if not args.skip_retrieval:
        print(f"I2T R@1: {retrieval_metrics['i2t_recall@1']:.2f}%")
        print(f"T2I R@1: {retrieval_metrics['t2i_recall@1']:.2f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
