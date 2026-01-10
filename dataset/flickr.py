import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image
import torch
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    """
    Flickr8k Dataset for BLIP-2 Q-Former training.

    Loads images and their corresponding captions from the Flickr8k dataset.
    Each image has 5 captions, and we randomly sample one during training.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: Path to Flickr8k_Dataset folder
            split: One of 'train', 'val', 'test'
            transform: Image transform (from CLIP preprocessing)
            max_samples: Limit number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.images_dir = self.data_root / "images"

        # Load captions: image_name -> list of captions
        self.image_captions = self._load_captions()

        # Load split image list
        self.image_list = self._load_split_images(split)

        # Filter to only images that exist and have captions
        self.image_list = [
            img for img in self.image_list
            if img in self.image_captions and (self.images_dir / img).exists()
        ]

        if max_samples is not None:
            self.image_list = self.image_list[:max_samples]

        logging.info(f"Loaded {len(self.image_list)} images for {split} split")

    def _load_captions(self) -> Dict[str, List[str]]:
        """Load captions from Flickr8k.token.txt"""
        captions_file = self.data_root / "Flickr8k.token.txt"
        image_captions = defaultdict(list)

        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: image_name#caption_id\tcaption_text
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                image_id = parts[0].split('#')[0]  # Remove caption index
                caption = parts[1]
                image_captions[image_id].append(caption)

        return dict(image_captions)

    def _load_split_images(self, split: str) -> List[str]:
        """Load image list for the given split"""
        split_map = {
            'train': 'Flickr_8k.trainImages.txt',
            'val': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt',
        }

        split_file = self.data_root / split_map[split]
        with open(split_file, 'r') as f:
            images = [line.strip() for line in f if line.strip()]

        return images

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        image_name = self.image_list[idx]
        image_path = self.images_dir / image_name

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Random caption selection during training
        captions = self.image_captions[image_name]
        caption = random.choice(captions)

        return {
            'image': image,
            'caption': caption,
            'image_name': image_name,
        }


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

