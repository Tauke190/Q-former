"""
BLIP-2 OPT Inference Script

This script demonstrates how to:
1. Initialize the BLIP2-OPT model
2. Run forward pass (compute loss)
3. Generate text from images

Usage:
    python inference.py
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from models import Blip2OPT


def download_sample_image(url=None):
    """
    Download a sample image for testing.

    Args:
        url: Image URL (default: a sample dog image)

    Returns:
        PIL Image
    """
    if url is None:
        url = "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400"

    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to download image: {e}")
        print("Creating a blank image instead...")
        return Image.new('RGB', (224, 224), color='red')


def test_forward_pass():
    """
    Test the forward pass of BLIP2-OPT model.
    This computes the language modeling loss.
    """
    print("="*70)
    print("TEST 1: Forward Pass (Loss Computation)")
    print("="*70)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model with smaller OPT for faster testing
    print("\nInitializing BLIP2-OPT model...")
    print("- Vision Encoder: ViT-L/14 (frozen)")
    print("- Q-Former: 12 layers, 32 query tokens")
    print("- LLM: facebook/opt-125m (frozen, small for testing)")

    model = Blip2OPT(
        clip_model_name="ViT-L/14",
        opt_model_name="facebook/opt-125m",  # Small model for faster inference
        num_query_tokens=32,
        qformer_hidden_size=768,
        qformer_num_layers=12,
        qformer_num_heads=12,
        cross_attention_freq=2,
        device=device,
    )
    model = model.to(device)
    model.eval()

    print("\nModel initialized successfully!")

    # Prepare input
    print("\nPreparing inputs...")
    image = download_sample_image()
    text = ["A photo of a dog playing in the park"]

    print(f"Image size: {image.size}")
    print(f"Text: {text[0]}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(image, text)

    print("\n" + "="*70)
    print("FORWARD PASS RESULTS:")
    print("="*70)
    print(f"Loss (Language Modeling): {outputs['loss'].item():.4f}")
    print(f"Loss LM: {outputs['loss_lm'].item():.4f}")
    print("\n✓ Forward pass completed successfully!")

    return model


def test_generation(model):
    """
    Test text generation from images.
    """
    print("\n" + "="*70)
    print("TEST 2: Image-to-Text Generation")
    print("="*70)

    # Prepare input
    print("\nPreparing image...")
    image = download_sample_image()

    # Test 1: Generation without prompt
    print("\n--- Test 2.1: Generation without prompt ---")
    generated_text = model.generate(
        image=image,
        prompt="",
        max_new_tokens=30,
        num_beams=3,
        temperature=1.0,
    )
    print(f"Generated: {generated_text[0]}")

    # Test 2: Generation with prompt
    print("\n--- Test 2.2: Generation with prompt ---")
    prompt = "Question: What is in this image? Answer:"
    generated_text = model.generate(
        image=image,
        prompt=prompt,
        max_new_tokens=30,
        num_beams=3,
        temperature=1.0,
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text[0]}")

    # Test 3: Generation with different parameters
    print("\n--- Test 2.3: Generation with top-p sampling ---")
    generated_text = model.generate(
        image=image,
        prompt="Describe this image:",
        max_new_tokens=40,
        num_beams=1,
        top_p=0.9,
        temperature=0.8,
    )
    print(f"Generated: {generated_text[0]}")

    print("\n✓ Generation tests completed successfully!")


def test_batch_processing(model):
    """
    Test batch processing with multiple images.
    """
    print("\n" + "="*70)
    print("TEST 3: Batch Processing")
    print("="*70)

    # Prepare batch of images
    print("\nPreparing batch of 2 images...")
    image1 = download_sample_image()
    image2 = download_sample_image()

    # Stack images into batch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711))
    ])

    image_batch = torch.stack([transform(image1), transform(image2)]).to(model.device)
    texts = [
        "A beautiful landscape with mountains",
        "A cute dog playing in the park"
    ]

    print(f"Batch size: {image_batch.size(0)}")
    print(f"Image shape: {image_batch.shape}")

    # Forward pass with batch
    print("\nRunning forward pass on batch...")
    with torch.no_grad():
        outputs = model(image_batch, texts)

    print(f"\nBatch Loss: {outputs['loss'].item():.4f}")

    # Generation with batch
    print("\nGenerating captions for batch...")
    generated_texts = model.generate(
        image=image_batch,
        prompt="",
        max_new_tokens=25,
        num_beams=3,
    )

    for i, text in enumerate(generated_texts):
        print(f"Image {i+1}: {text}")

    print("\n✓ Batch processing completed successfully!")


def print_model_info(model):
    """
    Print model architecture information.
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")

    print("\nTrainable Components:")
    print("  - Q-Former (12 layers)")
    print("  - Query Tokens (32 tokens)")
    print("  - LLM Projector (768 -> LLM hidden size)")

    print("\nFrozen Components:")
    print("  - Vision Encoder (CLIP ViT-L/14)")
    print("  - OPT Language Model")

    print("\nModel Architecture:")
    print("  Image -> Vision Encoder -> Q-Former -> Projector -> LLM -> Text")


def main():
    """
    Main inference script.
    """
    print("\n" + "="*70)
    print("BLIP-2 OPT MODEL INFERENCE DEMONSTRATION")
    print("="*70)
    print("\nThis script will test:")
    print("1. Forward pass (loss computation)")
    print("2. Text generation from images")
    print("3. Batch processing")

    try:
        # Test 1: Forward pass
        model = test_forward_pass()

        # Print model info
        print_model_info(model)

        # Test 2: Generation
        test_generation(model)

        # Test 3: Batch processing
        test_batch_processing(model)

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe BLIP2-OPT model is working correctly!")
        print("You can now use this model for:")
        print("  - Image captioning")
        print("  - Visual question answering")
        print("  - Image-to-text generation")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
