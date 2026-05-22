#!/usr/bin/env python3
"""Basic inference example."""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from fashn_vton import TryOnPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FASHN VTON v1.5 Inference"
    )

    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory containing model weights",
    )

    parser.add_argument(
        "--person-image",
        type=str,
        required=True,
        help="Path to person image",
    )

    parser.add_argument(
        "--garment-image",
        type=str,
        required=True,
        help="Path to garment image",
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=["tops", "bottoms", "one-pieces"],
        required=True,
        help="Garment category",
    )

    parser.add_argument(
        "--garment-photo-type",
        type=str,
        choices=["model", "flat-lay"],
        default="flat-lay",
        help="Type of garment image",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )

    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=20,
        help="Sampling steps",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="CFG guidance scale",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------

    person_path = Path(args.person_image)
    garment_path = Path(args.garment_image)
    weights_path = Path(args.weights_dir)

    if not person_path.exists():
        print(f"[ERROR] Person image not found: {person_path}")
        sys.exit(1)

    if not garment_path.exists():
        print(f"[ERROR] Garment image not found: {garment_path}")
        sys.exit(1)

    if not weights_path.exists():
        print(f"[ERROR] Weights directory not found: {weights_path}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------

    print("\nLoading images...")

    person_image = Image.open(person_path).convert("RGB")
    garment_image = Image.open(garment_path).convert("RGB")

    # ---------------------------------------------------------
    # Load pipeline
    # ---------------------------------------------------------

    print("\nLoading TryOnPipeline...")

    start_time = time.time()

    pipeline = TryOnPipeline(
        weights_dir=str(weights_path),
        device=args.device,
    )

    print(f"Pipeline loaded in {time.time() - start_time:.2f}s")

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    print("\nRunning inference...")

    start_time = time.time()

    result = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=args.category,
        garment_photo_type=args.garment_photo_type,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    inference_time = time.time() - start_time

    print(f"Inference completed in {inference_time:.2f}s")

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "result.png"

    result.images[0].save(output_path)

    print(f"\nSaved result to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()


# python examples\basic_inference.py --weights-dir ./weights --person-image ./examples/data/model.webp --garment-image ./examples/data/garment.webp --category tops