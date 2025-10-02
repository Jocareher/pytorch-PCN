#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference for pytorch-PCN.

- Iterates through a directory with images and runs PCN.
- Draws the OBB on the original image (folder: <output_dir>/images)
- Saves predictions in txt files (folder: <output_dir>/labels), one line per detection:
    x1 y1 x2 y2 angle score
  (x1, y1, x2, y2 in pixels of the original image; angle in degrees; score in [0,1])

Example:
    python inference_pcn.py \
        --source ./some_images \
        --output_dir ./predictions_pcn \
        --min_score 0.01
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np

# - pcn/models.py -> load_model()
# - pcn/pcn.py    -> pcn_detect(img, nets)
# - pcn/utils.py  -> draw_face, Window
from pcn.models import load_model
from pcn.pcn import pcn_detect
from pcn.utils import draw_face, Window


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference for pytorch-PCN")
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory with images (common extensions are searched).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./predictions_pcn",
        help="Output directory with subfolders images/ and labels/",
    )
    p.add_argument(
        "--min_score",
        type=float,
        default=0.01,
        help="Minimum score threshold to save a prediction.",
    )
    p.add_argument(
        "--save_vis",
        action="store_true",
        help="If passed, also saves images with drawn OBB.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="(Optional) Limit the number of images to process (debug). -1 = all.",
    )
    return p.parse_args()


def list_images(root: Path):
    """
    List all image files in the given directory with supported extensions.

    Args:
        root (Path): The root directory to search for image files.

    Returns:
        List[Path]: A sorted list of image file paths with supported extensions.
    """
    # Supported image file extensions
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    # Initialize an empty list to store file paths
    files = []

    # Iterate over each extension pattern and collect matching files
    for pat in exts:
        files += list(root.glob(pat))

    # Sort the collected file paths for consistent processing order
    files = sorted(files)

    return files


def ensure_dirs(base_out: Path):
    """
    Ensure the existence of the required output directories.

    This function creates the necessary subdirectories for storing
    images and labels under the specified base output directory.

    Args:
        base_out (Path): The base output directory where subfolders
                         'images' and 'labels' will be created.

    Notes:
        - If the directories already exist, this function does nothing.
        - The 'parents=True' argument ensures that any missing parent
          directories are also created.
        - The 'exist_ok=True' argument prevents errors if the directories
          already exist.
    """
    # Create the 'images' subdirectory
    (base_out / "images").mkdir(parents=True, exist_ok=True)

    # Create the 'labels' subdirectory
    (base_out / "labels").mkdir(parents=True, exist_ok=True)


def window_to_xyxy_angle_score(win: Window):
    """
    Convert a PCN Window object to bounding box coordinates and additional attributes.

    The PCN model returns a Window object with the following attributes:
    - x, y: Top-left corner of the square bounding box.
    - width: Width (and height) of the square bounding box.
    - angle: Rotation angle of the bounding box in degrees.
    - score: Confidence score of the detection.

    This function converts the square bounding box to the format:
    (x1, y1, x2, y2, angle, score), where:
    - (x1, y1): Top-left corner of the bounding box.
    - (x2, y2): Bottom-right corner of the bounding box.
    - angle: Rotation angle in degrees.
    - score: Confidence score in the range [0, 1].

    Args:
        win (Window): A Window object containing detection information.

    Returns:
        Tuple[int, int, int, int, float, float]: A tuple containing:
            - x1 (int): Top-left x-coordinate of the bounding box.
            - y1 (int): Top-left y-coordinate of the bounding box.
            - x2 (int): Bottom-right x-coordinate of the bounding box.
            - y2 (int): Bottom-right y-coordinate of the bounding box.
            - angle (float): Rotation angle in degrees.
            - score (float): Confidence score in the range [0, 1].
    """
    # Top-left corner of the bounding box
    x1 = int(win.x)
    y1 = int(win.y)

    # Bottom-right corner of the bounding box
    x2 = int(win.x + win.width - 1)
    y2 = int(win.y + win.width - 1)

    # Rotation angle in degrees
    angle = float(win.angle)

    # Confidence score in the range [0, 1]
    score = float(win.score)

    return x1, y1, x2, y2, angle, score


def draw_windows(img: np.ndarray, wins):
    """
    Draw oriented bounding boxes (OBBs) on the input image.

    This function uses the `draw_face` utility from the repository to draw
    the oriented bounding boxes (OBBs) on a copy of the input image. The OBBs
    are defined by the `wins` parameter, which contains detection information.

    Args:
        img (np.ndarray): The input image as a NumPy array (BGR format).
        wins (List[Window]): A list of Window objects, each representing
                             a detected face with bounding box and attributes.

    Returns:
        np.ndarray: A copy of the input image with the OBBs drawn on it.
    """
    # Create a copy of the input image to avoid modifying the original
    vis = img.copy()

    # Iterate over each detection and draw the OBB on the image
    for w in wins:
        draw_face(vis, w)

    # Return the image with the drawn OBBs
    return vis


def main():
    """
    Main function for batch inference using pytorch-PCN.

    This function performs the following steps:
    1. Parses command-line arguments to get input and output directories,
       minimum score threshold, and other options.
    2. Ensures the output directory structure exists.
    3. Loads the PCN model for inference.
    4. Lists all images in the source directory.
    5. Processes each image:
       - Runs the PCN model to detect faces.
       - Filters detections based on the minimum score threshold.
       - Optionally saves visualizations with drawn oriented bounding boxes (OBBs).
       - Saves detection results in text files with the format:
         x1 y1 x2 y2 angle score
    6. Prints a summary of the results.

    Notes:
        - The PCN model detects faces and returns oriented bounding boxes (OBBs).
        - The output directory will contain two subfolders:
          - 'images/' for visualizations (if --save_vis is passed).
          - 'labels/' for text files with detection results.

    Raises:
        SystemExit: If no images are found in the source directory.
    """
    # Parse command-line arguments
    args = parse_args()
    src_dir = Path(args.source)
    out_dir = Path(args.output_dir)

    # Ensure the output directory structure exists
    ensure_dirs(out_dir)

    # Load the PCN model (default is CPU)
    nets = load_model()

    # List all images in the source directory
    imgs = list_images(src_dir)
    if args.max_images is not None and args.max_images > -1:
        imgs = imgs[: args.max_images]

    # Exit if no images are found
    if not imgs:
        print(f"[WARN] No images found in: {src_dir}")
        sys.exit(0)

    print(f"[INFO] Number of images found: {len(imgs)}")
    print(f"[INFO] Saving results to: {out_dir}")

    total_written = 0  # Counter for the number of processed images

    # Process each image
    for img_path in imgs:
        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        # Perform inference using the PCN model
        # pcn_detect returns a list of Window objects (x, y, width, angle, score)
        wins = pcn_detect(img, nets)

        # Filter detections based on the minimum score threshold
        wins = [w for w in wins if float(w.score) >= args.min_score]

        # Save visualization with drawn OBBs (if requested)
        if args.save_vis:
            vis = draw_windows(img, wins)
            cv2.imwrite(str(out_dir / "images" / img_path.name), vis)

        # Save detection results to a text file
        label_name = img_path.with_suffix(".txt").name
        label_path = out_dir / "labels" / label_name

        with open(label_path, "w") as f:
            for w in wins:
                x1, y1, x2, y2, angle, score = window_to_xyxy_angle_score(w)
                # Write detection in the format: x1 y1 x2 y2 angle score
                # Coordinates are integers, angle and score are floats
                f.write(f"{x1} {y1} {x2} {y2} {angle:.3f} {score:.6f}\n")

        print(f"[OK] {img_path.name}: {len(wins)} detections -> {label_name}")
        total_written += 1

    # Print summary of results
    print(f"\n[DONE] Number of .txt files written: {total_written}")
    if args.save_vis:
        print(f"[DONE] Images with OBBs saved in: {out_dir / 'images'}")


if __name__ == "__main__":
    main()
