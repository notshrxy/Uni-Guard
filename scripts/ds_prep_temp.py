"""
Dataset Preparation Utilities
==============================
Tools for splitting, validating, and analyzing the dataset.
Script used to split the dataset based on requirement

This Python file is a dataset preparation tool for training an object detection model (like YOLOv8).

It mainly does two things:

1️⃣ Split a dataset into train / validation / test sets
2️⃣ Validate that the dataset labels follow YOLO format correctly

Think of it as a utility script used before training your model.

Basically takes labelled images and splits the set into train, validation and test

"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter


def split_dataset(source_images, source_labels, output_dir, train=0.8, val=0.1, test=0.1, seed=42):
    """Split a YOLO-format dataset into train/validate/test.

    Expected input structure:
        source_images/   → *.jpg files
        source_labels/   → *.txt files (YOLO format)

        (Mapping labels to corresponding sources)

    Output structure:
        output_dir/ (Sample)
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/

    Arguments:
        source_images: Path to images directory.
        source_labels: Path to labels directory.
        output_dir: Output dataset root.
        train: Train split ratio.
        val: Validation split ratio.
        test: Test split ratio.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Find all images with matching labels
    images = sorted(Path(source_images).glob('*.jpg'))
    pairs = []
    for img_path in images:
        label_path = Path(source_labels) / (img_path.stem + '.txt')
        if label_path.exists():
            pairs.append((img_path, label_path))

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:],
    }

    output = Path(output_dir)
    for split_name, split_pairs in splits.items():
        img_dir = output / 'images' / split_name
        lbl_dir = output / 'labels' / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

        print(f"  {split_name}: {len(split_pairs)} images")

    print(f"\nDataset split complete → {output}")
    print(f"Total: {n} | Train: {n_train} | Val: {n - n_train - len(splits['test'])} | Test: {len(splits['test'])}")


def validate_dataset(dataset_dir, num_classes=3):
    """Validate YOLO dataset structure and labels.

    Args:
        dataset_dir: Root dataset directory.
        num_classes: Expected number of classes (0: ID Card, 1: Person, 2: Shoes).

    Returns:
        dict with validation results.
    """
    dataset = Path(dataset_dir)
    issues = []
    stats = {'total_images': 0, 'total_labels': 0, 'class_counts': Counter(), 'empty_labels': 0}

    for split in ['train', 'val', 'test']:
        img_dir = dataset / 'images' / split
        lbl_dir = dataset / 'labels' / split

        if not img_dir.exists():
            issues.append(f"Missing directory: {img_dir}")
            continue

        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        stats['total_images'] += len(images)

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + '.txt')

            if not lbl_path.exists():
                issues.append(f"Missing label: {lbl_path.name}")
                continue

            stats['total_labels'] += 1

            with open(lbl_path, 'r') as f:
                lines = f.read().strip().split('\n')

            if not lines or lines == ['']:
                stats['empty_labels'] += 1
                continue

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"{lbl_path.name}:{line_num} — Expected 5 values, got {len(parts)}")
                    continue

                cls = int(parts[0])
                if cls < 0 or cls >= num_classes:
                    issues.append(f"{lbl_path.name}:{line_num} — Invalid class {cls}")

                values = [float(v) for v in parts[1:]]
                for v in values:
                    if v < 0 or v > 1:
                        issues.append(f"{lbl_path.name}:{line_num} — Value out of range: {v}")

                stats['class_counts'][cls] += 1

    print(f"\n{'='*50}")
    print(f"  Dataset Validation: {dataset_dir}")
    print(f"{'='*50}")
    print(f"  Images:       {stats['total_images']}")
    print(f"  Labels:       {stats['total_labels']}")
    print(f"  Empty labels: {stats['empty_labels']}")
    print(f"  Class dist:   {dict(stats['class_counts'])}")
    print(f"  Issues:       {len(issues)}")
    if issues:
        for issue in issues[:20]:
            print(f"    - {issue}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
    print()

    return {'stats': stats, 'issues': issues}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset utilities')
    sub = parser.add_subparsers(dest='command')

    sp = sub.add_parser('split', help='Split dataset')
    sp.add_argument('--images', required=True, help='Source images directory')
    sp.add_argument('--labels', required=True, help='Source labels directory')
    sp.add_argument('--output', required=True, help='Output directory')
    sp.add_argument('--train', type=float, default=0.8)
    sp.add_argument('--val', type=float, default=0.1)
    sp.add_argument('--test', type=float, default=0.1)

    vp = sub.add_parser('validate', help='Validate dataset')
    vp.add_argument('--dataset', required=True, help='Dataset root directory')
    vp.add_argument('--classes', type=int, default=3)

    args = parser.parse_args()

    if args.command == 'split':
        split_dataset(args.images, args.labels, args.output, args.train, args.val, args.test)
    elif args.command == 'validate':
        validate_dataset(args.dataset, args.classes)

"""
This Python file is a dataset preparation tool for training an object detection model (like YOLOv8).

It mainly does two things:

1️⃣ Split a dataset into train / validation / test sets
2️⃣ Validate that the dataset labels follow YOLO format correctly

Think of it as a utility script used before training your model.

Basically takes labelled images and splits the set into train, validation and test

"""