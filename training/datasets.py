

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


# --- Pose -----------------------------------------------------------------

def _default_pose_train_transform(input_size: int,
                                  mean: List[float],
                                  std: List[float]) -> Callable:
    """
    Color-only augmentations. NO horizontal flip (it would invert pose
    labels) and NO rotation beyond a few degrees (most roads aren't tilted).
    """
    return transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32), antialias=True),
        transforms.RandomCrop((input_size, input_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _default_pose_val_transform(input_size: int,
                                mean: List[float],
                                std: List[float]) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class PoseDataset(Dataset):
    """
    Directory-per-class dataset.

        root/
            BL/*.jpg
            BR/*.jpg
            ...

    Classes are discovered alphabetically (matches ModelConfig.pose.classes
    which the config validator enforces is sorted — so training labels and
    inference labels line up).
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str | Path, transform: Callable | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Pose dataset root does not exist: {self.root}")

        self.classes: List[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        if not self.classes:
            raise RuntimeError(f"No class subdirectories found in {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            for p in sorted((self.root / c).iterdir()):
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
        if self.transform is None:
            # Default: minimal preprocessing
            im = transforms.functional.to_tensor(im)
        else:
            im = self.transform(im)
        return im, label

    def class_counts(self) -> List[int]:
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts


# --- Triplet --------------------------------------------------------------

class TripletDataset(Dataset):
    """
    (anchor, positive, negative) dataset for Siamese training.

    Layout:
        root/
            vehicle_0001/*.jpg   # multiple photos of the SAME car
            vehicle_0002/*.jpg
            ...

    Each directory = one vehicle identity.
    Negative sampling is random across other identities. This is
    intentionally "easy mining" at the Dataset level; the TRAINER does
    batch-hard mining on top (see train_siamese.py).

    Also yields the identity index (group id) so the trainer can apply
    batch-hard triplet loss without re-deriving grouping.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        rng_seed: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Triplet dataset root does not exist: {self.root}")

        self.identities: List[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self.id_to_idx = {name: i for i, name in enumerate(self.identities)}

        self.paths_by_id: dict[int, List[Path]] = {}
        for name in self.identities:
            paths = [
                p for p in sorted((self.root / name).iterdir())
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS
            ]
            # Need at least 2 images to form (anchor, positive)
            if len(paths) >= 2:
                self.paths_by_id[self.id_to_idx[name]] = paths

        if len(self.paths_by_id) < 2:
            raise RuntimeError(
                f"Need at least 2 identities each with ≥2 images under {self.root}"
            )

        self.identity_ids: List[int] = sorted(self.paths_by_id.keys())
        self.transform = transform
        self.rng = random.Random(rng_seed)

        # Enumerate (id, image_index) pairs so __len__ scales with data,
        # not with identity count. Used as the "anchor" index.
        self.anchors: List[Tuple[int, int]] = []
        for iid, paths in self.paths_by_id.items():
            for i in range(len(paths)):
                self.anchors.append((iid, i))

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        anchor_id, anchor_i = self.anchors[idx]
        anchor_paths = self.paths_by_id[anchor_id]

        # Positive: another image from same identity
        pos_i = anchor_i
        while pos_i == anchor_i:
            pos_i = self.rng.randrange(len(anchor_paths))
        pos_path = anchor_paths[pos_i]

        # Negative: random image from a different identity
        neg_ids = [i for i in self.identity_ids if i != anchor_id]
        neg_id = self.rng.choice(neg_ids)
        neg_paths = self.paths_by_id[neg_id]
        neg_path = neg_paths[self.rng.randrange(len(neg_paths))]

        anchor = self._load(anchor_paths[anchor_i])
        pos = self._load(pos_path)
        neg = self._load(neg_path)
        return anchor, pos, neg, anchor_id

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
        if self.transform is None:
            return transforms.functional.to_tensor(im)
        return self.transform(im)


def default_siamese_train_transform(input_size: int,
                                    mean: List[float],
                                    std: List[float]) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size + 16, input_size + 16), antialias=True),
        transforms.RandomCrop((input_size, input_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def default_siamese_val_transform(input_size: int,
                                  mean: List[float],
                                  std: List[float]) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
