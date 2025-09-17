# tiny_imagenet_3epoch_compare.py
import os, math, time, random, zipfile, shutil, hashlib
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from urllib.request import urlretrieve

# ---------------------------
# Extra dependency (timm for ViT)
# ---------------------------
try:
    import timm
except ImportError as e:
    raise RuntimeError(
        "This script requires 'timm'. Install it via: pip install timm"
    ) from e

# ---------------------------
# Utils (seed, counts, meter)
# ---------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@dataclass(frozen=True)
class TrainConfig:
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    tiny_url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    tiny_dirname: str = "tiny-imagenet-200"
    num_classes: int = 200

# ---------------------------
# Tiny ImageNet download & prepare
# ---------------------------
def _download_if_needed(url: str, dst_zip: str) -> None:
    if os.path.exists(dst_zip):
        return
    os.makedirs(os.path.dirname(dst_zip), exist_ok=True)
    print(f"Downloading: {url}")
    urlretrieve(url, dst_zip)
    print("Download done.")

def _unzip_if_needed(zip_path: str, extract_to: str) -> None:
    out_dir = os.path.join(extract_to, "tiny-imagenet-200")
    if os.path.exists(out_dir):
        return
    print(f"Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print("Unzip done.")

def _restructure_val_to_imagefolder(tiny_root: str) -> None:
    """
    Convert tiny-imagenet-200/val to ImageFolder-style:
      val/
        images/ -> move each image into val/<wnid>/ and create folders by wnid, based on val_annotations.txt
    Safe to call multiple times.
    """
    val_dir = os.path.join(tiny_root, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")

    # already converted?
    if not os.path.exists(images_dir) and os.path.exists(ann_path):
        # images/ already moved; nothing to do
        return

    if not os.path.exists(ann_path):
        # Some mirrors ship already-converted structure
        return

    print("Restructuring val/ to ImageFolder format...")
    # Read annotations: filename \t wnid \t ...
    with open(ann_path, "r") as f:
        rows = [line.strip().split("\t") for line in f if line.strip()]
    # Create wnid folders
    wnids = sorted(set(r[1] for r in rows))
    for w in wnids:
        os.makedirs(os.path.join(val_dir, w), exist_ok=True)
    # Move images
    for fname, wnid, *_ in rows:
        src = os.path.join(images_dir, fname)
        dst = os.path.join(val_dir, wnid, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    # Remove images/ and annotations
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir, ignore_errors=True)
    # Keep val_annotations.txt for record, but it’s not needed anymore
    print("Val restructuring done.")

def prepare_tiny_imagenet(cfg: TrainConfig) -> str:
    dst_zip = os.path.join(cfg.data_root, "tiny-imagenet-200.zip")
    _download_if_needed(cfg.tiny_url, dst_zip)
    _unzip_if_needed(dst_zip, cfg.data_root)
    tiny_root = os.path.join(cfg.data_root, cfg.tiny_dirname)
    _restructure_val_to_imagefolder(tiny_root)
    return tiny_root

# ---------------------------
# Models (pure constructors)
# ---------------------------
class MLP(nn.Module):
    # For 64x64x3 input
    def __init__(self, in_dim: int = 64*64*3, hidden: Tuple[int,int] = (1024, 512), num_classes: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

class SmallCNN(nn.Module):
    # 3x64x64 -> (Conv+Pool)*3 -> 128x8x8 -> 8192 -> 512 -> 200
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 32x32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 64x16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2) # 128x8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def make_resnet18_scratch(num_classes: int = 200) -> nn.Module:
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_resnet18_pretrained_linear_probe(num_classes: int = 200, freeze_backbone: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # train only this by default
    return m

# ---------------------------
# ViT-Small (timm)
# ---------------------------
def make_vit_small_scratch(num_classes: int = 200) -> nn.Module:
    """
    ViT Small (patch16, 224) from timm, trained from scratch.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model

def make_vit_small_pretrained_linear_probe(num_classes: int = 200, freeze_backbone: bool = True) -> nn.Module:
    """
    ViT Small (patch16, 224) pretrained on ImageNet-1k, linear probe by default.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,  # replace head to 200 classes
    )
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only classifier head
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad = True
    return model

# ---------------------------
# Data (distinct transforms)
# ---------------------------
def get_tiny_imagenet_loaders(cfg: TrainConfig, tiny_root: str) -> Dict[str, DataLoader]:
    # Basic (64x64) pipeline: for MLP/CNN/ResNet18(scratch without 224-resize)
    tf_basic_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
    ])
    tf_basic_test = transforms.ToTensor()

    # ResNet/ViT(pretrained) pipeline: resize to 224 + ImageNet norm
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    tf_resnet_train = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_resnet_test = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dir = os.path.join(tiny_root, "train")
    val_dir   = os.path.join(tiny_root, "val")

    train_basic = datasets.ImageFolder(train_dir, transform=tf_basic_train)
    test_basic  = datasets.ImageFolder(val_dir,   transform=tf_basic_test)

    train_res   = datasets.ImageFolder(train_dir, transform=tf_resnet_train)
    test_res    = datasets.ImageFolder(val_dir,   transform=tf_resnet_test)

    assert len(train_basic.classes) == 200, f"Expected 200 classes, got {len(train_basic.classes)}"

    loaders = {
        "train_basic": DataLoader(train_basic, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
        "test_basic":  DataLoader(test_basic,  batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
        "train_res":   DataLoader(train_res,   batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
        "test_res":    DataLoader(test_res,    batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
    }
    return loaders

# ---------------------------
# Train / Eval (pure style)
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def one_epoch(model: nn.Module, loader: DataLoader, device: str, optimizer: torch.optim.Optimizer,
              model_name: str = "", epoch_idx: int = 1) -> Tuple[float, float]:
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    n_batches = len(loader)

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        if i % max(1, n_batches // 5) == 0 or i == n_batches:
            avg_loss = total_loss / total
            acc = correct / total
            pct = 100.0 * i / n_batches
            print(f"[{model_name}] Epoch {epoch_idx}: {pct:5.1f}% | "
                  f"Loss {avg_loss:.4f} | Acc {acc*100:.2f}%")

    return total_loss / max(1, total), correct / max(1, total)

def train_nepochs(model: nn.Module, epochs: int, device: str, train_loader: DataLoader, test_loader: DataLoader,
                  lr: float, wd: float, model_name: str) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    tr_loss, tr_acc, te_loss, te_acc = 0.0, 0.0, 0.0, 0.0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = one_epoch(model, train_loader, device, optimizer, model_name=model_name, epoch_idx=ep)
        te_loss, te_acc = evaluate(model, test_loader, device)

    total_params, trainable_params = count_params(model)
    return {
        "train_loss": tr_loss, "train_acc": tr_acc,
        "test_loss": te_loss,  "test_acc": te_acc,
        "params_total": total_params, "params_trainable": trainable_params
    }

def run_3epochs(model: nn.Module, cfg: TrainConfig, loaders: Dict[str, DataLoader], use_resnet_loader: bool,
                lr: float, wd: float, name: str) -> Dict[str, float]:
    device = cfg.device
    train_loader = loaders["train_res"] if use_resnet_loader else loaders["train_basic"]
    test_loader  = loaders["test_res"]  if use_resnet_loader else loaders["test_basic"]
    return train_nepochs(model, cfg.epochs, device, train_loader, test_loader, lr, wd, name)

# ---------------------------
# Orchestrator
# ---------------------------
def main() -> None:
    # seed
    set_seed(123)
    cfg = TrainConfig()

    # data
    tiny_root = prepare_tiny_imagenet(cfg)
    loaders = get_tiny_imagenet_loaders(cfg, tiny_root)

    results: Dict[str, Dict[str, float]] = {}

    # 1) MLP (scratch)
    mlp = MLP(num_classes=cfg.num_classes)
    results["MLP (scratch)"] = run_3epochs(
        mlp, cfg, loaders, use_resnet_loader=False, lr=1e-3, wd=0.0, name="MLP"
    )

    # 2) CNN (scratch)
    cnn = SmallCNN(num_classes=cfg.num_classes)
    results["CNN (scratch)"] = run_3epochs(
        cnn, cfg, loaders, use_resnet_loader=False, lr=1e-3, wd=0.0, name="SmallCNN"
    )

    # 3) ResNet18 (scratch; 64x64 입력 그대로)
    r18_s = make_resnet18_scratch(num_classes=cfg.num_classes)
    results["ResNet18 (scratch)"] = run_3epochs(
        r18_s, cfg, loaders, use_resnet_loader=False, lr=3e-4, wd=1e-4, name="ResNet18_scratch"
    )

    # 4) ResNet18 (pretrained; linear-probe: fc만 학습, 224 resize + ImageNet norm)
    r18_p = make_resnet18_pretrained_linear_probe(num_classes=cfg.num_classes, freeze_backbone=True)
    results["ResNet18 (pretrained, linear-probe)"] = run_3epochs(
        r18_p, cfg, loaders, use_resnet_loader=True, lr=1e-2, wd=0.0, name="ResNet18_pretrained_LP"
    )

    # 5) ViT-Small (scratch; 224 파이프라인 사용)
    vit_s_scratch = make_vit_small_scratch(num_classes=cfg.num_classes)
    results["ViT-Small (scratch)"] = run_3epochs(
        vit_s_scratch, cfg, loaders, use_resnet_loader=True,  # 224 + ImageNet norm
        lr=3e-4, wd=1e-4, name="ViT_Small_scratch"
    )

    # 6) ViT-Small (pretrained; linear-probe)
    vit_s_lp = make_vit_small_pretrained_linear_probe(num_classes=cfg.num_classes, freeze_backbone=True)
    results["ViT-Small (pretrained, linear-probe)"] = run_3epochs(
        vit_s_lp, cfg, loaders, use_resnet_loader=True,
        lr=1e-2, wd=0.0, name="ViT_Small_pretrained_LP"
    )

    # pretty print
    hdr = f"{'Model':38s} | {'Params(Total/Trainable)':>23s} | {'Train Acc':>9s} | {'Test Acc':>8s}"
    print(hdr)
    print("-"*len(hdr))
    for name, r in results.items():
        pt = f"{r['params_total']:,}/{r['params_trainable']:,}"
        print(f"{name:38s} | {pt:>23s} | {r['train_acc']*100:8.2f}% | {r['test_acc']*100:7.2f}%")

if __name__ == "__main__":
    main()
