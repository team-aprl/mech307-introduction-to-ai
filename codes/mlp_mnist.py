# MNIST 2-layer MLP with periodic progress prints (no dropout, no functionization)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device} ...")
torch.manual_seed(42)
BATCH_SIZE = 512
EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 256
PRINT_EVERY = 100  # print training status every N iterations

# ---------------- Data -----------------
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST statistics
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# --------------- Model -----------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, HIDDEN),
    nn.ReLU(inplace=True),
    nn.Linear(HIDDEN, 10),
).to(device)

# He initialization for the first Linear (ReLU-friendly)
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

lossfn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ------------- Training ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, seen = 0.0, 0, 0
    total_iters = len(train_loader)

    for it, (x, y) in enumerate(train_loader, start=1):
        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lossfn(logits, y)
        loss.backward()
        opt.step()

        # accumulate running metrics
        with torch.no_grad():
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        # periodic progress print
        if it % PRINT_EVERY == 0 or it == total_iters:
            avg_loss = running_loss / seen
            avg_acc  = 100.0 * running_correct / seen
            print(f"[Epoch {epoch}/{EPOCHS} | Iter {it}/{total_iters}] "
                  f"loss={avg_loss:.4f}  acc={avg_acc:.2f}%")

# --------------- Eval ------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"\nTest Accuracy: {100.0 * correct / total:.2f}%")

# --------------- Eval with storage ------------------
model.eval()
correct, total = 0, 0
all_imgs, all_preds, all_targets = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_imgs.append(x.detach().cpu())
        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())

print(f"\nTest Accuracy: {100.0 * correct / total:.2f}%")

# ------------- Visualization (6x5 = 30 samples with probabilities) -------------
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

all_imgs   = torch.cat(all_imgs, dim=0)        # [N, 1, 28, 28]
all_preds  = torch.cat(all_preds, dim=0)       # [N]
all_targets= torch.cat(all_targets, dim=0)     # [N]

# Recompute logits to also extract softmax probabilities
all_probs = []
model.eval()
with torch.no_grad():
    for i in range(0, len(all_imgs), 256):
        batch = all_imgs[i:i+256].to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
all_probs = torch.cat(all_probs, dim=0)  # [N,10]

wrong_idx = (all_preds != all_targets).nonzero(as_tuple=True)[0].tolist()
right_idx = (all_preds == all_targets).nonzero(as_tuple=True)[0].tolist()

SHOW_N = 30
np.random.seed(0)
n_wrong = min(10, len(wrong_idx))
chosen_wrong = np.random.choice(wrong_idx, n_wrong, replace=False).tolist() if n_wrong > 0 else []
need_more = SHOW_N - len(chosen_wrong)
chosen_right = np.random.choice(right_idx, need_more, replace=False).tolist() if need_more > 0 else []
chosen = chosen_wrong + chosen_right

rows, cols = 6, 5
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
fig.suptitle("MNIST Test Samples (red = wrong)", fontsize=14)

for i, idx in enumerate(chosen):
    r, c = divmod(i, cols)
    ax = axes[r, c]
    img = all_imgs[idx, 0].numpy()
    p = int(all_preds[idx].item())
    t = int(all_targets[idx].item())
    prob = all_probs[idx, p].item()
    ax.imshow(img, cmap="gray")
    is_wrong = (p != t)
    title = f"pred={p} ({prob*100:.1f}%) / true={t}"
    ax.set_title(title, color=("red" if is_wrong else "black"), fontsize=9)
    ax.axis("off")

for j in range(len(chosen), rows*cols):
    r, c = divmod(j, cols)
    axes[r, c].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
