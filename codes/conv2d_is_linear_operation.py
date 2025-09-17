import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 1. Custom Conv2d (using unfold + einsum)
# -----------------------------
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def conv2d_scratch(x, weight, bias=None, stride=1, padding=0):
    """
    A scratch implementation of Conv2D using unfold + einsum.
    - No nn.Conv2d, only tensor linear ops.
    - Keeps autograd (so training works).
    
    Args:
        x: (N, C_in, H, W) input tensor
           N = batch size (number of images in one batch)
           C_in = number of input channels (e.g. 1 for grayscale, 3 for RGB)
           H, W = input image height and width
        weight: (C_out, C_in, K_h, K_w) filter weights
        bias: (C_out,) bias terms
        stride: int
        padding: int
    
    Returns:
        out: (N, C_out, H_out, W_out) tensor
    """

    N, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channel mismatch"

    # Step 1: Extract local patches from the input
    # unfold -> (N, C_in*K_h*K_w, L)
    # shape after unfold: (N, C_in*K_h*K_w, L)
    # , where:
    #    N                  : batch size
    #    C_in*K_h*K_w       : number of elements in one patch (flattened)
    #    L = H_out*W_out    : number of patches (i.e., output spatial locations)
    # 👉 Each patch is represented as a 1D vector (has C_in*K_h*K_w elements) here
    """
    -------------------------------------------------------------
     Example of unfold:

        Input (1×1×3×3):
            [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]]

        With kernel size (2×2), stride=1, padding=0:

        unfold → patches (each 2×2 block):
            [[0, 1, 3, 4],   # top-left patch
            [1, 2, 4, 5],   # top-right patch
            [3, 4, 6, 7],   # bottom-left patch
            [4, 5, 7, 8]]   # bottom-right patch

            Shape: (N=1, C_in*K_h*K_w=4, L=4)

        Each patch then dot-products with the kernel weights → one output value.
    -------------------------------------------------------------
    """
    patches = F.unfold(x, (K_h, K_w), stride=stride, padding=padding)

    # Compute output spatial size
    H_out = (H + 2*padding - K_h) // stride + 1
    W_out = (W + 2*padding - K_w) // stride + 1

    # Step 2: Convolution as einsum (wx+b for each patch)
    # Step 2: Convolution as einsum (wx+b for each patch)
    """
    einsum("npl, op -> nol")

    Index meanings:
      n : batch index (0 ≤ n < N)
      p : flattened patch element index 
          (p = c*K_h*K_w + i*K_w + j, i.e. channel+kernel flattened)
      l : patch index (spatial location, L = H_out*W_out)
      o : output channel

    For each (n, l, o):
        out[n, o, l] = sum_p patches[n, p, l] * weight[o, p]

    LaTeX equivalent:
        \[
        \text{out}[n,o,l] \;=\; \sum_{p=1}^{P} \text{patches}[n,p,l] \cdot \text{weight}[o,p]
        \]

    👉 This is exactly a dot product between
       (flattened patch vector) and (kernel weight vector).
       In other words, einsum here = wx in MLP notation.

    And when we add bias later:
        y = w^T x + b

    So conv2d is nothing more than a linear operation (wx+b),
    just applied repeatedly to local patches,
    with the same weights shared across all spatial positions.
    """
    out = torch.einsum("npl, op -> nol", patches, weight.view(C_out, -1))

    # Step 3: Add bias
    if bias is not None:
        out = out + bias[None, :, None]

    # Step 4: Reshape back to feature maps (same elements, different view)
    # (N, C_out, L) == (N, C_out, H_out, W_out) in # of elements
    out = out.view(N, C_out, H_out, W_out)
    return out


class ScratchConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return conv2d_scratch(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


# -----------------------------
# 2. CNN with custom Conv2d
# -----------------------------
class ScratchSmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # --- Original ---
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # --- Scratch version ---
        self.conv1 = ScratchConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = ScratchConv2d(32, 64, kernel_size=3, padding=1)

        # Classifier (unchanged)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 28->14

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 14->7

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# 3. Training setup
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScratchSmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# -----------------------------
# 4. Training loop
# -----------------------------
train_losses, test_accs = [], []

for epoch in range(1, 4):  # 3 epochs
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss/len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(acc)
    print(f"Epoch {epoch} done! Train Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%\n")


# -----------------------------
# 5. Visualization
# -----------------------------
# Loss curve
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(test_accs, marker="s", color="orange")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()

# Show sample predictions
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():
    output = model(example_data)

plt.figure(figsize=(12,6))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(example_data[i][0].cpu(), cmap="gray")
    pred = output.argmax(dim=1)[i].item()
    plt.title(f"Pred: {pred}, True: {example_targets[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()


""" 
The result example:

Epoch 1 [0/60000] Loss: 2.2997
Epoch 1 [51200/60000] Loss: 0.0895
Epoch 1 done! Train Loss: 0.2943, Test Accuracy: 97.66%

Epoch 2 [0/60000] Loss: 0.0665
Epoch 2 [51200/60000] Loss: 0.0432
Epoch 2 done! Train Loss: 0.0711, Test Accuracy: 98.13%

Epoch 3 [0/60000] Loss: 0.0554
Epoch 3 [51200/60000] Loss: 0.0600
Epoch 3 done! Train Loss: 0.0498, Test Accuracy: 98.63%
"""
