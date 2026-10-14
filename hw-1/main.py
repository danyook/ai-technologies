import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

transform = transforms.ToTensor()
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# CrossEntropyLoss сам применяет log_softmax к логитам, поэтому softmax в модели не нужен
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def run_epoch(loader, train):
    model.train(train)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    test_loss, test_acc = run_epoch(test_loader, train=False)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    print(f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc:.4f} | "
          f"test loss {test_loss:.4f}, acc {test_acc:.4f}")

model.eval()
x, y = next(iter(test_loader))
with torch.no_grad():
    probs = torch.softmax(model(x[:5]), dim=1)
for p, label in zip(probs, y[:5]):
    print(f"true: {label.item()}, predicted: {p.argmax().item()}, "
          f"probs: {[round(v, 3) for v in p.tolist()]}")

epochs = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, history["train_loss"], label="train")
ax1.plot(epochs, history["test_loss"], label="test")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()
ax2.plot(epochs, history["train_acc"], label="train")
ax2.plot(epochs, history["test_acc"], label="test")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.savefig("plots.png")
plt.show()
