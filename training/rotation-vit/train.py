import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.datasets.mnist import MNIST

from torchvision.transforms import ToTensor
from tqdm import tqdm, trange

from vit import ViTForImageRotation, MyVitBlock, get_positional_embeddings

rad2deg = lambda x: x * 180 / np.pi

np.random.seed(42)
torch.manual_seed(42)


def main():
    transform = ToTensor()

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # define model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    device = torch.device("cuda")

    model = ViTForImageRotation(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    )
    model = model.to(device)

    N_EPOCHS = 10
    LR = 5e-3

    print("Training...")
    print(f"LR = {LR}")

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # batch = [t.to(device) for t in batch]

    # training loop
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # testign loop
    print("Testing...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(test_loader, desc="Testing"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().detach().cpu().item()

        print(f"Test loss: {test_loss:.3f}")
        print(f"Test accuracy: {100 * correct / total:.2f}%")


def test_vit():
    # model = MyVit(chw = (1,28,28), n_patches=7)
    device = 'cpu'
    model = ViTForImageRotation(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)

    print(model)

    x = torch.randn(7, 1, 28, 28)  # dummy input
    print(model(x).shape)  # torch.Size([7, 49, 16])


def test_pos_emb():

    # If your get the following error:
    # Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure
    # You need to install tkinter and pyqt5
    # sudo apt-get install python3-tk
    # pip install pyqt5

    import matplotlib.pyplot as plt

    plt.imshow(get_positional_embeddings(100, 300), cmap="hot", interpolation="nearest")
    plt.show()


def test_block():
    # Encoder block outputs a tensor of the same dimensionality as the input tensor
    model = MyVitBlock(hidden_d=8, n_heads=2)
    x = torch.randn(7, 50, 8)  # dummy input
    print(model(x).shape)  # torch.Size([7, 50, 8])


if __name__ == '__main__':

    # test_vit()
    # test_pos_emb()
    # test_block()

    main()
