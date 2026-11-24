import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def get_binary_mnist_loaders(digit1: int = 0, digit2: int = 7, batch_size: int = 222) -> tuple[DataLoader, DataLoader]:
    """
    Downloads MNIST, filters for two target digits, downscales images to 8x8,
    and returns PyTorch DataLoaders for training and testing.

    Args:
        digit1: First MNIST digit, mapped to binary label 0. Defaults to 0.
        digit2: Second MNIST digit, mapped to binary label 1. Defaults to 7.
        batch_size: Number of samples per mini-batch. Defaults to 222.

    Returns:
        train_loader: Shuffled DataLoader for the filtered training set.
        test_loader: Unshuffled DataLoader for the filtered test set.
    """
    
    transform = transforms.Compose([
        transforms.Resize((8, 8), antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    raw_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    raw_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def extract_and_filter(dataset: datasets.MNIST) -> TensorDataset:
        """Filters a raw MNIST dataset for the two target digits and downscales images to 8x8."""
        mask = (dataset.targets == digit1) | (dataset.targets == digit2)
        
        filtered_targets = dataset.targets[mask]
        binary_targets = torch.where(filtered_targets == digit1, 0, 1)

        filtered_images = dataset.data[mask].unsqueeze(1).float() / 255.0
        resized_images = transforms.functional.resize(filtered_images, [8, 8], antialias=True)
        flattened_images = resized_images.squeeze(1).reshape(-1, 64)

        return TensorDataset(flattened_images, binary_targets)

    train_dataset = extract_and_filter(raw_train)
    test_dataset = extract_and_filter(raw_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_binary_mnist_loaders(digit1=0, digit2=7, batch_size=222)
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)  # Expected: (222, 64)
    print("Batch labels shape:", labels.shape)  # Expected: (222,)
    print("Label values:", torch.unique(labels)) # Expected: tensor([0, 1])
    print("Pixel value min/max:", images.min().item(), images.max().item()) # Expected: in [0.0, 1.0]