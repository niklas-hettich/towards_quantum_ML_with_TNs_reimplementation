import torch
import math
from tqdm import tqdm
from src.data_loader import get_binary_mnist_loaders
from src.models.ttn import TreeTensorNetwork
from src.loss import compute_probabilities, ttn_loss
from src.optim.spsa import SPSAOptimizer
from src.noise import get_amplitude_damping_kraus, get_dephasing_kraus

def forward_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs a forward pass through the TTN and computes the margin loss.

    Args:
        model: The TreeTensorNetwork model.
        x: Input batch of flattened images of shape (batch_size, 64).
        y: True binary class labels of shape (batch_size,).

    Returns:
        loss: Scalar tensor representing the mean margin loss over the batch.
    """

    rho = model(x)
    probs = compute_probabilities(rho)
    return ttn_loss(probs, y)


def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, kraus_ops: list[list[torch.Tensor]] | None = None) -> tuple[float, float]:
    """
    Evaluates the model on a dataset, optionally under simulated quantum noise.

    Args:
        model: The TreeTensorNetwork model to evaluate.
        data_loader: DataLoader providing batches of (images, labels).
        kraus_ops: Optional list of CPTP noise channels (each a list of Kraus operators)
            applied during tree contraction. Defaults to None (noise-free).

    Returns:
        avg_loss: Average loss over all samples.
        avg_accuracy: Fraction of correctly classified samples.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            rho = model(x_batch.double(), kraus_ops=kraus_ops)

            probs = compute_probabilities(rho)

            loss = ttn_loss(probs, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            preds = torch.argmax(probs, dim=-1)
            correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = correct / total_samples
    return avg_loss, avg_accuracy


def run_noise_experiment(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> None:
    """
    Evaluates the trained model under the noise parameters defined in Section 4.3
    of Huggins et al. (2019) (p_a = 0.039, p_d = 0.028 at T_g = 200 ns).

    Args:
        model: The trained TreeTensorNetwork model.
        test_loader: DataLoader for the binary MNIST test set.
    """
    device = next(model.parameters()).device
    
    T_g = 0.2  # microseconds
    T_1 = -T_g / math.log(1 - 0.039)
    T_2 = -T_g / math.log(1 - 0.028)
    
    _, acc_clean = evaluate(model, test_loader)
    print(f"Baseline Accuracy (No Noise): {acc_clean * 100:.2f}%")
    
    damping_kraus = get_amplitude_damping_kraus(Tg=T_g, T1=T_1, device=device)
    dephasing_kraus = get_dephasing_kraus(Tg=T_g, T2=T_2, device=device)
    
    kraus_channels = [dephasing_kraus, damping_kraus]
    
    _, acc_noisy = evaluate(model, test_loader, kraus_ops=kraus_channels)
    print(f"Noisy Accuracy (p_a=0.039, p_d=0.028): {acc_noisy * 100:.2f}%")


def train() -> None:
    """
    Runs the full training pipeline for the binary MNIST (e.g., for 0 vs. 7) TTN classifier
    using SPSA optimization, followed by the Section 4.3 noise-resilience evaluation.
    """
    num_epochs = 30
    batch_size = 222

    test_digit_1 = 0
    test_digit_2 = 7

    print(f"Loading data (digits {test_digit_1} and {test_digit_2})")
    train_loader, test_loader = get_binary_mnist_loaders(digit1=test_digit_1, digit2=test_digit_2, batch_size=batch_size)

    print("Initializing TTN")
    model = TreeTensorNetwork().double()
    spsa = SPSAOptimizer(model=model, a=28.0, b=33.0, A=74.1, s=4.13, t=0.658, gamma=0.882)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x_batch, y_batch in progress_bar:
            batch_loss = spsa.step(forward_loss, x_batch.double(), y_batch, epoch=epoch)
            running_loss += batch_loss * x_batch.size(0)
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%\n")

    run_noise_experiment(model, test_loader)


if __name__ == "__main__":
    train()