# Quantum Machine Learning with Tree Tensor Networks

A PyTorch replication of the numerical experiments in **"Towards Quantum Machine Learning with Tensor Networks"** by Huggins et al. (2019).

This repository implements a discriminative **Tree Tensor Network (TTN)** quantum classifier trained on binary MNIST using gradient-free **SPSA** optimization, followed by the **Section 4.3 noise-resilience evaluation** under simulated amplitude damping and dephasing channels.

---

## Repository Structure

```text
quantum_ml_tn/
├── main.py                  # Training loop, evaluation, and Section 4.3 noise experiment
├── README.md                # Project overview, setup, and usage
├── environment.yml          # Conda environment specification
└── src/
    ├── README.md            # Technical documentation of all modules
    ├── data_loader.py       # Binary MNIST loading, 8x8 downscaling, and mini-batching
    ├── feature_map.py       # Single-qubit trigonometric feature encoding
    ├── loss.py              # Born's rule readout and margin-based loss function
    ├── noise.py             # CPTP amplitude damping and dephasing Kraus channels
    ├── models/
    │   ├── unitaries.py     # Parameterized 2-qubit unitary blocks U = exp(iH)
    │   └── ttn.py           # 64-to-1 binary Tree Tensor Network with partial traces
    └── optim/
        └── spsa.py          # Finite-difference SPSA optimizer with momentum
```

---

## Approach Overview

- **Architecture:** A 6-layer binary Tree Tensor Network taking $64$ classical pixel inputs (from $8 \times 8$ downscaled MNIST images) encoded into single-qubit states and contracting them through $63$ parameterized 2-qubit unitary gates ($63 \times 16 = 1008$ real trainable parameters) down to a single output qubit density matrix $\rho \in \mathbb{C}^{2 \times 2}$.
- **Optimization:** Trained without backpropagation (`torch.no_grad()`) using Simultaneous Perturbation Stochastic Approximation (**SPSA**) with momentum, matching the hyperparameters and finite-difference schedule from Section 3.2 of the paper.
- **Noise Simulation:** Evaluates the trained classifier under Completely Positive Trace-Preserving (**CPTP**) Kraus channels for pure dephasing ($p_d = 0.028$) and amplitude damping ($p_a = 0.039$), applied to both incoming qubits before each 2-qubit unitary gate (Section 4.3).
- **Double Precision (`float64` / `complex128`):** The paper's SPSA schedule produces a very small perturbation step size ($\alpha_0 \approx 5 \times 10^{-7}$). Combined with the margin loss exponent ($\eta = 5.59$), 32-bit floats suffer from floating-point truncation during finite-difference evaluation (`loss_plus - loss_minus == 0`). All model parameters, inputs, and density matrices therefore use 64-bit precision (`float64` / `complex128`).

---

## Setup

### Use `environment.yml`

```bash
conda env create -f environment.yml
conda activate quantum_ml
```

> **Note (macOS / Conda PATH):** If your shell does not automatically resolve the active Conda environment's binaries when running `python`, ensure the Conda `bin` directory is prioritized:
> ```bash
> export PATH="$CONDA_PREFIX/bin:$PATH"
> ```

---

## Execution

Both the **noise-free** and **noise-inserted** experiments are executed sequentially from a single entry point at the repository root:

```bash
python main.py
```

### 1. Noise-Free Experiment (Training & Clean Baseline)
When `python main.py` starts, it automatically runs the noise-free discriminative experiment:
1. Downloads the MNIST dataset into `./data/` (if not already present), filters for digits `0` and `7`, and downscales images to $8 \times 8$.
2. Initializes the 1,008-parameter `TreeTensorNetwork` in double precision (`complex128`).
3. Trains the model for **30 epochs** with mini-batch size **222** (randomly reshuffled each epoch) using the SPSA optimizer without noise (`kraus_ops=None`).
4. Logs the per-epoch training loss, clean test loss, and clean test accuracy, followed by the final `Baseline Accuracy (No Noise)`.

### 2. Noise-Inserted Experiment (Section 4.3 Noise Resilience)
Immediately after epoch 30 finishes, `main.py` automatically invokes `run_noise_experiment(model, test_loader)` using the freshly trained model weights:
1. Computes the coherence times $T_1$ and $T_2$ corresponding to the paper's target error probabilities ($p_a = 0.039$, $p_d = 0.028$) at gate time $T_g = 0.2\;\mu\text{s}$.
2. Constructs the composed CPTP Kraus channels `[dephasing_kraus, damping_kraus]` and passes them to `evaluate(model, test_loader, kraus_ops=kraus_channels)`.
3. Prints the final `Noisy Accuracy (p_a=0.039, p_d=0.028)` directly below the noise-free baseline for direct comparison.

**Note on Training vs. Evaluation under Noise:** In Section 4.3 (Figure 13), Huggins et al. (2019) note that training the TTN directly under noise required a different set of hyperparameters, which are not published in the paper. Accordingly, this replication trains the model noise-free using the published Section 3.3 hyperparameters and evaluates the resulting parameters under the Section 4.3 amplitude damping and dephasing channels to measure inference-time noise resilience.

*(To run only the noise-free experiment, comment out `run_noise_experiment(model, test_loader)` at the end of `train()` in `main.py`. To test custom noise levels, adjust `0.039` ($p_a$), `0.028` ($p_d$), or `T_g` inside `run_noise_experiment` in `main.py`.)*

---

## Replicated Results (MNIST `0` vs. `7`)

| Evaluation Setting | Test Accuracy | Paper Reference (Huggins et al., 2019) |
| :--- | :---: | :---: |
| **Noise-Free (Clean Baseline)** | 98.0% – 99.5% | 97.6% (Fig. 8) |
| **Noise-Inserted ($p_a = 0.039,\; p_d = 0.028$)** | 0.1% – 0.5% drop | 97.1% (Fig. 13) |

*(Exact values vary slightly across runs due to random parameter initialization and stochastic SPSA perturbations.)*

---

## Note on AI Usage

AI (Gemini) was used for repository structuring, for code review and for documentation. The implementation itself was done without AI.