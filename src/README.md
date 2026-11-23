# Source Modules (`src/`)

This directory contains the core components of the Tree Tensor Network simulation. Detailed mathematical descriptions of each module are summarized below.

---

## Core Pipeline (`src/`)

### `data_loader.py`
- **`get_binary_mnist_loaders(digit1=0, digit2=7, batch_size=222)`**
  Downloads MNIST via `torchvision`, filters for the target digit pair (`0` mapped to label `0`, `7` mapped to label `1`), normalizes pixel intensities to $[0, 1]$, resizes images from $28 \times 28$ to $8 \times 8$ with antialiasing, and flattens each sample to a 64-element vector. The training `DataLoader` uses `shuffle=True` to randomly re-partition mini-batches at the start of every epoch.

### `feature_map.py`
- **`feature_map(x)`**
  Maps each normalized classical pixel $x_j \in [0, 1]$ to a single-qubit pure state:
  $$|\phi(x_j)\rangle = \cos\left(\frac{\pi}{2} x_j\right)|0\rangle + \sin\left(\frac{\pi}{2} x_j\right)|1\rangle$$
  Returns a tensor of shape `(batch_size, 64, 2)`.

### `loss.py`
- **`compute_probabilities(rho)`**
  Applies Born's rule to the final single-qubit density matrix $\rho \in \mathbb{C}^{2 \times 2}$ by extracting the real diagonal entries $p_0 = \text{Re}(\rho_{00})$ and $p_1 = \text{Re}(\rho_{11})$.
- **`ttn_loss(probs, y_true, margin=0.234, eta=5.59)`**
  Computes the batch-averaged margin loss from Huggins et al. (2019):
  $$L = \frac{1}{n} \sum_{i=1}^{n} \max\left(p_{\text{false}}^{(i)} - p_{\text{true}}^{(i)} + \lambda,\; 0\right)^{\eta}$$
  with $\lambda = 0.234$ and $\eta = 5.59$.

### `noise.py`
- **`get_amplitude_damping_kraus(Tg, T1, device)`**
  Constructs the two Kraus operators $[M_0, M_1]$ for amplitude damping with probability $p_a = 1 - e^{-T_g / T_1}$.
- **`get_dephasing_kraus(Tg, T2, device)`**
  Constructs the three Kraus operators $[M_0, M_1, M_2]$ for pure dephasing with probability $p_d = 1 - e^{-T_g / T_2}$.
- **`apply_cptp_channel(rho, kraus_ops)`**
  Applies a trace-preserving quantum channel across a batch of density matrices via Einstein summation:
  $$\mathcal{E}(\rho) = \sum_k M_k \rho M_k^\dagger$$

---

## Models (`src/models/`)

### `unitaries.py`
- **`UnitaryBlock`**
  Parameterizes a general 2-qubit unitary gate $U = \exp(iH) \in U(4)$ using $16$ unconstrained real parameters that construct a $4 \times 4$ Hermitian matrix $H$ (4 real diagonal entries, 6 real upper-triangular parts, 6 imaginary upper-triangular parts, with the lower triangle filled by complex conjugation). Uses `torch.linalg.matrix_exp` in `complex128`.

### `ttn.py`
- **`TreeTensorNetwork`**
  Converts the 64 encoded qubit states into initial density matrices $\rho_j = |\phi_j\rangle\langle\phi_j|$ (`complex128`) and contracts them bottom-up across 6 binary-tree layers ($32 + 16 + 8 + 4 + 2 + 1 = 63$ `UnitaryBlock` gates) down to a single output density matrix.
- **`apply_node(rho_left, rho_right, U, kraus_ops=None)`**
  1. *(Optional)* Applies each CPTP noise channel in `kraus_ops` to both `rho_left` and `rho_right`.
  2. Forms the 2-qubit product density matrix $\rho_{\text{in}} = \rho_{\text{left}} \otimes \rho_{\text{right}} \in \mathbb{C}^{4 \times 4}$.
  3. Evolves the state under the unitary gate: $\rho_{\text{out}} = U \rho_{\text{in}} U^\dagger$.
  4. Traces out the right qubit ($\text{Tr}_R(\rho_{\text{out}})$) to return the reduced $2 \times 2$ density matrix passed to the next layer.

---

## Optimizer (`src/optim/`)

### `spsa.py`
- **`get_flat_params(model)` / `set_flat_params(model, flat_params)`**
  Flattens all 63 unitary parameter blocks into a single 1D vector $\Lambda \in \mathbb{R}^{1008}$ and writes updated vectors back into the model.
- **`SPSAOptimizer`**
  Implements the gradient-free SPSA algorithm with momentum (Section 3.2):
  - Step schedules: $\alpha_k = \frac{a}{(k + 1 + A)^s}$ and $\beta_k = \frac{b}{(k + 1)^t}$ with paper hyperparameters $a = 28.0$, $b = 33.0$, $A = 74.1$, $s = 4.13$, $t = 0.658$, $\gamma = 0.882$.
  - Samples a random perturbation vector $\Delta \in \{-1, +1\}^{1008}$.
  - Evaluates the two-point finite difference $g = \frac{L(\Lambda + \alpha_k \Delta) - L(\Lambda - \alpha_k \Delta)}{2 \alpha_k}$ inside `torch.no_grad()`.
  - Updates velocity $v \leftarrow \gamma v - g \beta_k \Delta$ and parameters $\Lambda \leftarrow \Lambda + v$.