'''
Evaluates the effects of explicit manifold regularization on diffusion "language"/sequence models.

Notes:
1. In order to make the ArithmeticDataset computationally efficient and PyTorch-friendly,
I avoid string manipulation, tokenization, and dynamic equation generation inside the __getitem__ 
method; since Python operations inside __getitem__ become a sizable bottleneck during multi-worker 
data loading.

2. 

'''

import random
import operator
from typing import Tuple, List, Dict

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Part 1: A Dataset of Synthetic Structured Language
# =============================================================================

class ArithmeticDataset(Dataset):
    """
    A highly optimized dataset for generating synthetic arithmetic expressions.
    All data is pre-generated and stored in a single memory-mapped tensor for O(1) retrieval.
    """
    def __init__(self, num_samples: int, seq_len: int = 16):
        self.num_samples = num_samples
        self.seq_len = seq_len

        # 1. Define Vocabulary and Mappings
        self.chars = [
            '[PAD]', '[MASK]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            '+', '-', '*', '/', '=', '(', ')'
        ]
        self.vocab_size = len(self.chars)
        self.char2id: Dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.id2char: Dict[int, str] = {i: c for i, c in enumerate(self.chars)}

        self.pad_id = self.char2id['[PAD]']
        self.mask_id = self.char2id['[MASK]']

        # 2. Pre-generate and store as a contiguous 2D tensor (int16 is sufficient and saves RAM)
        self.data = self._build_dataset()

    def _build_dataset(self) -> dict:
        """
        Generates unique, valid mathematical equations, tokenizes them, and pads them.
        Returns mapping of tokens to integer IDs
        """

        print(f"Generating {self.num_samples} synthetic arithmetic sequences...")
        unique_equations = set()

        # Simple template engines to ensure valid syntax and controllable lengths
        ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul
        }
        op_keys = list(ops.keys())

        while len(unique_equations) < self.num_samples:
            # Template 1: A op B = C (e.g., 45+12=57)
            # Template 2: (A op B) op C = D (e.g., (3+4)*2=14)

            template_choice = random.choice([1, 2])
            try:
                if template_choice == 1:
                    a, b = random.randint(1, 99), random.randint(1, 99)
                    op1 = random.choice(op_keys)
                    res = ops[op1](a, b)
                    eq_str = f"{a}{op1}{b}={res}"

                else:
                    a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)
                    op1, op2 = random.choice(op_keys), random.choice(op_keys)
                    res_inner = ops[op1](a, b)
                    res_final = ops[op2](res_inner, c)
                    eq_str = f"({a}{op1}{b}){op2}{c}={res_final}"

                # Filter out equations that exceed the sequence length
                if len(eq_str) <= self.seq_len:
                    unique_equations.add(eq_str)

            except ZeroDivisionError:
                continue

        # Convert strings to padded integer tensors
        tensor_data = torch.full((self.num_samples, self.seq_len), self.pad_id, dtype=torch.long)

        for i, eq in enumerate(unique_equations):
            tokenized = [self.char2id[char] for char in eq]
            tensor_data[i, :len(tokenized)] = torch.tensor(tokenized, dtype=torch.long)

        print("Dataset generation complete.")
        return tensor_data

    def decode(self, token_ids: torch.Tensor) -> str:
        """Utility to convert a tensor of IDs back to a human-readable string."""
        chars = [self.id2char[id.item()] for id in token_ids if id.item() != self.pad_id]
        return "".join(chars)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # O(1) memory view retrieval.
        return self.data[idx]

def compute_edit_distance(seq_a: torch.Tensor, seq_b: torch.Tensor) -> float:
    """
    Computes the normalized Levenshtein (edit) distance using Dynamic Programming.
    Returns a float between 0.0 and 1.0.
    """
    # 1. Convert to lists to bypass PyTorch scalar indexing overhead
    a = seq_a.tolist()
    b = seq_b.tolist()
    len_a, len_b = len(a), len(b)

    # 2. Handle base cases for normalization
    if len_a == 0:
        return 1.0 if len_b > 0 else 0.0
    if len_b == 0:
        return 1.0

    # 3. Initialize the DP table
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    # 4. Populate the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    # 5. Extract absolute distance and normalize
    absolute_distance = dp[len_a][len_b]
    max_len = max(len_a, len_b)

    return absolute_distance / max_len

def batch_compute_edit_distance(batch_a: torch.Tensor, batch_b: torch.Tensor) -> torch.Tensor:
    """
    Applies the DP edit distance function across a batch.
    """
    batch_size = batch_a.size(0)
    distances = torch.zeros(batch_size, dtype=torch.float32)

    # Iterate through the batch. Using the DP function from the previous step.
    for i in range(batch_size):
        distances[i] = compute_edit_distance(batch_a[i], batch_b[i])

    return distances

def sample_sequence_pairs(
    batch: torch.Tensor,
    vocab_size: int,
    min_mutations: int = 1,
    max_mutations: int = 4,
    pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates x_B by applying EXACTLY K random mutations to valid tokens in x_A.
    Computes and returns the normalized edit distances.
    """
    device = batch.device
    batch_size, seq_len = batch.shape
    x_a = batch

    # 1. Sample exact number of mutations per sequence in the batch
    num_mutations = torch.randint(min_mutations, max_mutations + 1, (batch_size, 1), device=device)

    # 2. Generate random scores for every token position
    random_scores = torch.rand((batch_size, seq_len), device=device)

    # 3. Protect PAD tokens from being mutated by assigning them a score of -infinity
    pad_mask = x_a == pad_id
    random_scores.masked_fill_(pad_mask, -float('inf'))

    # 4. Apply the Double-Argsort Trick to get the rank of each token's score per row
    # The highest score gets rank 0, the second highest gets rank 1, etc.
    ranks = torch.argsort(torch.argsort(random_scores, dim=-1, descending=True), dim=-1)

    # 5. Create the exact mutation mask: Select exactly `num_mutations` highest-scoring indices
    mutation_mask = ranks < num_mutations

    # 6. Generate random replacement tokens (excluding PAD=0 and MASK=1)
    random_tokens = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)

    # 7. Apply the precise mask to create x_B
    x_b = torch.where(mutation_mask, random_tokens, x_a)

    # 8. Compute the exact edit distances (move to CPU for DP, then back)
    distances = batch_compute_edit_distance(x_a.cpu(), x_b.cpu()).to(device)

    return x_a, x_b, distances


# =============================================================================
# Part 2: The Micro-MDLM Architecture
# =============================================================================

class MicroMDLM(nn.Module):
    """
    A minimal Transformer encoder for Masked Diffusion.
    """
    def __init__(self, vocab_size: int, num_timesteps: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        # Token and Position Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 16, d_model))

        # Timestep Embedding
        self.t_embedding = nn.Embedding(num_timesteps, d_model)

        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Unembedding projection to vocabulary logits
        self.unbedding = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Noised token sequences at timestep t. Shape: [B, L]
            t: Timestep conditioning. Shape: [B]
        Returns:
            logits: Predictions over the vocabulary. Shape: [B, L, V]
            latents: The continuous latent representations z_t. Shape: [B, L, D]
        """
        # 1. Embed tokens and add spatial positional encodings, shape: [B, L, D]
        emb = self.embedding(x) + self.positional_encoding

        # 2. Embed timesteps, shape: [B, D]
        t_emb = self.t_embedding(t)

        # 3. Inject timestep embeddings by broadcasting across the sequence length, shape: [B, L, D]
        emb = emb + t_emb.unsqueeze(1)

        # 4. Process through Transformer to extract continuous latents z_t, shape: [B, L, D]
        latents = self.transformer(emb)

        # 5. Project to discrete vocabulary space, shape: [B, L, V]
        logits = self.unbedding(latents)

        return logits, latents

class MaskedDiffusionProcess:
    """
    Handles the forward transitions for discrete diffusion with an absorbing [MASK] state.
    Optimized to use marginal probabilities instead of dense transition matrices.
    """
    def __init__(self, num_timesteps: int, vocab_size: int, mask_token_id: int):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

        # Define the masking schedule
        self.beta = torch.linspace(1e-4, 0.05, num_timesteps)

        # alpha_t is the probability of surviving a single timestep without being masked
        self.alpha = 1.0 - self.beta

        # alpha_bar_t is the cumulative probability of surviving from t=0 to t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward process: Adds noise to x_0 to get x_t using absorbing state logic.
        Args:
            x_0: Clean, original token sequences. Shape: [B, L]
            t: Timesteps to sample for each sequence. Shape: [B]
        Returns:
            x_t: Noised sequences where some tokens are replaced by [MASK].
        """
        device = x_0.device
        batch_size, seq_len = x_0.shape

        # 1. Fetch the survival probabilities for the given timesteps
        # Move alpha_bar to the correct device and extract the batched probabilities
        self.alpha_bar = self.alpha_bar.to(device)
        survival_probs = self.alpha_bar[t] # Shape: [B]

        # 2. Reshape for broadcasting across the sequence length
        survival_probs = survival_probs.unsqueeze(1).expand(batch_size, seq_len) # Shape: [B, L]

        # 3. Sample a random matrix to determine which tokens survive
        random_matrix = torch.rand((batch_size, seq_len), device=device)

        # 4. Create the mask: True if the token survives, False if it gets masked
        # Note: In practice, you should never mask padding tokens. Assuming pad_id=0.
        keep_mask = random_matrix < survival_probs
        pad_mask = x_0 == 0
        final_keep_mask = keep_mask | pad_mask

        # 5. Apply the mask to generate x_t
        x_t = torch.where(final_keep_mask, x_0, torch.tensor(self.mask_token_id, device=device))

        return x_t


# =============================================================================
# Part 3: The Dual-Objective Loss Function
# =============================================================================

def compute_mdlm_loss(logits: torch.Tensor, targets: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """
    Computes $L_{\\text{MDLM}}$, the standard cross-entropy for predicting unmasked tokens.
    Loss is calculated exclusively on the masked positions to match standard MDLM objectives.
    """
    # Filter logits and targets to only the masked positions using boolean indexing
    # logits shape becomes: [num_masked_tokens, vocab_size]
    # targets shape becomes: [num_masked_tokens]
    logits_masked = logits[mask]
    targets_masked = targets[mask]

    # Handle the edge case where no tokens are masked in the entire batch
    if logits_masked.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(logits_masked, targets_masked)

def compute_isometric_loss(z_t_a: torch.Tensor, z_t_b: torch.Tensor, d_edit: torch.Tensor,
                           d_max: float) -> torch.Tensor:
    """
    Computes the manifold penalty:
    $$L_{\\text{isometric}} 
     = \\left( \\frac{||z_{t,A} - z_{t,B}||_2}{D_{\\max}} - d_{\\text{edit}}(x_A, x_B) \\right)^2$$
    """
    batch_size = z_t_a.size(0)

    # Reshape from [B, L, D] to [B, L * D] to compute the L2 norm across the entire sequence space
    z_t_a_flat = z_t_a.view(batch_size, -1)
    z_t_b_flat = z_t_b.view(batch_size, -1)

    # Compute L2 norm per sequence pair: ||z_t_a - z_t_b||_2
    # Resulting shape: [B]
    diff_squared = (z_t_a_flat - z_t_b_flat) ** 2
    l2_distances = torch.sqrt(torch.sum(diff_squared, dim=1))

    # Normalize the L2 distance
    normalized_l2 = l2_distances / d_max

    # Compute the squared error between the normalized latent distance
    # and ground truth edit distance
    penalty = (normalized_l2 - d_edit) ** 2

    return penalty.mean()

def compute_total_loss(
    model: 'MicroMDLM',
    diffusion: 'MaskedDiffusionProcess',
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    d_edit: torch.Tensor,
    lambda_iso: float,
    d_max: float,
    pad_id: int = 0
) -> Tuple[torch.Tensor, dict]:
    """
    Executes the full forward pass for a pair, computes both losses, and combines them.
    """
    device = x_a.device
    batch_size = x_a.size(0)

    # 1. Sample random timesteps
    # CRITICAL: We use the same timestep t for both A and B to ensure valid manifold comparisons.
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)

    # 2. Apply forward diffusion
    x_t_a = diffusion.q_sample(x_a, t)
    x_t_b = diffusion.q_sample(x_b, t)

    # 3. Forward pass through the model
    logits_a, z_t_a = model(x_t_a, t)
    logits_b, z_t_b = model(x_t_b, t)

    # 4. Generate prediction masks
    # We only penalize predictions where the token was actually masked, AND ignore padding tokens
    mask_a = (x_t_a == diffusion.mask_token_id) & (x_a != pad_id)
    mask_b = (x_t_b == diffusion.mask_token_id) & (x_b != pad_id)

    # 5. Compute MDLM Base Loss (averaged across both sequence forward passes)
    loss_mdlm_a = compute_mdlm_loss(logits_a, x_a, mask_a)
    loss_mdlm_b = compute_mdlm_loss(logits_b, x_b, mask_b)
    loss_mdlm = (loss_mdlm_a + loss_mdlm_b) / 2.0

    # 6. Compute Manifold Penalty
    loss_iso = compute_isometric_loss(z_t_a, z_t_b, d_edit, d_max)

    # 7. Compute Total Regularized Loss
    total_loss = loss_mdlm + (lambda_iso * loss_iso)

    metrics = {
        "loss_mdlm": loss_mdlm.item(),
        "loss_iso": loss_iso.item(),
        "total_loss": total_loss.item()
    }

    return total_loss, metrics


# =============================================================================
# Part 4: Latent Interpolation & Lipschitz Continuity Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_latent_interpolation(
    model: 'MicroMDLM',
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    alphas: List[float]
) -> List[torch.Tensor]:
    """
    Encodes, interpolates, and decodes pairs of sequences.
    """
    model.eval()
    device = x_a.device
    batch_size = x_a.size(0)

    # Use t=0 (fully clean state) to extract the foundational latents
    t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)

    # 1. Encode x_A and x_B
    _, z_a = model(x_a, t_zero)
    _, z_b = model(x_b, t_zero)

    decoded_sequences = []

    # 2 & 3. Interpolate and decode for each alpha
    for alpha in alphas:
        # Linear interpolation in the continuous latent space
        z_interp = alpha * z_a + (1.0 - alpha) * z_b

        # Project continuous latents back to discrete vocabulary logits
        logits_interp = model.unbedding(z_interp)

        # Get discrete tokens via argmax
        tokens_interp = torch.argmax(logits_interp, dim=-1)
        decoded_sequences.append(tokens_interp)

    return decoded_sequences


class ValueTwistMLP(nn.Module):
    """
    A tiny MLP trained on top of frozen latents $z$ to predict if the equation 
    evaluates to an even number.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # z shape: [B, L, d_model]
        # Pool across the sequence length L to get a single continuous vector per equation
        z_pooled = z.mean(dim=1) # Shape: [B, d_model]

        return self.net(z_pooled)

def measure_lipschitz_continuity(
    model: 'MicroMDLM',
    value_model: ValueTwistMLP,
    dataloader: DataLoader
) -> float:
    """
    Measures the variance of the gradients $\\nabla_z V$.
    Lower variance indicates a smoother, more Lipschitz-continuous latent manifold.
    """
    model.eval()
    value_model.eval()
    device = next(model.parameters()).device

    all_grad_norms = []

    for batch in dataloader:
        # Depending on your DataLoader collate_fn, extract just the base sequence
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        batch_size = x.size(0)
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)

        # 1. Extract latents $z$ from the frozen MDLM
        with torch.no_grad():
            _, z = model(x, t_zero)

        # 2. Require grad on the detached $z$ manifold
        z.requires_grad_(True)

        # 3. Pass $z$ through the value_model
        v_pred = value_model(z)

        # 4. Compute gradients of the value output with respect to $z$
        # Using autograd.grad to extract the exact derivative tensor
        v_grad = torch.autograd.grad(
            outputs=v_pred,
            inputs=z,
            grad_outputs=torch.ones_like(v_pred),
            create_graph=False,
            retain_graph=False
        )[0]

        # Flatten the gradient spatial dimensions [B, L, D] -> [B, L*D]
        # to calculate the sequence-level gradient magnitude
        v_grad_flat = v_grad.view(batch_size, -1)

        # Calculate the L2 norm of the gradient for each sequence in the batch
        # Math: ||\nabla_z V||_2
        grad_norms = torch.sqrt(torch.sum(v_grad_flat ** 2, dim=1))

        all_grad_norms.append(grad_norms.detach().cpu())

    # 5. Compute the variance of the gradient norms across the dataset
    all_grad_norms = torch.cat(all_grad_norms)
    variance = torch.var(all_grad_norms).item()
    return variance

if __name__ == "__main__":
    def main():
        # 1. Configuration & Hyperparameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Executing on device: {device}")

        num_samples = 10000
        batch_size = 128
        seq_len = 16
        mdlm_epochs = 16
        mlp_epochs = 3
        learning_rate = 5e-4
        d_max = 10.0  # Assumed max L2 distance in the latent space

        # 2. Shared Initialization
        dataset = ArithmeticDataset(num_samples=num_samples, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        vocab_size = dataset.vocab_size
        pad_id = dataset.pad_id
        mask_id = dataset.mask_id
        num_timesteps = 50

        # Helper to extract the ground truth parity (is_even) from the dataset strings
        def get_parity_labels(x_batch):
            labels = []
            for seq in x_batch:
                eq_str = dataset.decode(seq)
                try:
                    # Extract the result after the '=' sign
                    ans = int(eq_str.split('=')[1])
                    labels.append(1.0 if ans % 2 == 0 else 0.0)
                except:
                    labels.append(0.0) # Fallback for any malformed sequences
            return torch.tensor(labels, dtype=torch.float32, device=device)

        # Extract a fixed test pair so both models interpolate the exact same equations
        test_batch = next(iter(dataloader)).to(device)
        fixed_x_a = test_batch[0:1]
        fixed_x_b = test_batch[1:2]

        # Define the two experiment configurations
        experiments = [
            {"name": "Baseline", "lambda_iso": 0.0, "prefix": "baseline"},
            {"name": "Regularized", "lambda_iso": 1, "prefix": "regularized"}
        ]

        results = {}
        os.makedirs("checkpoints", exist_ok=True)

        # 3. Main Experiment Loop
        for exp in experiments:
            exp_name = exp["name"]
            l_iso = exp["lambda_iso"]
            prefix = exp["prefix"]

            print(f"\n{'='*60}")
            print(f" EXPERIMENT: {exp_name} (lambda_iso = {l_iso})")
            print(f"{'='*60}")

            # Initialize fresh models for this run
            model = MicroMDLM(vocab_size=vocab_size, num_timesteps=num_timesteps).to(device)
            diffusion = MaskedDiffusionProcess(num_timesteps=num_timesteps, 
                                               vocab_size=vocab_size, mask_token_id=mask_id)
            value_model = ValueTwistMLP(d_model=128).to(device)

            optimizer_mdlm = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            optimizer_mlp = torch.optim.AdamW(value_model.parameters(), lr=learning_rate)

            # --- Phase 1: Train MicroMDLM ---
            print(f"\n--- Phase 1: Training {exp_name} MicroMDLM ---")
            model.train()

            for epoch in range(mdlm_epochs):
                pbar = tqdm(dataloader, desc=f"MDLM Epoch {epoch+1}/{mdlm_epochs}")

                for batch in pbar:
                    batch = batch.to(device)
                    optimizer_mdlm.zero_grad()

                    x_a, x_b, d_edit = sample_sequence_pairs(
                        batch, vocab_size, min_mutations=1, max_mutations=4, pad_id=pad_id
                    )

                    loss, metrics = compute_total_loss(
                        model, diffusion, x_a, x_b, d_edit, l_iso, d_max, pad_id
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_mdlm.step()

                    pbar.set_postfix({
                        "Loss": f"{metrics['total_loss']:.3f}", 
                        "MDLM": f"{metrics['loss_mdlm']:.3f}", 
                        "Iso": f"{metrics['loss_iso']:.3f}"
                    })

            # Freeze MDLM weights for Phase 2
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            # --- Phase 2: Train ValueTwistMLP on frozen latents ---
            print(f"\n--- Phase 2: Training {exp_name} ValueTwistMLP Probe ---")
            value_model.train()
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)

            for epoch in range(mlp_epochs):
                pbar = tqdm(dataloader, desc=f"MLP Epoch {epoch+1}/{mlp_epochs}")

                for batch in pbar:
                    batch = batch.to(device)
                    labels = get_parity_labels(batch)

                    optimizer_mlp.zero_grad()

                    with torch.no_grad():
                        _, z = model(batch, t_zero)

                    preds = value_model(z).squeeze(-1)

                    loss = F.binary_cross_entropy(preds, labels)
                    loss.backward()
                    optimizer_mlp.step()

                    pbar.set_postfix({"BCE Loss": f"{loss.item():.4f}"})

            # --- Phase 3: Evaluation & Saving ---
            print(f"\n--- Phase 3: Evaluating {exp_name} ---")

            variance = measure_lipschitz_continuity(model, value_model, dataloader)
            print(f"Gradient Variance: {variance:.6f}")

            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            interpolated_seqs = evaluate_latent_interpolation(model, fixed_x_a, fixed_x_b, alphas)

            # Store results for final comparison
            results[exp_name] = {
                "variance": variance,
                "interpolations": [dataset.decode(seq[0]) for seq in interpolated_seqs]
            }

            # Save Models
            mdlm_path = os.path.join("checkpoints", f"mdlm_{prefix}.pt")
            mlp_path = os.path.join("checkpoints", f"value_mlp_{prefix}.pt")
            torch.save(model.state_dict(), mdlm_path)
            torch.save(value_model.state_dict(), mlp_path)
            print(f"Saved {exp_name} models to checkpoints/")

        # 4. Final Comparative Output
        print("\n\n" + "="*60)
        print(" FINAL COMPARISON REPORT")
        print("="*60)

        print("\n1. Lipschitz Continuity (Gradient Variance)")
        print("-" * 40)
        print(f"Baseline (λ=0.0):    {results['Baseline']['variance']:.6f}")
        print(f"Regularized (λ=0.5): {results['Regularized']['variance']:.6f}")
        print("* Note: Lower variance indicates a smoother, more continuous latent manifold.\n")

        print("2. Latent Interpolation Comparison")
        print("-" * 40)
        print(f"Sequence A (\\alpha=0.0): {dataset.decode(fixed_x_a[0])}")
        print(f"Sequence B (\\alpha=1.0): {dataset.decode(fixed_x_b[0])}\n")

        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        print(f"{'Alpha':<10} | {'Baseline Path':<20} | {'Regularized Path':<20}")
        print("-" * 55)
        for i, alpha in enumerate(alphas):
            base_str = results['Baseline']['interpolations'][i]
            reg_str = results['Regularized']['interpolations'][i]
            print(f"{alpha:<10.2f} | {base_str:<20} | {reg_str:<20}")
        print("\nExecution Complete.")

    # Execute the script
    main()