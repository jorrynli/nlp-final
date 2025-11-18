# Setup
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor|None=None):
    """Compute scaled dot-product attention.

    Implements the equation from "Attention is All You Need":

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    Args:
        Q: Query tensor of shape (B, h, T_q, d_k).
        K: Key tensor of shape (B, h, T_k, d_k).
        V: Value tensor of shape (B, h, T_k, d_v).
        mask: Optional additive mask broadcastable to (B, h, T_q, T_k) with 0 or -inf.

    Returns:
        out: Attention output, shape (B, h, T_q, d_v).
        attn: Attention weights (softmax probabilities), shape (B, h, T_q, T_k).
    """
    # d_k is the dimensionality of queries/keys per head
    d_k = Q.size(-1)  # read last dimension of Q for scaling

    # Compute raw attention scores by matrix-multiplying Q and K^T
    # Q @ K^T yields shape (B, h, T_q, T_k)
    #TODO
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)# scale by sqrt(d_k)

    # If a mask was provided, add it to the scores. Mask entries are 0 (keep) or -inf (block)
    if mask is not None:
        # Ensure mask is same dtype and on same device as scores to avoid runtime errors
        mask = mask.to(dtype=scores.dtype, device=scores.device)
        scores = scores + mask  # additive masking prior to softmax

    # Convert scores to probabilities along the key dimension with softmax
    # use torch functional library you important above, which is a PyTorch
    # module containing functional (stateless) implementations of layers
    # and operations like softmax, relu, cross_entropy, etc.
    #TODO
    attn = F.softmax(scores, dim=-1)  # softmax over T_k

    # Use attention weights to produce weighted sum over values
    # This line of code will perform a batched matrix multiplication over the last two dimensions
    out = torch.matmul(attn, V) # (B, h, T_q, d_v)

    # Return both the attended outputs and the attention weights for inspection
    return out, attn

def causal_mask(T_q: int, T_k: int, device, dtype: torch.dtype=torch.float32):
    """Create an additive causal mask to prevent attention to future positions.

    The mask returned can be added directly to attention logits before softmax.

    Args:
        T_q: Number of query positions.
        T_k: Number of key/value positions.
        device: Torch device to create the mask on.
        dtype: Desired floating dtype for the returned mask (default: torch.float32).

    Returns:
        mask: Tensor of shape (1, 1, T_q, T_k) with 0.0 where allowed and -inf where masked.
    """
    # Allocate a mask filled with -inf (all positions masked initially) with requested dtype
    mask = torch.full((1,1,T_q,T_k), float('-inf'), device=device, dtype=dtype)

    # Build a lower-triangular matrix of ones (allowed positions are 1)
    tril = torch.tril(torch.ones(T_q, T_k, device=device, dtype=dtype))

    # Wherever tril == 1, set the mask value to 0.0 (meaning "allowed")
    mask = mask.masked_fill(tril == 1, 0.0)

    # Return mask shaped (1,1,T_q,T_k) which will broadcast over batch and heads
    return mask

class TinyMultiHeadAttention(nn.Module):
    """A minimal multi-head self-attention implementation.

    This class implements the core mechanics of multi-head attention without
    dropout or biases. It projects inputs to Q/K/V, splits into heads, applies
    scaled dot-product attention per head, and concatenates the results.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # Ensure d_model is divisible by number of heads for equal head size
        assert d_model % num_heads == 0
        self.d_model = d_model  # full model dimensionality
        self.num_heads = num_heads  # number of parallel attention heads
        self.d_k = d_model // num_heads  # dimensionality per head

        # Linear projections for queries, keys and values (project then split into heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # projects input -> Q_all
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # projects input -> K_all
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # projects input -> V_all

        # Output linear projection that combines concatenated head outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # projects heads concat -> output

    def split_heads(self, X):
        """Split the last dimension into (num_heads, d_k) and transpose.

        Args:
            X: Tensor of shape (B, T, D)
        Returns:
            Tensor of shape (B, h, T, d_k)
        """
        # Unpack batch, time, and feature dims
        B, T, D = X.shape
        # Reshape to separate heads and then transpose head dim upfront
        X = X.view(B, T, self.num_heads, self.d_k).transpose(1,2)  # (B,h,T,d_k)
        return X

    def combine_heads(self, X):
        """Inverse of split_heads: transpose and merge heads into feature dim.

        Args:
            X: Tensor of shape (B, h, T, d_k)
        Returns:
            Tensor of shape (B, T, D)
        """
        # Unpack shapes
        B, h, T, d_k = X.shape
        # Transpose to (B, T, h, d_k) then flatten the last two dims
        X = X.transpose(1,2).contiguous().view(B, T, h*d_k)  # (B,T,D)
        return X

    def forward(self, X, mask=None):
        """Forward pass for TinyMultiHeadAttention.

        Args:
            X: Input tensor of shape (B, T, D=d_model).
            mask: Optional additive mask to apply to attention logits.

        Returns:
            out_proj: Output tensor of shape (B, T, D).
            attn: Attention weights from scaled_dot_product_attention (B, h, T, T).
        """
        # Project inputs to combined Q/K/V of shape (B, T, D)
        Q_all = self.W_q(X)  # (B, T, D)
        K_all = self.W_k(X)  # (B, T, D)
        V_all = self.W_v(X)  # (B, T, D)

        # Split the combined Q/K/V into multiple heads: (B, h, T, d_k)
        Q = self.split_heads(Q_all)
        K = self.split_heads(K_all)
        V = self.split_heads(V_all)

        # Compute attention per head using scaled dot-product attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # Combine head outputs back into (B, T, D)
        out_combined = self.combine_heads(out)

        # Final linear projection
        out_proj = self.W_o(out_combined)

        return out_proj, attn


def sinusoidal_positional_encoding(T:int, d_model:int, device):
    """Create sinusoidal positional encodings.

    Implements the original formulation from Vaswani et al. where each dimension
    of the positional encoding uses a different frequency.

    Args:
        T: Sequence length (number of positions).
        d_model: Model dimensionality (must be even to pair sin/cos dims nicely).
        device: Torch device for the returned tensor.

    Returns:
        PE: Tensor of shape (T, d_model) containing positional encodings.
    """
    # Ensure d_model is even so even/odd pairing works
    assert d_model % 2 == 0, "d_model must be even for sinusoidal positional encoding"

    # position indices (T, 1) as float
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)

    # dimension indices (1, d_model) as float
    i = torch.arange(d_model, device=device, dtype=torch.float32).unsqueeze(0)

    # compute the rate term 1/10000^{2i/d_model}
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d_model)

    # outer product to get angles for every position and dimension
    angles = pos * angle_rates  # (T, d_model)

    # allocate and fill even/odd indices with sin/cos
    PE = torch.zeros((T, d_model), device=device)
    PE[:, 0::2] = torch.sin(angles[:, 0::2])
    PE[:, 1::2] = torch.cos(angles[:, 1::2])
    return PE

def main():
    # Use GPU when available for faster training/visuals; fall back to CPU otherwise
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', DEVICE)
    # Quick shape test: verify the function returns expected tensor shapes
    B, h, T, d_k, d_v = 2, 3, 4, 8, 8  # batch, heads, time, key-dim, value-dim
    Q = torch.randn(B, h, T, d_k)  # random queries
    K = torch.randn(B, h, T, d_k)  # random keys
    V = torch.randn(B, h, T, d_v)  # random values
    out, attn = scaled_dot_product_attention(Q, K, V)  # call the function
    assert out.shape == (B, h, T, d_v) and attn.shape == (B, h, T, T)  # sanity assert
    print('Scaled dot-product attention shapes OK:', out.shape, attn.shape)

    x = torch.linspace(-10, 10, 400)
    unscaled = torch.softmax(x, dim=0)
    scaled = torch.softmax(x/4, dim=0)
    plt.figure(figsize=(6,4))
    plt.plot(x, unscaled, label='Unscaled')
    plt.plot(x, scaled, label='Scaled by sqrt(d_k)')
    plt.legend(); plt.title('Softmax Saturation vs. Scaling'); plt.xlabel('logit'); plt.ylabel('probability'); plt.grid(True)
    plt.show()

    # Quick visual/print check to ensure masked attention has zeros in upper triangle
    B, h, T, d = 1, 2, 6, 8  # small example sizes for demonstration
    Q = torch.randn(B,h,T,d)  # random queries
    K = torch.randn(B,h,T,d)  # random keys
    V = torch.randn(B,h,T,d)  # random values
    mask = causal_mask(T, T, dtype=Q.dtype)  # create causal mask with same dtype as Q
    _, attn = scaled_dot_product_attention(Q,K,V,mask)  # compute attention with mask
    print('Masked attention OK, upper-triangular ~ 0? Check a head slice:')
    print(attn[0,0])  # print attention weights for head 0, batch 0

    # Deterministic seed (change this integer to get different but reproducible demos)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility when using CUDA, set cuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    B, h, T, d = 1, 1, 6, 8
    Q = torch.randn(B,h,T,d)
    K = torch.randn(B,h,T,d)
    V = torch.randn(B,h,T,d)
    # No mask (allow attending everywhere)
    _, attn_no_mask = scaled_dot_product_attention(Q, K, V, mask=None)
    # Causal mask (no peeking ahead)
    mask = causal_mask(T, T, device=Q.device)
    _, attn_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    # Squeeze to 2D for plotting: (T_q, T_k)
    a0 = attn_no_mask[0,0].cpu().numpy()
    a1 = attn_causal[0,0].cpu().numpy()
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.title('No mask'); plt.imshow(a0, aspect='auto', interpolation='nearest'); plt.colorbar(); plt.xlabel('K positions'); plt.ylabel('Q positions')
    plt.subplot(1,2,2); plt.title('Causal mask'); plt.imshow(a1, aspect='auto', interpolation='nearest'); plt.colorbar(); plt.xlabel('K positions'); plt.ylabel('Q positions')
    plt.suptitle(f'Attention w/ and w/o causal mask (head 0) — seed={seed}')
    plt.tight_layout(); plt.show()

    # Print a small numeric slice to inspect zeros above diagonal
    print('No mask (head 0):')
    print(a0)
    print('\nCausal mask (head 0):')
    print(a1)

    # Sanity check
    B,T,D,h = 2,5,32,4
    x = torch.randn(B,T,D)
    mha = TinyMultiHeadAttention(D,h)
    y, attn = mha(x)
    print('Tiny MHA out shape:', y.shape, '| attn:', attn.shape)

    # Quick visualization
    T, D = 32, 64
    PE = sinusoidal_positional_encoding(T, D)
    plt.figure(figsize=(6,3))
    plt.imshow(PE.cpu().numpy().T, aspect='auto', interpolation='nearest')
    plt.title('Sinusoidal Positional Encoding (dims x positions)')
    plt.xlabel('Position'); plt.ylabel('Dimension'); plt.colorbar(); plt.show()
