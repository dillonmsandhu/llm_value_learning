from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Tuple, Optional, Any, Callable


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Standard in modern SOTA LLMs (e.g. LLaMA 3, Gemma 2, Mistral).
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return normed * scale


def compute_rope_freqs(max_len: int, head_dim: int, theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes rotary position embedding (RoPE) cosine and sine frequencies."""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs_outer = jnp.outer(t, freqs)
    return jnp.cos(freqs_outer), jnp.sin(freqs_outer)


def apply_rope(
    x: jnp.ndarray,
    freqs_cos: jnp.ndarray,
    freqs_sin: jnp.ndarray,
) -> jnp.ndarray:
    """
    Applies Rotary Position Embedding (RoPE) to queries or keys.
    x: shape (batch, seq_len, n_heads, head_dim)
    freqs_cos, freqs_sin: shape (seq_len, head_dim // 2)
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)

    # Broadcast cos and sin: (1, seq_len, 1, head_dim)
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)[:, None, :]
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)[:, None, :]

    # Expand for batch dimension if present
    if x.ndim == 4:
        cos = cos[None, ...]
        sin = sin[None, ...]

    return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Standard FFN in modern SOTA LLMs (LLaMA, Gemma, PaLM, Mistral).
    FFN(x) = (SiLU(x W_gate) * (x W_up)) W_down
    """
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.d_ff, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        up = nn.Dense(self.d_ff, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        hidden = nn.silu(gate) * up
        out = nn.Dense(self.d_model, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))(hidden)
        return out


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Embeddings (RoPE) and Pre-LayerNorm.
    """
    d_model: int
    n_heads: int
    max_len: int = 64
    use_rope: bool = True

    def setup(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        self.norm = RMSNorm(self.d_model)

        self.q_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))
        self.k_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))
        self.v_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))
        self.out_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2)))

        if self.use_rope:
            self.freqs_cos, self.freqs_sin = compute_rope_freqs(self.max_len, self.head_dim)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        x: (batch, seq_len, d_model)
        mask: optional attention mask (batch, 1, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Pre-Normalization (LLaMA style)
        x_norm = self.norm(x)

        # 2. Project Q, K, V
        q = self.q_proj(x_norm).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # 3. Rotary Position Embeddings (RoPE)
        if self.use_rope:
            q = apply_rope(q, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])
            k = apply_rope(k, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])

        # 4. Transpose to (batch, n_heads, seq_len, head_dim)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        # 5. Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale

        # 6. Apply Causal Mask (lower triangular)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(weights, v)  # (batch, n_heads, seq_len, head_dim)

        # 7. Merge heads and project output
        context = jnp.swapaxes(context, 1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(context)

        # 8. Residual connection
        return x + out


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-RMSNorm and SwiGLU FFN (LLaMA architecture).
    """
    d_model: int
    n_heads: int
    d_ff: int
    max_len: int = 64
    use_rope: bool = True

    def setup(self):
        self.attn = CausalSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            max_len=self.max_len,
            use_rope=self.use_rope,
        )
        self.ffn_norm = RMSNorm(self.d_model)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Pre-norm Self-Attention + Residual
        x = self.attn(x, mask=mask)
        # Pre-norm SwiGLU FFN + Residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerTorso(nn.Module):
    """
    Transformer backbone mapping sequence observations to token representations.
    Includes token embedding, optional BOS token prefix, N transformer blocks, and final RMSNorm.
    """
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: Optional[int] = None
    vocab_size: int = 4   # 0: left branch, 1: right branch, 2: pad token (-1.0), 3: BOS token
    max_len: int = 64
    pad_token_id: int = 0
    use_rope: bool = True
    add_bos: bool = True  # Prepend a learned BOS token so root empty state has representation

    def setup(self):
        ff_dim = self.d_ff if self.d_ff is not None else int(8 / 3 * self.d_model)
        self.token_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=orthogonal(1.0),
        )
        # Ensure RoPE freqs allocate enough positions even with prepended BOS
        rope_max_len = max(self.max_len + 4, 128)
        self.layers = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=ff_dim,
                max_len=rope_max_len,
                use_rope=self.use_rope,
            )
            for _ in range(self.n_layers)
        ]
        self.final_norm = RMSNorm(self.d_model)

    def _obs_to_tokens(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Maps observations to discrete token IDs in [0, vocab_size-1].
        Supports:
        1. Pre-tokenized integer sequences (e.g. ReasoningDAGEnv).
        2. Float vector observations [-1, 0, 1] (e.g. BinaryTreeEnv).
        """
        # Ensure batch dimension
        if x.ndim == 1:
            x = x[None, :]

        if jnp.issubdtype(x.dtype, jnp.integer):
            token_ids = x.astype(jnp.int32)
            valid_mask = (token_ids != self.pad_token_id)
            if self.add_bos:
                # Only prepend BOS if first token is not already BOS
                # But typically for pre-tokenized inputs, caller sets add_bos=False
                pass
            return token_ids, valid_mask

        # Discrete mapping for legacy float observations [-1, 0, 1]
        token_ids = jnp.where(x == 1.0, 1, jnp.where(x == 0.0, 0, 2)).astype(jnp.int32)
        valid_mask = (token_ids != 2)

        if self.add_bos:
            batch_size = x.shape[0]
            bos_tokens = jnp.full((batch_size, 1), 3, dtype=jnp.int32)
            bos_mask = jnp.ones((batch_size, 1), dtype=bool)
            token_ids = jnp.concatenate([bos_tokens, token_ids], axis=1)
            valid_mask = jnp.concatenate([bos_mask, valid_mask], axis=1)

        return token_ids, valid_mask

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            hidden_states: (batch, seq_len, d_model)
            valid_mask: (batch, seq_len)
        """
        token_ids, valid_mask = self._obs_to_tokens(x)
        h = self.token_embed(token_ids)

        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        return h, valid_mask


class TransformerCritic(nn.Module):
    """
    SOTA LLM-Style Transformer Value Network.

    Architecture:
      1. Token Embedding of history prefix + Pre-pended BOS token.
      2. Multi-layer Transformer Decoder blocks with:
         - Pre-Layer RMSNorm
         - Causal Multi-Head Self-Attention
         - Rotary Position Embeddings (RoPE)
         - SwiGLU Feed-Forward Network (SiLU gating)
      3. Final RMSNorm.
      4. Last-valid-token extraction (representing the current reasoning state).
      5. Small MLP on the final layer:
         h -> Dense -> LayerNorm/RMSNorm -> SiLU -> Dense -> SiLU -> phi
      6. Scalar Value Head:
         phi -> Dense(1, kernel_init=zeros, bias_init=zeros) -> V(s).

    Interface matches reference_code/core/networks.py:
      - value(x): Returns scalar value V(s)
      - value_features(x): Returns final MLP feature vector phi
      - value_from_features(phi): Maps phi to scalar V(s)
      - __call__(x): Returns V(s)
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: Optional[int] = None
    final_hidden_dim: int = 64
    vocab_size: int = 4
    max_len: int = 64
    pad_token_id: int = 0
    use_rope: bool = True
    add_bos: bool = True

    def setup(self):
        self.torso = TransformerTorso(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            pad_token_id=self.pad_token_id,
            use_rope=self.use_rope,
            add_bos=self.add_bos,
        )
        # Small MLP on the final pooled representation
        self.mlp_dense1 = nn.Dense(
            self.final_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.mlp_norm = RMSNorm(self.final_hidden_dim)
        self.mlp_dense2 = nn.Dense(
            self.final_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        # Final linear value projection (zero-initialized for stable RL start)
        self.critic_head = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=constant(0.0),
        )

    def _extract_last_valid_token(self, h: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Extracts the hidden representation at the last valid (non-pad) token position.
        h: (batch, seq_len, d_model)
        valid_mask: (batch, seq_len) bool
        Returns:
            (batch, d_model) representation of current state
        """
        # Index of last valid token for each batch element
        # e.g. sum(valid_mask) - 1
        last_indices = jnp.maximum(0, jnp.sum(valid_mask.astype(jnp.int32), axis=-1) - 1)
        batch_indices = jnp.arange(h.shape[0])
        return h[batch_indices, last_indices]

    def value_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Extracts the final hidden features phi for state x.
        Returns:
            phi: (batch, final_hidden_dim) or (final_hidden_dim,) if single observation.
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x[None, :]

        # 1. Forward through transformer torso
        h_seq, valid_mask = self.torso(x)

        # 2. Extract last valid token representation
        h_last = self._extract_last_valid_token(h_seq, valid_mask)

        # 3. Small MLP on top of the final layer
        phi = self.mlp_dense1(h_last)
        phi = self.mlp_norm(phi)
        phi = nn.silu(phi)
        phi = self.mlp_dense2(phi)
        phi = nn.silu(phi)

        if is_single:
            return phi.squeeze(0)
        return phi

    def value_from_features(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Maps hidden features phi to scalar value."""
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns V(s)."""
        phi = self.value_features(x)
        return self.value_from_features(phi)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns V(s)."""
        return self.value(x)


class TransformerPolicy(nn.Module):
    """Independent prefix-only actor using the same feature architecture as the critic."""
    vocab_size: int
    num_actions: int
    max_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    final_hidden_dim: int = 32

    @nn.compact
    def __call__(self, x):
        features = TransformerCritic(
            vocab_size=self.vocab_size, max_len=self.max_len, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            final_hidden_dim=self.final_hidden_dim, add_bos=False,
        ).value_features(x)
        return nn.Dense(self.num_actions, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(features)
