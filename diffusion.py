"""
Lab 5: Galaxy Image Generation with Diffusion Models
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

IMG_SHAPE = (32, 32, 1)


# ============================================================
# TODO #1: Implement the noise schedule
# ============================================================


def linear_noise_schedule(beta_start, beta_end, T):
    """Compute the linear noise schedule for diffusion.

    Args:
        beta_start: Starting noise level (e.g. 1e-4)
        beta_end: Ending noise level (e.g. 0.02)
        T: Number of diffusion timesteps

    Returns:
        Dictionary with keys:
            "betas":      (T,) noise levels, linearly spaced from beta_start to beta_end
            "alphas":     (T,) where alpha_t = 1 - beta_t
            "alpha_bars": (T,) cumulative product of alphas
    """
    # TODO: Implement the linear noise schedule.
    # 1. Create T evenly spaced beta values from beta_start to beta_end
    # 2. Compute alphas = 1 - betas
    # 3. Compute alpha_bars as the cumulative product of alphas (jnp.cumprod)
    # Return a dict with keys "betas", "alphas", "alpha_bars"

    ...

    # return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


# ============================================================
# TODO #2: Implement the diffusion kernel
# ============================================================


def q_sample(x_0, t, noise, alpha_bars):
    """Sample from the diffusion kernel q(z_t | x_0).

    Because each forward step is Gaussian, the composition is also Gaussian
    and we can jump directly to any timestep t in closed form:

        z_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        x_0: Clean images, shape (batch, 32, 32, 1)
        t: Timesteps, shape (batch,), integer indices into alpha_bars
        noise: Pre-sampled noise epsilon ~ N(0, I), same shape as x_0
        alpha_bars: Cumulative product of alphas, shape (T,)

    Returns:
        z_t: Noisy images at timestep t, same shape as x_0

    Hint: Use alpha_bars[t] to index, then reshape with [:, None, None, None]
    for broadcasting with the (batch, H, W, C) image tensor.
    """
    # TODO: Implement the diffusion kernel (closed-form sampling).

    ...


# ============================================================
# GIVEN: Sinusoidal time embedding (do not modify)
# ============================================================


def sinusoidal_embedding(t, dim=64):
    """Sinusoidal positional embedding for timesteps.

    Maps integer timesteps to continuous vectors using sin/cos at
    different frequencies, letting the network distinguish timesteps.

    Args:
        t: Integer timesteps, shape (batch,)
        dim: Embedding dimension (must be even)

    Returns:
        Embeddings of shape (batch, dim)
    """
    half_dim = dim // 2
    emb = jnp.log(10000.0) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None].astype(jnp.float32) * emb[None, :]
    return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


# ============================================================
# TODO #3: Implement the denoising network
# ============================================================


class EpsNet(nn.Module):
    """Time-conditioned network for noise prediction: epsilon_theta(z_t, t).

    Input:  z_t of shape (batch, 32, 32, 1) and t of shape (batch,)
    Output: predicted noise — same shape as z_t

    Requirements:
    - Output must have the same shape as z_t.
    - The network must be conditioned on t. Use sinusoidal_embedding(t, dim)
      to embed the timestep, then inject it into intermediate features.
      Flax Conv uses padding='SAME' by default, so spatial dims are preserved.

    Spatial structure hint:
    Galaxy images are rotation-symmetric but NOT translation-invariant —
    structure concentrates at the center, not uniformly across the image.
    A standard CNN cannot tell where in the image it is (it applies the same
    filter everywhere). Think about how to give the network information about
    each pixel's position relative to the center.
    """

    @nn.compact
    def __call__(self, z_t, t):
        # TODO: Implement the noise predictor network.
        # Return predicted noise with the same shape as z_t.

        ...


# ============================================================
# TODO #4: Implement the training loss
# ============================================================


def compute_loss(params, model, x_batch, key, alpha_bars, T):
    """Compute the diffusion training loss: E[||noise - predicted_noise||^2].

    Training procedure (from the lecture):
    1. Sample random timesteps t ~ Uniform(0, T-1) for each image in the batch (remember to use the key for randomness)
    2. Sample noise epsilon ~ N(0, I) for each image
    3. Compute noisy images by using the diffusion kernel q_sample
    4. Predict noise by running a forward pass through the `model`
    5. Return mean squared error: mean((noise - pred_noise)^2)

    Args:
        params: Model parameters
        model: EpsNet instance
        x_batch: Clean images, shape (batch, 32, 32, 1)
        key: JAX random key
        alpha_bars: Cumulative product of alphas, shape (T,)
        T: Number of timesteps

    Returns:
        loss: Scalar MSE loss

    Hint: Use jr.split(key) to get separate keys for timestep sampling
    and noise sampling. Use jr.randint for timesteps, jr.normal for noise.
    """
    # TODO: Implement the diffusion training loss.

    ...


# ============================================================
# GIVEN: Training step (do not modify)
# ============================================================


def train_step(params, opt_state, x_batch, model, optimizer, key, alpha_bars, T):
    """Single training step: compute loss, gradients, update parameters."""
    loss, grads = jax.value_and_grad(compute_loss)(params, model, x_batch, key, alpha_bars, T)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# ============================================================
# GIVEN: Reverse sampling step (do not modify)
# ============================================================


def ddpm_sample_step(model, params, z_t, t, noise_schedule, key):
    """One step of DDPM reverse process: z_t -> z_{t-1}.

    Reverse step formula:
        mu = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1 - alpha_bar_t) * pred_noise)
        z_{t-1} = mu + sqrt(beta_t) * eta,   eta ~ N(0, I)   [skip noise at t=0]
    """
    beta_t = noise_schedule["betas"][t]
    alpha_t = noise_schedule["alphas"][t]
    alpha_bar_t = noise_schedule["alpha_bars"][t]

    batch_size = z_t.shape[0]
    t_batch = jnp.full((batch_size,), t)
    pred_noise = model.apply(params, z_t, t_batch)

    mu = (1.0 / jnp.sqrt(alpha_t)) * (z_t - (beta_t / jnp.sqrt(1.0 - alpha_bar_t)) * pred_noise)

    noise = jax.random.normal(key, z_t.shape)
    return mu + jnp.sqrt(beta_t) * noise * (t > 0)


# ============================================================
# GIVEN: Save and load model (do not modify)
# ============================================================


def save_model(params, path="model_params.pkl"):
    """Save model parameters to a file."""
    import pickle

    params_np = jax.tree.map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(params_np, f)


def load_model(path="model_params.pkl"):
    """Load model parameters from a file."""
    import pickle

    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree.map(jnp.asarray, params_np)
