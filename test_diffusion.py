"""
Autograding tests for Lab 5: Galaxy Image Generation with Diffusion Models.

Grading (4 points total):
  test_schedule_and_forward  (1 pt) — noise schedule + forward process correct
  test_loss_decreases        (1 pt) — training reduces loss
  test_sample_quality        (1 pt) — saved model generates valid samples
  test_sample_centered       (1 pt) — generated galaxies have radial structure
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from diffusion import EpsNet, compute_loss, ddpm_sample_step, linear_noise_schedule, load_model, q_sample, train_step


def _load_data():
    data = np.load("galaxy_data.npz")
    return {
        k: jnp.array(data[k].astype(np.float32) / 255.0) if k.startswith("X") else jnp.array(data[k])
        for k in data.files
    }


def _generate_samples(n=16):
    """Helper: load saved model, run reverse process, return samples."""
    T = 300
    ns = linear_noise_schedule(1e-4, 0.02, T)
    model = EpsNet()
    params = load_model("model_params.pkl")

    key = jr.PRNGKey(42)
    z = jr.normal(key, (n, 32, 32, 1))
    for t in range(T - 1, -1, -1):
        key, step_key = jr.split(key)
        z = ddpm_sample_step(model, params, z, t, ns, step_key)
    return np.array(z)


def test_schedule_and_forward():
    """Noise schedule and forward process are correct. (1 pt)"""
    T = 300
    ns = linear_noise_schedule(1e-4, 0.02, T)

    # --- Schedule checks ---
    assert ns["betas"].shape == (T,), f"betas shape: expected ({T},), got {ns['betas'].shape}"
    assert ns["alphas"].shape == (T,), f"alphas shape: expected ({T},), got {ns['alphas'].shape}"
    assert ns["alpha_bars"].shape == (T,), f"alpha_bars shape: expected ({T},), got {ns['alpha_bars'].shape}"
    diffs = np.diff(np.array(ns["alpha_bars"]))
    assert np.all(diffs < 0), "alpha_bars should be monotonically decreasing"
    assert float(ns["alpha_bars"][0]) > 0.99, f"alpha_bars[0] = {float(ns['alpha_bars'][0]):.4f}, expected > 0.99"
    assert float(ns["alpha_bars"][-1]) < 0.5, f"alpha_bars[-1] = {float(ns['alpha_bars'][-1]):.4f}, expected < 0.5"

    # --- Forward process checks ---
    key = jr.PRNGKey(0)
    x_0 = jr.uniform(key, (8, 32, 32, 1))
    noise = jr.normal(jr.PRNGKey(1), x_0.shape)

    # At t=0: almost no noise
    t_early = jnp.zeros(8, dtype=jnp.int32)
    z_early = q_sample(x_0, t_early, noise, ns["alpha_bars"])
    assert z_early.shape == x_0.shape, f"Expected shape {x_0.shape}, got {z_early.shape}"
    early_diff = float(jnp.mean((z_early - x_0) ** 2))
    assert early_diff < 0.01, f"At t=0, z_t should be close to x_0 (MSE={early_diff:.4f})"

    # At t=T-1: mostly noise
    t_late = jnp.full(8, T - 1, dtype=jnp.int32)
    z_late = q_sample(x_0, t_late, noise, ns["alpha_bars"])
    assert z_late.shape == x_0.shape
    dist_to_noise = float(jnp.mean((z_late - noise) ** 2))
    dist_to_data = float(jnp.mean((z_late - x_0) ** 2))
    assert dist_to_noise < dist_to_data, "At t=T-1, z_t should be closer to noise than to x_0"


def test_loss_decreases():
    """Train for 20 steps on a small batch; loss should decrease. (1 pt)"""
    data = _load_data()
    X_batch = data["X_train"][:64]

    T = 300
    ns = linear_noise_schedule(1e-4, 0.02, T)

    model = EpsNet()
    key = jr.PRNGKey(0)
    key, init_key = jr.split(key)
    dummy_t = jnp.zeros(64, dtype=jnp.int32)
    params = model.init(init_key, X_batch, dummy_t)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    key, loss_key = jr.split(key)
    initial_loss = float(compute_loss(params, model, X_batch, loss_key, ns["alpha_bars"], T))

    for _ in range(20):
        key, step_key = jr.split(key)
        params, opt_state, loss = train_step(
            params, opt_state, X_batch, model, optimizer, step_key, ns["alpha_bars"], T
        )

    final_loss = float(loss)
    assert np.isfinite(initial_loss), "Initial loss is not finite"
    assert np.isfinite(final_loss), "Final loss is not finite"
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_sample_quality():
    """Saved model generates valid samples. (1 pt)"""
    samples = _generate_samples(n=16)

    # Samples should be finite
    assert np.all(np.isfinite(samples)), "Samples contain NaN or Inf"

    # Samples should be in roughly [0, 1] range
    clipped = np.clip(samples, 0, 1)
    clip_diff = np.mean(np.abs(samples - clipped))
    assert clip_diff < 0.1, f"Samples are far from [0,1] range (mean clip diff = {clip_diff:.3f})"

    # Samples should have reasonable variance (not constant)
    assert float(np.std(samples)) > 0.01, "Samples have near-zero variance (model may not have trained)"


def test_sample_centered():
    """Generated galaxies should be brighter at the center than the edges. (1 pt)

    Real galaxies are centered objects. A good model should capture this
    spatial structure, not scatter blobs uniformly across the image.

    Hint: consider adding spatial information (e.g. radial distance from
    center) as an extra input channel to your network.
    """
    samples = np.clip(_generate_samples(n=16), 0, 1)

    # Center: 12x12 crop in the middle
    center = samples[:, 10:22, 10:22, 0]

    # Edge: 4-pixel border strip
    edge_mask = np.ones((32, 32), dtype=bool)
    edge_mask[4:28, 4:28] = False
    edge = samples[:, :, :, 0][:, edge_mask]

    center_mean = float(np.mean(center))
    edge_mean = float(np.mean(edge))

    ratio = center_mean / (edge_mean + 1e-8)
    assert ratio > 2.5, (
        f"Center/edge intensity ratio = {ratio:.2f}, expected > 2.5. "
        f"Generated galaxies should be brighter at the center. "
        f"Consider adding spatial information to your network."
    )
