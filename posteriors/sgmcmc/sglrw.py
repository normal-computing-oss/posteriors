from dataclasses import field

# type: ignore          posteriors is not typed
from typing import Any
from functools import partial
import torch
from torch import Tensor
from torch.func import grad_and_value
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.tree_utils import flexi_tree_map, tree_insert_
from posteriors.utils import CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float | Schedule,
    temperature: float | Schedule = 1.0,
) -> Transform:
    """Builds SGLRW transform - Stochastic Gradient Lattice Random Walk.

    Algorithm from [Mensch et al, 2026](https://arxiv.org/abs/2602.15925)
    adapted from [Duffield et al, 2025](https://arxiv.org/abs/2508.20883):
    $$
    θ_{t+1} = θ_t + δx Δ(θₜ, t)
    $$
    where $δx = √(lr * 2 * T)$ is a spatial stepsize and $Δ(θₜ, t)$ is a random
    binary valued vector defined in the paper.

    Targets $p_T(θ) \\propto \\exp( \\log p(θ) / T)$ with temperature $T$,
    as it discretizes the overdamped Langevin SDE:
    $$
    dθ = ∇ log p_T(θ) dt + √(2 T) dW
    $$

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data and variable batch size.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate,
            scalar or schedule (callable taking step index, returning scalar).
        temperature: Temperature of the sampling distribution.
            Scalar or schedule (callable taking step index, returning scalar).

    Returns:
        SGLRW transform (posteriors.types.Transform instance).
    """
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        temperature=temperature,
    )
    return Transform(init, update_fn)


class SGLRWState(TensorClass["frozen"]):
    """State encoding params for SG-LRW (binary).

    Attributes:
        params: Parameters.
        log_posterior: Last log posterior evaluation.
        step: Current step count.
    """

    params: TensorTree
    log_posterior: Tensor = field(default_factory=lambda: torch.tensor(torch.nan))
    step: Tensor = field(default_factory=lambda: torch.tensor(0))


def init(params: TensorTree) -> SGLRWState:
    """Initialise SG-LRW."""
    return SGLRWState(params)


def update(
    state: SGLRWState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float | Schedule,
    temperature: float | Schedule = 1.0,
    inplace: bool = False,
) -> tuple[SGLRWState, TensorTree]:
    with torch.no_grad(), CatchAuxError():
        grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(
            state.params, batch
        )

    # Resolve schedules
    lr_val = lr(state.step) if callable(lr) else lr
    T_val = temperature(state.step) if callable(temperature) else temperature
    # Spatial stepsize to make update binary
    diffusion_val = (2.0 * T_val) ** 0.5
    delta_x = lr_val**0.5 * diffusion_val

    # Per-parameter binary LRW transform
    def transform_params(p, g):
        p_plus = ternary_probs(g, diffusion_val, lr_val, delta_x)[..., 2]

        u = torch.rand_like(p_plus)
        step_sign = torch.where(
            u < p_plus, torch.ones_like(p_plus), -torch.ones_like(p_plus)
        )
        step = delta_x * step_sign
        return p + step

    params = flexi_tree_map(transform_params, state.params, grads, inplace=inplace)

    if inplace:
        tree_insert_(state.log_posterior, log_post.detach())
        tree_insert_(state.step, state.step + 1)
        return state, aux
    return SGLRWState(params, log_post.detach(), state.step + 1), aux


def ternary_probs(
    drift_val: Tensor,
    diffusion_val: Tensor,
    stepsize: Tensor,
    delta_x: Tensor,
) -> Tensor:
    """
    Generate the probabilities for the ternary update
    from the discretization parameters.

    Args:
        drift_val: Evaluation of the Drift function.
        diffusion_val: Evaluation of the Diffusion function.
        stepsize: Temporal stepsize value.
        delta_x: Spatial stepsize value.

    Returns:
        Update probabilities as a tensor, with last axis being [p_minus, p_zero, p_plus].
    """
    diffusion_val = torch.as_tensor(
        diffusion_val, dtype=drift_val.dtype, device=drift_val.device
    )
    stepsize = torch.as_tensor(stepsize, dtype=drift_val.dtype, device=drift_val.device)
    delta_x = torch.as_tensor(delta_x, dtype=drift_val.dtype, device=drift_val.device)
    desired_mean = stepsize * drift_val
    desired_var = stepsize * diffusion_val**2
    scaled_mean = desired_mean / delta_x
    scaled_var = desired_var / delta_x**2

    # Ensure p_minus + p_plus <= 1
    scaled_var = torch.clamp(scaled_var, 0.0, 1.0)

    # Ensure positive probs
    scaled_mean = torch.clamp(scaled_mean, -scaled_var, scaled_var)

    # Clip probs for numerical stability
    p_plus = torch.clamp(0.5 * (scaled_var + scaled_mean), 0.0, 1.0)
    p_minus = torch.clamp(0.5 * (scaled_var - scaled_mean), 0.0, 1.0)
    p_zero = torch.clamp(1 - p_plus - p_minus, 0.0, 1.0)

    return torch.stack([p_minus, p_zero, p_plus], dim=-1)
