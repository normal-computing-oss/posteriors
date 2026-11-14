from dataclasses import field
from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.tree_utils import flexi_tree_map, tree_insert_
from posteriors.utils import is_scalar, CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float | Schedule,
    alpha: float = 0.01,
    sigma: float = 1.0,
    temperature: float | Schedule = 1.0,
    momenta: TensorTree | float | None = None,
) -> Transform:
    """Builds BAOA transform.

    Algorithm from [Leimkuhler and Matthews, 2015 - p271](https://link.springer.com/ok/10.1007/978-3-319-16375-8).

    BAOA is conjugate to BAOAB (in Leimkuhler and Matthews' terminology) but requires
    only a single gradient evaluation per iteration.
    The two are equivalent when analyzing functions of the parameter trajectory.
    Unlike BAOAB, BAOA is not reversible, but since we don't apply Metropolis-Hastings 
    or momenta reversal, the algorithm remains functionally identical to BAOAB.
    
    \\begin{align}
    m_{t+1/2} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}), \\\\
    θ_{t+1/2} &= θ_t + (ε / 2) σ^{-2} m_{t+1/2}, \\\\
    m_{t+1} &= e^{-ε γ} m_{t+1/2} + N(0, ζ^2 σ^2), \\\\
    θ_{t+1} &= θ_{t+1/2} + (ε / 2) σ^{-2} m_{t+1} \\
    \\end{align}
 
    for learning rate $\\epsilon$, temperature $T$, transformed friction $γ = α σ^{-2}$
    and transformed noise variance$ζ^2 = T(1 - e^{-2γε})$.

    Targets $p_T(θ, m) \\propto \\exp( (\\log p(θ) - \\frac{1}{2σ^2} m^Tm) / T)$
    with temperature $T$.

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data and variable batch size.
    
    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        alpha: Friction coefficient.
        sigma: Standard deviation of momenta target distribution.
        temperature: Temperature of the joint parameter + momenta distribution.
            Scalar or schedule (callable taking step index, returning scalar).
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        BAOA transform instance.
    """
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        alpha=alpha,
        sigma=sigma,
        temperature=temperature,
    )
    return Transform(init_fn, update_fn)


class BAOAState(TensorClass["frozen"]):
    """State encoding params and momenta for BAOA.

    Attributes:
        params: Parameters.
        momenta: Momenta for each parameter.
        log_posterior: Log posterior evaluation.
        step: Current step count.
    """

    params: TensorTree
    momenta: TensorTree
    log_posterior: torch.Tensor = field(default_factory=lambda: torch.tensor(torch.nan))
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(params: TensorTree, momenta: TensorTree | float | None = None) -> BAOAState:
    """Initialise momenta for BAOA.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        Initial BAOAState containing momenta.
    """
    if momenta is None:
        momenta = tree_map(
            lambda x: torch.randn_like(x, requires_grad=x.requires_grad),
            params,
        )
    elif is_scalar(momenta):
        momenta = tree_map(
            lambda x: torch.full_like(x, momenta, requires_grad=x.requires_grad),
            params,
        )

    return BAOAState(params, momenta)


def update(
    state: BAOAState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float | Schedule,
    alpha: float = 0.01,
    sigma: float = 1.0,
    temperature: float | Schedule = 1.0,
    inplace: bool = False,
) -> tuple[BAOAState, TensorTree]:
    """Updates parameters and momenta for BAOA.

    Algorithm from [Leimkuhler and Matthews, 2015 - p271](https://link.springer.com/ok/10.1007/978-3-319-16375-8).

    See [build](baoa.md#posteriors.sgmcmc.baoa.build) for more details.

    Args:
        state: BAOAState containing params and momenta.
        batch: Data batch to be send to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        alpha: Friction coefficient.
        sigma: Standard deviation of momenta target distribution.
        temperature: Temperature of the joint parameter + momenta distribution.
            Scalar or schedule (callable taking step index, returning scalar).
        inplace: Whether to modify state in place.

    Returns:
        Updated state
        (which are pointers to the inputted state tensors if inplace=True)
        and auxiliary information.
    """
    with torch.no_grad(), CatchAuxError():
        grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(
            state.params, batch
        )

    lr = lr(state.step) if callable(lr) else lr
    temperature = temperature(state.step) if callable(temperature) else temperature
    prec = sigma**-2
    gamma = torch.tensor(alpha * prec)
    zeta2 = (temperature * (1 - torch.exp(-2 * gamma * lr))) ** 0.5

    def BB_step(m, g):
        return m + lr * g

    def A_step(p, m):
        return p + (lr / 2) * prec * m

    def O_step(m):
        return torch.exp(-gamma * lr) * m + zeta2 * sigma * torch.randn_like(m)

    momenta = flexi_tree_map(BB_step, state.momenta, grads, inplace=inplace)
    params = flexi_tree_map(A_step, state.params, momenta, inplace=inplace)
    momenta = flexi_tree_map(O_step, momenta, inplace=inplace)
    params = flexi_tree_map(A_step, params, momenta, inplace=inplace)

    if inplace:
        tree_insert_(state.log_posterior, log_post.detach())
        tree_insert_(state.step, state.step + 1)
        return state, aux
    return BAOAState(params, momenta, log_post.detach(), state.step + 1), aux
