from dataclasses import field
from typing import Any
from functools import partial
import torch
from torch.func import jacrev
from optree import tree_map
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.tree_utils import flexi_tree_map, tree_insert_
from posteriors.utils import (
    diag_normal_sample,
    per_samplify,
    is_scalar,
    CatchAuxError,
)


def build(
    log_likelihood: LogProbFn,
    lr: float | Schedule,
    transition_sd: float = 0.0,
    per_sample: bool = False,
    init_sds: TensorTree | float = 1.0,
) -> Transform:
    """Builds a transform to implement an extended Kalman Filter update.

    EKF applies an online update to a (diagonal) Gaussian posterior over the parameters.

    The approximate Bayesian update is based on the linearization
    $$
    \\log p(θ | y) ≈ \\log p(θ) +  ε g(μ)ᵀ(θ - μ) +  \\frac12 ε (θ - μ)^T F_d(μ) (θ - μ)
    $$
    where $μ$ is the mean of the prior distribution, $ε$ is the learning rate
    (or equivalently the likelihood inverse temperature),
    $g(μ)$ is the gradient of the log likelihood at μ and $F_d(μ)$ is the diagonal
    empirical Fisher information matrix at $μ$ for data $y$. Completing the square
    regains a diagonal Normal distribution over the parameters.

    For more information on extended Kalman filtering as well as an equivalence
    to (online) natural gradient descent see [Ollivier, 2019](https://arxiv.org/abs/1703.00209).

    Args:
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        transition_sd: Standard deviation of the transition noise, to additively
            inflate the diagonal covariance before the update.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        init_sds: Initial square-root diagonal of the covariance matrix
            of the Normal distribution. Can be tree like params or scalar.

    Returns:
        Diagonal EKF transform instance.
    """
    init_fn = partial(init, init_sds=init_sds)
    update_fn = partial(
        update,
        log_likelihood=log_likelihood,
        lr=lr,
        transition_sd=transition_sd,
        per_sample=per_sample,
    )
    return Transform(init_fn, update_fn)


class EKFDiagState(TensorClass["frozen"]):
    """State encoding a diagonal Normal distribution over parameters.

    Attributes:
        params: Mean of the Normal distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            Normal distribution.
        log_likelihood: Log likelihood of the data given the parameters.
        step: Current step count.
    """

    params: TensorTree
    sd_diag: TensorTree
    log_likelihood: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(
    params: TensorTree,
    init_sds: TensorTree | float = 1.0,
) -> EKFDiagState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Initial mean of the Normal distribution.
        init_sds: Initial square-root diagonal of the covariance matrix
            of the Normal distribution. Can be tree like params or scalar.

    Returns:
        Initial EKFDiagState.
    """
    if is_scalar(init_sds):
        init_sds = tree_map(
            lambda x: torch.full_like(x, init_sds, requires_grad=x.requires_grad),
            params,
        )

    return EKFDiagState(params, init_sds)


def update(
    state: EKFDiagState,
    batch: Any,
    log_likelihood: LogProbFn,
    lr: float | Schedule,
    transition_sd: float = 0.0,
    per_sample: bool = False,
    inplace: bool = False,
) -> tuple[EKFDiagState, TensorTree]:
    """Applies an extended Kalman Filter update to the diagonal Normal distribution.

    See [build](diag_fisher.md#posteriors.ekf.diag_fisher.build) for details.

    Args:
        state: Current state.
        batch: Input data to log_likelihood.
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        transition_sd: Standard deviation of the transition noise, to additively
            inflate the diagonal covariance before the update.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        inplace: Whether to update the state parameters in-place.

    Returns:
        Updated EKFDiagState and auxiliary information.
    """

    if not per_sample:
        log_likelihood = per_samplify(log_likelihood)

    lr = lr(state.step) if callable(lr) else lr

    predict_sd_diag = flexi_tree_map(
        lambda x: (x**2 + transition_sd**2) ** 0.5, state.sd_diag, inplace=inplace
    )
    with torch.no_grad(), CatchAuxError():
        log_liks, aux = log_likelihood(state.params, batch)
        jac, _ = jacrev(log_likelihood, has_aux=True)(state.params, batch)
        grad = tree_map(lambda x: x.mean(0), jac)
        diag_lik_hessian_approx = tree_map(lambda x: -(x**2).mean(0), jac)

    update_sd_diag = flexi_tree_map(
        lambda sig, h: (sig**-2 - lr * h) ** -0.5,
        predict_sd_diag,
        diag_lik_hessian_approx,
        inplace=inplace,
    )
    update_mean = flexi_tree_map(
        lambda mu, sig, g: mu + sig**2 * lr * g,
        state.params,
        update_sd_diag,
        grad,
        inplace=inplace,
    )

    if inplace:
        tree_insert_(state.log_likelihood, log_liks.mean().detach())
        tree_insert_(state.step, state.step + 1)
        return state, aux

    return EKFDiagState(
        update_mean, update_sd_diag, log_liks.mean().detach(), state.step + 1
    ), aux


def sample(
    state: EKFDiagState, sample_shape: torch.Size = torch.Size([])
) -> TensorTree:
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and standard deviations.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from Normal distribution.
    """
    return diag_normal_sample(state.params, state.sd_diag, sample_shape=sample_shape)
