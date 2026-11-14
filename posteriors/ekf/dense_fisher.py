from dataclasses import field
from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree.integrations.torch import tree_ravel
from tensordict import TensorClass

from posteriors.tree_utils import tree_size, tree_insert_
from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.utils import (
    per_samplify,
    empirical_fisher,
    is_scalar,
    CatchAuxError,
)


def build(
    log_likelihood: LogProbFn,
    lr: float | Schedule,
    transition_cov: torch.Tensor | float = 0.0,
    per_sample: bool = False,
    init_cov: torch.Tensor | float = 1.0,
) -> Transform:
    """Builds a transform to implement an extended Kalman Filter update.

    EKF applies an online update to a Gaussian posterior over the parameters.

    The approximate Bayesian update is based on the linearization
    $$
    \\log p(θ | y) ≈ \\log p(θ) +  ε g(μ)ᵀ(θ - μ) +  \\frac12 ε (θ - μ)^T F(μ) (θ - μ)
    $$
    where $μ$ is the mean of the prior distribution, $ε$ is the learning rate
    (or equivalently the likelihood inverse temperature),
    $g(μ)$ is the gradient of the log likelihood at μ and $F(μ)$ is the
    empirical Fisher information matrix at $μ$ for data $y$.

    For more information on extended Kalman filtering as well as an equivalence
    to (online) natural gradient descent see [Ollivier, 2019](https://arxiv.org/abs/1703.00209).

    Args:
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        transition_cov: Covariance of the transition noise, to additively
            inflate the covariance before the update.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        init_cov: Initial covariance of the Normal distribution. Can be torch.Tensor or scalar.

    Returns:
        EKF transform instance.
    """
    init_fn = partial(init, init_cov=init_cov)
    update_fn = partial(
        update,
        log_likelihood=log_likelihood,
        lr=lr,
        transition_cov=transition_cov,
        per_sample=per_sample,
    )
    return Transform(init_fn, update_fn)


class EKFDenseState(TensorClass["frozen"]):
    """State encoding a Normal distribution over parameters.

    Attributes:
        params: Mean of the Normal distribution.
        cov: Covariance matrix of the
            Normal distribution.
        log_likelihood: Log likelihood of the data given the parameters.
        step: Current step count.
    """

    params: TensorTree
    cov: torch.Tensor
    log_likelihood: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(
    params: TensorTree,
    init_cov: torch.Tensor | float = 1.0,
) -> EKFDenseState:
    """Initialise Multivariate Normal distribution over parameters.

    Args:
        params: Initial mean of the Normal distribution.
        init_cov: Initial covariance matrix of the Multivariate Normal distribution.
            If it is a float, it is defined as an identity matrix scaled by that float.

    Returns:
        Initial EKFDenseState.
    """
    if is_scalar(init_cov):
        num_params = tree_size(params)
        init_cov = init_cov * torch.eye(num_params, requires_grad=False)

    return EKFDenseState(params, init_cov)


def update(
    state: EKFDenseState,
    batch: Any,
    log_likelihood: LogProbFn,
    lr: float,
    transition_cov: torch.Tensor | float = 0.0,
    per_sample: bool = False,
    inplace: bool = False,
) -> tuple[EKFDenseState, TensorTree]:
    """Applies an extended Kalman Filter update to the Multivariate Normal distribution.

    See [build](dense_fisher.md#posteriors.ekf.dense_fisher.build) for details.

    Args:
        state: Current state.
        batch: Input data to log_likelihood.
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            Scalar or schedule (callable taking step index, returning scalar).
        transition_cov: Covariance of the transition noise, to additively
            inflate the covariance before the update.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        inplace: Whether to update the state parameters in-place.

    Returns:
        Updated EKFDenseState and auxiliary information.
    """
    if not per_sample:
        log_likelihood = per_samplify(log_likelihood)

    lr = lr(state.step) if callable(lr) else lr

    with torch.no_grad(), CatchAuxError():

        def log_likelihood_reduced(params, batch):
            per_samp_log_lik, internal_aux = log_likelihood(params, batch)
            return per_samp_log_lik.mean(), internal_aux

        grad, (log_liks, aux) = grad_and_value(log_likelihood_reduced, has_aux=True)(
            state.params, batch
        )
        fisher, _ = empirical_fisher(
            lambda p: log_likelihood(p, batch), has_aux=True, normalize=True
        )(state.params)

        predict_cov = state.cov + transition_cov
        predict_cov_inv = torch.cholesky_inverse(torch.linalg.cholesky(predict_cov))
        update_cov_inv = predict_cov_inv - lr * fisher
        update_cov = torch.cholesky_inverse(torch.linalg.cholesky(update_cov_inv))

        mu_raveled, mu_unravel_f = tree_ravel(state.params)
        update_mean = mu_raveled + lr * update_cov @ tree_ravel(grad)[0]
        update_mean = mu_unravel_f(update_mean)

    if inplace:
        tree_insert_(state.params, update_mean)
        tree_insert_(state.cov, update_cov)
        tree_insert_(state.log_likelihood, log_liks.mean().detach())
        tree_insert_(state.step, state.step + 1)
        return state, aux

    return EKFDenseState(
        update_mean, update_cov, log_liks.mean().detach(), state.step + 1
    ), aux


def sample(
    state: EKFDenseState, sample_shape: torch.Size = torch.Size([])
) -> TensorTree:
    """Single sample from Multivariate Normal distribution over parameters.

    Args:
        state: State encoding mean and covariance.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from Multivariate Normal distribution.
    """
    mean_flat, unravel_func = tree_ravel(state.params)

    samples = torch.distributions.MultivariateNormal(
        loc=mean_flat,
        covariance_matrix=state.cov,
        validate_args=False,
    ).sample(sample_shape)

    samples = torch.vmap(unravel_func)(samples)
    return samples
