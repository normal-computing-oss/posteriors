from dataclasses import field
from typing import Any
from functools import partial
import torch
from optree import tree_map
from optree.integrations.torch import tree_ravel
from tensordict import TensorClass
from posteriors.types import TensorTree, Transform, LogProbFn
from posteriors.tree_utils import tree_size, tree_insert_
from posteriors.utils import (
    per_samplify,
    empirical_fisher,
    is_scalar,
    CatchAuxError,
)


def build(
    log_posterior: LogProbFn,
    per_sample: bool = False,
    init_prec: torch.Tensor | float = 0.0,
) -> Transform:
    """Builds a transform for dense empirical Fisher information
    Laplace approximation.

    The empirical Fisher is defined here as:
    $$
    F(θ) = \\sum_i ∇_θ \\log p(y_i, θ | x_i) ∇_θ \\log p(y_i, θ | x_i)^T
    $$
    where $p(y_i, θ | x_i)$ is the joint model distribution (equivalent to the posterior
    up to proportionality) with parameters $θ$, inputs $x_i$ and labels $y_i$.

    More info on empirical Fisher matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf) and
    their use within a Laplace approximation in [Daxberger et al, 2021](https://arxiv.org/abs/2106.14806).

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        init_prec: Initial precision matrix.
            If it is a float, it is defined as an identity matrix
            scaled by that float.

    Returns:
        Empirical Fisher information Laplace approximation transform instance.
    """
    init_fn = partial(init, init_prec=init_prec)
    update_fn = partial(update, log_posterior=log_posterior, per_sample=per_sample)
    return Transform(init_fn, update_fn)


class DenseLaplaceState(TensorClass["frozen"]):
    """State encoding a Normal distribution over parameters,
    with a dense precision matrix

    Attributes:
        params: Mean of the Normal distribution.
        prec: Precision matrix of the Normal distribution.
        step: Current step count.
    """

    params: TensorTree
    prec: torch.Tensor
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(
    params: TensorTree,
    init_prec: torch.Tensor | float = 0.0,
) -> DenseLaplaceState:
    """Initialise Normal distribution over parameters
    with a dense precision matrix.

    Args:
        params: Mean of the Normal distribution.
        init_prec: Initial precision matrix.
            If it is a float, it is defined as an identity matrix
            scaled by that float.

    Returns:
        Initial DenseLaplaceState.
    """

    if is_scalar(init_prec):
        num_params = tree_size(params)
        init_prec = init_prec * torch.eye(num_params, requires_grad=False)

    return DenseLaplaceState(params, init_prec)


def update(
    state: DenseLaplaceState,
    batch: Any,
    log_posterior: LogProbFn,
    per_sample: bool = False,
    inplace: bool = False,
) -> tuple[DenseLaplaceState, TensorTree]:
    """Adds empirical Fisher information matrix of covariance summed over
    given batch.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        inplace: If True, the state is updated in place. Otherwise, a new
            state is returned.

    Returns:
        Updated DenseLaplaceState and auxiliary information.
    """
    if not per_sample:
        log_posterior = per_samplify(log_posterior)

    with torch.no_grad(), CatchAuxError():
        fisher, aux = empirical_fisher(
            lambda p: log_posterior(p, batch), has_aux=True, normalize=False
        )(state.params)

    if inplace:
        state.prec.data += fisher
        tree_insert_(state.step, state.step + 1)
        return state, aux
    else:
        return DenseLaplaceState(state.params, state.prec + fisher, state.step + 1), aux


def sample(
    state: DenseLaplaceState,
    sample_shape: torch.Size = torch.Size([]),
) -> TensorTree:
    """Sample from Normal distribution over parameters.

    Args:
        state: State encoding mean and precision matrix.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from the Normal distribution.
    """
    samples = torch.distributions.MultivariateNormal(
        loc=torch.zeros(state.prec.shape[0], device=state.prec.device),
        precision_matrix=state.prec,
        validate_args=False,
    ).sample(sample_shape)
    samples = samples.flatten(end_dim=-2)  # ensure samples is 2D
    mean_flat, unravel_func = tree_ravel(state.params)
    samples += mean_flat
    samples = torch.vmap(unravel_func)(samples)
    samples = tree_map(lambda x: x.reshape(sample_shape + x.shape[-1:]), samples)
    return samples
