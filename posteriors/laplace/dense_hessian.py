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
    is_scalar,
    CatchAuxError,
)

from torch.func import jacrev, jacfwd


def build(
    log_posterior: LogProbFn,
    init_prec: torch.Tensor | float = 0.0,
    epsilon: float = 0.0,
    rescale: float = 1.0,
) -> Transform:
    """Builds a transform for dense Hessian Laplace.

    **Warning:**
    The Hessian is not guaranteed to be positive definite,
    so setting epsilon > 0 ought to be considered.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        init_prec: Initial precision matrix.
            If it is a float, it is defined as an identity matrix
            scaled by that float.
        epsilon: Added to the diagonal of the Hessian
            for numerical stability.
        rescale: Value to multiply the Hessian by
            (i.e. to normalize by batch size)

    Returns:
        Hessian Laplace transform instance.
    """
    init_fn = partial(init, init_prec=init_prec)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        epsilon=epsilon,
        rescale=rescale,
    )
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
    epsilon: float = 0.0,
    rescale: float = 1.0,
    inplace: bool = False,
) -> tuple[DenseLaplaceState, TensorTree]:
    """Adds the Hessian of the negative log-posterior over given batch.

    **Warning:**
    The Hessian is not guaranteed to be positive definite,
    so setting epsilon > 0 ought to be considered.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
        epsilon: Added to the diagonal of the Hessian
            for numerical stability.
        rescale: Value to multiply the Hessian by
            (i.e. to normalize by batch size)
        inplace: If True, the state is updated in place. Otherwise, a new
            state is returned.

    Returns:
        Updated DenseLaplaceState and auxiliary information.
    """
    with torch.no_grad(), CatchAuxError():
        flat_params, params_unravel = tree_ravel(state.params)
        num_params = flat_params.numel()

        def neg_log_p(p_flat):
            value, aux = log_posterior(params_unravel(p_flat), batch)
            return -value, aux

        hess, aux = jacfwd(jacrev(neg_log_p, has_aux=True), has_aux=True)(flat_params)
        hess = hess * rescale + epsilon * torch.eye(num_params)

    if inplace:
        state.prec.data += hess
        tree_insert_(state.step, state.step + 1)
        return state, aux
    else:
        return DenseLaplaceState(state.params, state.prec + hess, state.step + 1), aux


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
