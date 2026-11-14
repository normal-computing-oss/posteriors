from dataclasses import field
from functools import partial
from typing import Any
import torch
from optree import tree_map
from optree.integrations.torch import tree_ravel
from tensordict import TensorClass

from posteriors.types import (
    TensorTree,
    Transform,
    ForwardFn,
    OuterLogProbFn,
)
from posteriors.tree_utils import tree_insert_, tree_size
from posteriors.utils import (
    ggn,
    is_scalar,
    CatchAuxError,
)


def build(
    forward: ForwardFn,
    outer_log_likelihood: OuterLogProbFn,
    init_prec: TensorTree | float = 0.0,
) -> Transform:
    """Builds a transform for a Generalized Gauss-Newton (GGN)
    Laplace approximation.

    Equivalent to the (non-empirical) Fisher information matrix when
    the `outer_log_likelihood` is exponential family with natural parameter equal to
    the output from `forward`.

    `forward` should output auxiliary information (or `torch.tensor([])`),
    `outer_log_likelihood` should not.

    The GGN is defined as
    $$
    G(θ) = J_f(θ) H_l(z) J_f(θ)^T
    $$
    where $z = f(θ)$ is the output of the forward function $f$ and $l(z)$
    is a loss (negative log-likelihood) that maps the output of $f$ to a scalar output.

    More info on Fisher and GGN matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf) and
    their use within a Laplace approximation in [Daxberger et al, 2021](https://arxiv.org/abs/2106.14806).

    Args:
        forward: Function that takes parameters and input batch and
            returns a forward value (e.g. logits), not reduced over the batch,
            as well as auxiliary information.
        outer_log_likelihood: A function that takes the output of `forward` and batch
            then returns the log likelihood of the model output,
            with no auxiliary information.
        init_prec: Initial precision matrix.
            If it is a float, it is defined as an identity matrix
            scaled by that float.

    Returns:
        GGN Laplace approximation transform instance.
    """
    init_fn = partial(init, init_prec=init_prec)
    update_fn = partial(
        update, forward=forward, outer_log_likelihood=outer_log_likelihood
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
    forward: ForwardFn,
    outer_log_likelihood: OuterLogProbFn,
    inplace: bool = False,
) -> tuple[DenseLaplaceState, TensorTree]:
    """Adds GGN matrix over given batch.

    Args:
        state: Current state.
        batch: Input data to model.
        forward: Function that takes parameters and input batch and
            returns a forward value (e.g. logits), not reduced over the batch,
            as well as auxiliary information.
        outer_log_likelihood: A function that takes the output of `forward` and batch
            then returns the log likelihood of the model output,
            with no auxiliary information.
        inplace: If True, then the state is updated in place, otherwise a new state
            is returned.

    Returns:
        Updated DenseLaplaceState and auxiliary information.
    """

    def outer_loss(z, batch):
        return -outer_log_likelihood(z, batch)

    with torch.no_grad(), CatchAuxError():
        ggn_batch, aux = ggn(
            lambda params: forward(params, batch),
            lambda z: outer_loss(z, batch),
            forward_has_aux=True,
            loss_has_aux=False,
            normalize=False,
        )(state.params)

    if inplace:
        state.prec.data += ggn_batch
        tree_insert_(state.step, state.step + 1)
        return state, aux
    else:
        return DenseLaplaceState(
            state.params, state.prec + ggn_batch, state.step + 1
        ), aux


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
