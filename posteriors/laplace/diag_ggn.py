from dataclasses import field
from functools import partial
from typing import Any
import torch
from optree import tree_map
from tensordict import TensorClass

from posteriors.types import (
    TensorTree,
    Transform,
    ForwardFn,
    OuterLogProbFn,
)
from posteriors.tree_utils import flexi_tree_map, tree_insert_
from posteriors.utils import (
    diag_normal_sample,
    diag_ggn,
    is_scalar,
    CatchAuxError,
)


def build(
    forward: ForwardFn,
    outer_log_likelihood: OuterLogProbFn,
    init_prec_diag: TensorTree | float = 0.0,
) -> Transform:
    """Builds a transform for a diagonal Generalized Gauss-Newton (GGN)
    Laplace approximation.

    Equivalent to the diagonal of the (non-empirical) Fisher information matrix when
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
        init_prec_diag: Initial diagonal precision matrix.
            Can be tree like params or scalar.

    Returns:
        Diagonal GGN Laplace approximation transform instance.
    """
    init_fn = partial(init, init_prec_diag=init_prec_diag)
    update_fn = partial(
        update, forward=forward, outer_log_likelihood=outer_log_likelihood
    )
    return Transform(init_fn, update_fn)


class DiagLaplaceState(TensorClass["frozen"]):
    """State encoding a diagonal Normal distribution over parameters.

    Attributes:
        params: Mean of the Normal distribution.
        prec_diag: Diagonal of the precision matrix of the Normal distribution.
        step: Current step count.
    """

    params: TensorTree
    prec_diag: TensorTree
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(
    params: TensorTree,
    init_prec_diag: TensorTree | float = 0.0,
) -> DiagLaplaceState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Mean of the Normal distribution.
        init_prec_diag: Initial diagonal precision matrix.
            Can be tree like params or scalar.

    Returns:
        Initial DiagLaplaceState.
    """
    if is_scalar(init_prec_diag):
        init_prec_diag = tree_map(
            lambda x: torch.full_like(x, init_prec_diag, requires_grad=x.requires_grad),
            params,
        )

    return DiagLaplaceState(params, init_prec_diag)


def update(
    state: DiagLaplaceState,
    batch: Any,
    forward: ForwardFn,
    outer_log_likelihood: OuterLogProbFn,
    inplace: bool = False,
) -> tuple[DiagLaplaceState, TensorTree]:
    """Adds diagonal GGN matrix of covariance summed over given batch.

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
        Updated DiagLaplaceState and auxiliary information.
    """

    def outer_loss(z, batch):
        return -outer_log_likelihood(z, batch)

    with torch.no_grad(), CatchAuxError():
        diag_ggn_batch, aux = diag_ggn(
            lambda params: forward(params, batch),
            lambda z: outer_loss(z, batch),
            forward_has_aux=True,
            loss_has_aux=False,
            normalize=False,
        )(state.params)

    def update_func(x, y):
        return x + y

    prec_diag = flexi_tree_map(
        update_func, state.prec_diag, diag_ggn_batch, inplace=inplace
    )

    if inplace:
        tree_insert_(state.step, state.step + 1)
        return state, aux
    return DiagLaplaceState(state.params, prec_diag, state.step + 1), aux


def sample(
    state: DiagLaplaceState, sample_shape: torch.Size = torch.Size([])
) -> TensorTree:
    """Sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and diagonal precision.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.prec_diag)
    return diag_normal_sample(state.params, sd_diag, sample_shape=sample_shape)
