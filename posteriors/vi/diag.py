from dataclasses import field
from typing import Callable, Any, Tuple
from functools import partial
import torch
from torch.func import grad_and_value, vmap
from optree import tree_map
import torchopt
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.tree_utils import tree_insert_
from posteriors.utils import (
    diag_normal_log_prob,
    diag_normal_sample,
    is_scalar,
    CatchAuxError,
)


def build(
    log_posterior: Callable[[TensorTree, Any], float],
    optimizer: torchopt.base.GradientTransformation,
    temperature: float | Schedule = 1.0,
    n_samples: int = 1,
    stl: bool = True,
    init_log_sds: TensorTree | float = 0.0,
) -> Transform:
    """Builds a transform for variational inference with a diagonal Normal
    distribution over parameters.

    Find $\\mu$ and diagonal $\\Sigma$ that mimimize $\\text{KL}(N(θ| \\mu, \\Sigma) || p_T(θ))$
    where $p_T(θ) \\propto \\exp( \\log p(θ) / T)$ with temperature $T$.

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data.

    For more information on variational inference see [Blei et al, 2017](https://arxiv.org/abs/1601.00670).

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        optimizer: TorchOpt functional optimizer for updating the variational
            parameters. Make sure to use lower case like torchopt.adam()
        temperature: Temperature to rescale (divide) log_posterior.
            Scalar or schedule (callable taking step index, returning scalar).
        n_samples: Number of samples to use for Monte Carlo estimate.
        stl: Whether to use the stick-the-landing estimator
            from [Roeder et al](https://arxiv.org/abs/1703.09194).
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Can be a tree matching params or scalar.

    Returns:
        Diagonal VI transform instance.
    """
    init_fn = partial(init, optimizer=optimizer, init_log_sds=init_log_sds)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        optimizer=optimizer,
        temperature=temperature,
        n_samples=n_samples,
        stl=stl,
    )
    return Transform(init_fn, update_fn)


class VIDiagState(TensorClass["frozen"]):
    """State encoding a diagonal Normal variational distribution over parameters.

    Attributes:
        params: Mean of the variational distribution.
        log_sd_diag: Log of the square-root diagonal of the covariance matrix of the
            variational distribution.
        opt_state: TorchOpt state storing optimizer data for updating the
            variational parameters.
        nelbo: Negative evidence lower bound (lower is better).
        step: Current step count.
    """

    params: TensorTree
    log_sd_diag: TensorTree
    opt_state: torchopt.typing.OptState
    nelbo: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    step: torch.Tensor = field(default_factory=lambda: torch.tensor(0))


def init(
    params: TensorTree,
    optimizer: torchopt.base.GradientTransformation,
    init_log_sds: TensorTree | float = 0.0,
) -> VIDiagState:
    """Initialise diagonal Normal variational distribution over parameters.

    optimizer.init will be called on flattened variational parameters so hyperparameters
    such as learning rate need to pre-specified through TorchOpt's functional API:

    ```
    import torchopt

    optimizer = torchopt.adam(lr=1e-2)
    vi_state = init(init_mean, optimizer)
    ```

    It's assumed maximize=False for the optimizer, so that we minimize the NELBO.

    Args:
        params: Initial mean of the variational distribution.
        optimizer: TorchOpt functional optimizer for updating the variational
            parameters. Make sure to use lower case like torchopt.adam()
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Can be a tree matching params or scalar.

    Returns:
        Initial DiagVIState.
    """
    if is_scalar(init_log_sds):
        init_log_sds = tree_map(
            lambda x: torch.full_like(x, init_log_sds, requires_grad=x.requires_grad),
            params,
        )

    opt_state = optimizer.init([params, init_log_sds])
    return VIDiagState(params, init_log_sds, opt_state)


def update(
    state: VIDiagState,
    batch: Any,
    log_posterior: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
    temperature: float = 1.0,
    n_samples: int = 1,
    stl: bool = True,
    inplace: bool = False,
) -> tuple[VIDiagState, TensorTree]:
    """Updates the variational parameters to minimize the NELBO.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        optimizer: TorchOpt functional optimizer for updating the variational
            parameters. Make sure to use lower case like torchopt.adam()
        temperature: Temperature to rescale (divide) log_posterior.
            Scalar or schedule (callable taking step index, returning scalar).
        n_samples: Number of samples to use for Monte Carlo estimate.
        stl: Whether to use the stick-the-landing estimator
            from (Roeder et al](https://arxiv.org/abs/1703.09194).
        inplace: Whether to modify state in place.

    Returns:
        Updated DiagVIState and auxiliary information.
    """
    temperature = temperature(state.step) if callable(temperature) else temperature

    def nelbo_log_sd(m, lsd):
        sd_diag = tree_map(torch.exp, lsd)
        return nelbo(m, sd_diag, batch, log_posterior, temperature, n_samples, stl)

    with torch.no_grad(), CatchAuxError():
        nelbo_grads, (nelbo_val, aux) = grad_and_value(
            nelbo_log_sd, argnums=(0, 1), has_aux=True
        )(state.params, state.log_sd_diag)

    updates, opt_state = optimizer.update(
        nelbo_grads,
        state.opt_state,
        params=[state.params, state.log_sd_diag],
        inplace=inplace,
    )
    mean, log_sd_diag = torchopt.apply_updates(
        (state.params, state.log_sd_diag), updates, inplace=inplace
    )

    if inplace:
        tree_insert_(state.nelbo, nelbo_val.detach())
        tree_insert_(state.step, state.step + 1)
        return state, aux

    return VIDiagState(
        mean, log_sd_diag, opt_state, nelbo_val.detach(), state.step + 1
    ), aux


def nelbo(
    mean: TensorTree,
    sd_diag: TensorTree,
    batch: Any,
    log_posterior: LogProbFn,
    temperature: float = 1.0,
    n_samples: int = 1,
    stl: bool = True,
) -> Tuple[float, Any]:
    """Returns the negative evidence lower bound (NELBO) for a diagonal Normal
    variational distribution over the parameters of a model.

    Monte Carlo estimate with `n_samples` from q.
    $$
    \\text{NELBO} = - 𝔼_{q(θ)}[\\log p(y|x, θ) + \\log p(θ) - \\log q(θ) * T])
    $$
    for temperature $T$.

    `log_posterior` expects to take parameters and input batch and return a scalar
    as well as a TensorTree of any auxiliary information:

    ```
    log_posterior_eval, aux = log_posterior(params, batch)
    ```

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data and variable batch size.

    Args:
        mean: Mean of the variational distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            variational distribution.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        temperature: Temperature to rescale (divide) log_posterior.
        n_samples: Number of samples to use for Monte Carlo estimate.
        stl: Whether to use the stick-the-landing estimator
            from (Roeder et al](https://arxiv.org/abs/1703.09194).

    Returns:
        The sampled approximate NELBO averaged over the batch.
    """
    sampled_params = diag_normal_sample(mean, sd_diag, sample_shape=(n_samples,))
    if stl:
        mean = tree_map(lambda x: x.detach(), mean)
        sd_diag = tree_map(lambda x: x.detach(), sd_diag)

    # Don't use vmap for single sample, since vmap doesn't work with lots of models
    if n_samples == 1:
        single_param = tree_map(lambda x: x[0], sampled_params)
        log_p, aux = log_posterior(single_param, batch)
        log_q = diag_normal_log_prob(single_param, mean, sd_diag)
    else:
        log_p, aux = vmap(log_posterior, (0, None), (0, 0))(sampled_params, batch)
        log_q = vmap(diag_normal_log_prob, (0, None, None))(
            sampled_params, mean, sd_diag
        )
    return -(log_p - log_q * temperature).mean(), aux


def sample(state: VIDiagState, sample_shape: torch.Size = torch.Size([])) -> TensorTree:
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and log standard deviations.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from Normal distribution.
    """
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    return diag_normal_sample(state.params, sd_diag, sample_shape=sample_shape)
