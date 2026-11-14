from dataclasses import field
from functools import partial
import torch
import torchopt
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn
from posteriors.utils import CatchAuxError
from posteriors.tree_utils import tree_insert_


def build(
    loss_fn: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
) -> Transform:
    """Build a [TorchOpt](https://github.com/metaopt/torchopt) optimizer transformation.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    ```
    transform = build(loss_fn, torchopt.adam(lr=0.1))
    state = transform.init(params)

    for batch in dataloader:
        state, aux = transform.update(state, batch)
    ```

    Args:
        loss_fn: Loss function.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case e.g. torchopt.adam()

    Returns:
        Torchopt optimizer transform instance.
    """
    init_fn = partial(init, optimizer=optimizer)
    update_fn = partial(update, optimizer=optimizer, loss_fn=loss_fn)
    return Transform(init_fn, update_fn)


class TorchOptState(TensorClass["frozen"]):
    """State of a [TorchOpt](https://github.com/metaopt/torchopt) optimizer.

    Contains the parameters, the optimizer state for the TorchOpt optimizer,
    loss value, and auxiliary information.

    Attributes:
        params: Parameters to be optimized.
        opt_state: TorchOpt optimizer state.
        loss: Loss value.
    """

    params: TensorTree
    opt_state: torchopt.typing.OptState
    loss: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


def init(
    params: TensorTree,
    optimizer: torchopt.base.GradientTransformation,
) -> TorchOptState:
    """Initialise a [TorchOpt](https://github.com/metaopt/torchopt) optimizer.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    Args:
        params: Parameters to be optimized.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case e.g. torchopt.adam()

    Returns:
        Initial TorchOptState.
    """
    opt_state = optimizer.init(params)
    return TorchOptState(params, opt_state)


def update(
    state: TorchOptState,
    batch: TensorTree,
    loss_fn: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
    inplace: bool = False,
) -> tuple[TorchOptState, TensorTree]:
    """Update the [TorchOpt](https://github.com/metaopt/torchopt) optimizer state.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    Args:
        state: Current state.
        batch: Batch of data.
        loss_fn: Loss function.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case like torchopt.adam()
        inplace: Whether to update the state in place.

    Returns:
        Updated TorchOptState and auxiliary information.
    """
    params = state.params
    opt_state = state.opt_state
    with torch.no_grad(), CatchAuxError():
        grads, (loss, aux) = torch.func.grad_and_value(loss_fn, has_aux=True)(
            params, batch
        )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates, inplace=inplace)
    if inplace:
        tree_insert_(state.loss, loss.detach())
        return state, aux

    return TorchOptState(params, opt_state, loss), aux
