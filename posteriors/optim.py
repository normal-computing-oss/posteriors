from dataclasses import field
from typing import Type, Any
from functools import partial
import torch
from tensordict import TensorClass
from optree import tree_leaves

from posteriors.types import TensorTree, Transform, LogProbFn
from posteriors.utils import CatchAuxError
from posteriors.tree_utils import tree_insert_


def build(
    loss_fn: LogProbFn,
    optimizer: Type[torch.optim.Optimizer],
    **kwargs: Any,
) -> Transform:
    """Builds an optimizer transform from [torch.optim](https://pytorch.org/docs/stable/optim.html)

    ```
    transform = build(loss_fn, torch.optim.Adam, lr=0.1)
    state = transform.init(params)

    for batch in dataloader:
        state, aux = transform.update(state, batch)
    ```

    Args:
        loss_fn: Function that takes the parameters and returns the loss.
            of the form `loss, aux = fn(params, batch)`.
        optimizer: Optimizer class from torch.optim.
        **kwargs: Keyword arguments to pass to the optimizer class.

    Returns:
        `torch.optim` transform instance.
    """
    init_fn = partial(init, optimizer_cls=optimizer, **kwargs)
    update_fn = partial(update, loss_fn=loss_fn)
    return Transform(init_fn, update_fn)


class OptimState(TensorClass["frozen"]):
    """State of an optimizer from [torch.optim](https://pytorch.org/docs/stable/optim.html).

    Attributes:
        params: Parameters to be optimized.
        optimizer: torch.optim optimizer instance.
        loss: Loss value.
    """

    params: TensorTree
    optimizer: torch.optim.Optimizer
    loss: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


def init(
    params: TensorTree,
    optimizer_cls: Type[torch.optim.Optimizer],
    *args: Any,
    **kwargs: Any,
) -> OptimState:
    """Initialise a [torch.optim](https://pytorch.org/docs/stable/optim.html) optimizer
    state.

    Args:
        params: Parameters to be optimized.
        optimizer_cls: Optimizer class from torch.optim.
        *args: Positional arguments to pass to the optimizer class.
        **kwargs: Keyword arguments to pass to the optimizer class.

    Returns:
        Initial OptimState.
    """
    optimizer = optimizer_cls(tree_leaves(params), *args, **kwargs)
    return OptimState(params, optimizer)


def update(
    state: OptimState,
    batch: TensorTree,
    loss_fn: LogProbFn,
    inplace: bool = True,
) -> tuple[OptimState, TensorTree]:
    """Perform a single update step of a [torch.optim](https://pytorch.org/docs/stable/optim.html)
    optimizer.

    Args:
        state: Current optimizer state.
        batch: Input data to loss_fn.
        loss_fn: Function that takes the parameters and returns the loss.
            of the form `loss, aux = fn(params, batch)`.
        inplace: Whether to update the parameters in place.
            inplace=False not supported for posteriors.optim

    Returns:
        Updated OptimState and auxiliary information.
    """
    if not inplace:
        raise NotImplementedError("inplace=False not supported for posteriors.optim")
    state.optimizer.zero_grad()
    with CatchAuxError():
        loss, aux = loss_fn(state.params, batch)
    loss.backward()
    state.optimizer.step()
    tree_insert_(state.loss, loss.detach())
    return state, aux
