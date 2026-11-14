from importlib import import_module

import pytest
import torch
import torchopt
from tensordict import TensorClass, NonTensorData

from posteriors.tree_utils import tree_insert_


class TransformState(TensorClass["frozen"]):
    params: torch.Tensor
    aux: NonTensorData


def test_TransformState():
    s = TransformState(params=torch.ones(3), aux=None)

    params_new = torch.ones(3) * 2
    aux_new = ["dsadsa", torch.ones(3)]

    def update_state(state, params_new, aux_new):
        tree_insert_(state.params, params_new)
        # return state
        return state.replace(aux=NonTensorData(aux_new))

    state_new = update_state(s, params_new, aux_new)

    assert torch.allclose(state_new.params, params_new)
    assert state_new.aux == aux_new

    # Check params updated in-place
    assert torch.allclose(s.params, params_new)

    # aux never updated in-place as it is not guaranteed to be a TensorTree
    assert s.aux is None

    # Check works with vmap
    def init(params, aux=None):
        return TransformState(params=params, aux=aux)

    multi_params_new = torch.ones(5, 3)
    multi_aux_new = torch.ones(5, 3)

    multi_state_new = torch.vmap(init)(multi_params_new, multi_aux_new)

    assert torch.allclose(multi_state_new.params, multi_params_new)
    assert torch.allclose(multi_state_new.aux, multi_aux_new)


STATE_DEFAULTS = [
    ("ekf.diag_fisher", "log_likelihood"),
    ("ekf.diag_fisher", "step"),
    ("ekf.dense_fisher", "log_likelihood"),
    ("ekf.dense_fisher", "step"),
    ("laplace.diag_fisher", "step"),
    ("laplace.dense_fisher", "step"),
    ("laplace.diag_ggn", "step"),
    ("laplace.dense_ggn", "step"),
    ("laplace.dense_hessian", "step"),
    ("sgmcmc.sgld", "log_posterior"),
    ("sgmcmc.sgld", "step"),
    ("sgmcmc.sglrw", "log_posterior"),
    ("sgmcmc.sglrw", "step"),
    ("sgmcmc.sghmc", "log_posterior"),
    ("sgmcmc.sghmc", "step"),
    ("sgmcmc.sgnht", "log_posterior"),
    ("sgmcmc.sgnht", "step"),
    ("sgmcmc.baoa", "log_posterior"),
    ("sgmcmc.baoa", "step"),
    ("vi.diag", "nelbo"),
    ("vi.diag", "step"),
    ("vi.dense", "nelbo"),
    ("vi.dense", "step"),
    ("optim", "loss"),
    ("torchopt", "loss"),
]


@pytest.mark.parametrize("module_name,field_name", STATE_DEFAULTS)
def test_state_tensor_defaults_are_independent(module_name, field_name):
    module = import_module(f"posteriors.{module_name}")

    def init():
        params = torch.ones(2, requires_grad=True)
        if module_name == "optim":
            return module.init(params, torch.optim.SGD, lr=0.1)
        if module_name.startswith("vi.") or module_name == "torchopt":
            return module.init(params, torchopt.sgd(lr=0.1))
        return module.init(params)

    first, second = init(), init()
    original = getattr(first, field_name).clone()
    try:
        tree_insert_(getattr(first, field_name), torch.tensor(7.0))
        torch.testing.assert_close(
            getattr(second, field_name), original, equal_nan=True
        )
        torch.testing.assert_close(
            getattr(init(), field_name), original, equal_nan=True
        )
    finally:
        # Restore shared defaults when this regression is run on the original code.
        tree_insert_(getattr(first, field_name), original)


def test_sgld_inplace_update_does_not_advance_other_chains():
    from posteriors.sgmcmc import sgld

    learning_rate_steps = []

    def learning_rate(step):
        learning_rate_steps.append(step.item())
        return 0.1 / (step + 1)

    def log_posterior(params, batch):
        return -0.5 * params.square().sum(), torch.empty(0)

    transform = sgld.build(log_posterior, lr=learning_rate, temperature=0.0)
    first = transform.init(torch.ones(2))
    second = transform.init(torch.ones(2))
    original_step = first.step.clone()
    original_log_posterior = first.log_posterior.clone()
    try:
        transform.update(first, None, inplace=True)
        assert first.step.item() == 1
        assert second.step.item() == 0
        assert second.log_posterior.isnan()
        third = transform.init(torch.ones(2))
        assert third.step.item() == 0
        transform.update(second, None, inplace=True)
        assert learning_rate_steps == [0, 0]
        torch.testing.assert_close(first.params, second.params)
    finally:
        tree_insert_(first.step, original_step)
        tree_insert_(first.log_posterior, original_log_posterior)
