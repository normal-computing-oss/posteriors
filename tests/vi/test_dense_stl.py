import pytest
import torch
from optree import tree_map
from optree.integrations.torch import tree_ravel
from torch.func import grad

from posteriors import vi
from posteriors.utils import L_to_flat


@pytest.mark.parametrize("n_samples", [1, 3, 10])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("pytree", [False, True])
@pytest.mark.parametrize("stl", [False, True])
def test_dense_stl_gradient_at_exact_target(n_samples, seed, pytree, stl):
    mean = torch.tensor([1.0, -2.0])
    scale = torch.tensor([[1.5, 0.0], [0.5, 2.0]])
    target = torch.distributions.MultivariateNormal(mean, scale_tril=scale)
    if pytree:
        mean = {"a": mean[:1], "b": mean[1:]}

    def log_posterior(params, batch):
        flat, _ = tree_ravel(params)
        return target.log_prob(flat), torch.empty(0)

    def objective(params, factor):
        return vi.dense.nelbo(
            params, factor, None, log_posterior, n_samples=n_samples, stl=stl
        )[0]

    torch.manual_seed(seed)
    mean_grad, factor_grad = grad(objective, argnums=(0, 1))(mean, L_to_flat(scale))
    if stl:
        tree_map(
            lambda x: torch.testing.assert_close(x, torch.zeros_like(x)), mean_grad
        )
        torch.testing.assert_close(factor_grad, torch.zeros_like(factor_grad))
    else:
        torch.manual_seed(seed)
        samples = target.rsample((n_samples,))
        expected = torch.linalg.solve(
            target.covariance_matrix, (samples - target.mean).mean(0)
        )
        flat_grad, _ = tree_ravel(mean_grad)
        torch.testing.assert_close(flat_grad, expected)
