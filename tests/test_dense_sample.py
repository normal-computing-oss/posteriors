import pytest
import torch
import torchopt
from optree import tree_flatten
from optree.integrations.torch import tree_ravel

from posteriors import ekf, laplace, vi


@pytest.mark.parametrize(
    "module",
    [
        laplace.dense_fisher,
        laplace.dense_ggn,
        laplace.dense_hessian,
        ekf.dense_fisher,
        vi.dense,
    ],
    ids=["laplace_fisher", "laplace_ggn", "laplace_hessian", "ekf", "vi"],
)
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize(
    "params",
    [
        torch.arange(3.0),
        {"weight": torch.arange(6.0).reshape(2, 3)},
        {
            "weight": torch.arange(6.0).reshape(2, 3),
            "bias": torch.ones(2),
            "scale": torch.tensor(1.0),
        },
    ],
    ids=["vector", "matrix", "mixed"],
)
def test_dense_sample_shapes_and_values(module, sample_shape, params):
    mean_flat, _ = tree_ravel(params)
    covariance = torch.eye(mean_flat.numel())
    if module is vi.dense:
        state = module.init(params, torchopt.sgd(lr=0.1), init_L=covariance)
    elif module is ekf.dense_fisher:
        state = module.init(params, init_cov=covariance)
    else:
        state = module.init(params, init_prec=covariance)

    torch.manual_seed(42)
    samples = module.sample(state, sample_shape)
    sample_leaves, sample_spec = tree_flatten(samples)
    param_leaves, param_spec = tree_flatten(params)
    assert sample_spec == param_spec
    for sample, param in zip(sample_leaves, param_leaves):
        assert sample.shape == sample_shape + param.shape

    samples_flat = torch.cat(
        [
            sample.reshape(sample_shape + (param.numel(),))
            for sample, param in zip(sample_leaves, param_leaves)
        ],
        dim=-1,
    )
    torch.manual_seed(42)
    expected = torch.distributions.MultivariateNormal(
        mean_flat, covariance_matrix=covariance
    ).sample(sample_shape)
    torch.testing.assert_close(samples_flat, expected)


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_dense_vi_sample_gradients(sample_shape):
    params = {
        "weight": torch.zeros(2, 3, requires_grad=True),
        "bias": torch.tensor(0.0, requires_grad=True),
    }
    state = vi.dense.init(params, torchopt.sgd(lr=0.1))
    samples = vi.dense.sample(state, sample_shape)
    total = sum(leaf.sum() for leaf in tree_flatten(samples)[0])
    gradients = torch.autograd.grad(
        total, (params["weight"], params["bias"], state.L_factor)
    )
    n_samples = torch.Size(sample_shape).numel()
    torch.testing.assert_close(
        gradients[0], torch.full_like(params["weight"], n_samples)
    )
    torch.testing.assert_close(gradients[1], torch.full_like(params["bias"], n_samples))
    assert torch.isfinite(gradients[2]).all()
