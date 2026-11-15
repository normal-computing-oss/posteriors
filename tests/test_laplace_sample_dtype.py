import pytest
import torch

from posteriors import laplace


@pytest.mark.parametrize(
    "module", [laplace.dense_fisher, laplace.dense_ggn, laplace.dense_hessian]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_params", [1, 3])
@pytest.mark.parametrize("n_samples", [1, 5])
def test_dense_laplace_sample_dtype(module, dtype, n_params, n_samples):
    mean = torch.arange(n_params, dtype=dtype)
    precision = 2 * torch.eye(n_params, dtype=dtype)
    state = module.init(mean, init_prec=precision)

    torch.manual_seed(42)
    samples = module.sample(state, (n_samples,))
    torch.manual_seed(42)
    expected = torch.distributions.MultivariateNormal(
        mean, precision_matrix=precision
    ).sample((n_samples,))
    assert samples.dtype == dtype
    assert samples.device == precision.device
    torch.testing.assert_close(samples, expected)
