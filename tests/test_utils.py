import pytest
from functools import partial
import torch
from optree import tree_map, tree_flatten, tree_reduce
from optree.integrations.torch import tree_ravel

from posteriors import (
    CatchAuxError,
    model_to_function,
    linearized_forward_diag,
    hvp,
    fvp,
    empirical_fisher,
    ggnvp,
    ggn,
    diag_ggn,
    cg,
    diag_normal_log_prob,
    diag_normal_sample,
    per_samplify,
    is_scalar,
    L_from_flat,
    L_to_flat,
    cumulative_mean_and_cov,
)
from posteriors.utils import NO_AUX_ERROR_MSG, NON_TENSOR_AUX_ERROR_MSG
from tests.scenarios import TestModel, TestLanguageModel


def test_CatchAuxError():
    def func(x):
        return x**2

    def func_aux_none(x):
        return x**2, None

    def func_aux(x):
        return x**2, torch.tensor([])

    # Check NO_AUX_ERROR_MSG is correct
    try:
        torch.func.grad(func, has_aux=True)(torch.tensor(1.0))
    except Exception as e:
        assert NO_AUX_ERROR_MSG in str(e)

    # Check NON_TENSOR_AUX_ERROR_MSG is correct
    try:
        torch.func.grad(func_aux_none, has_aux=True)(torch.tensor(1.0))
    except Exception as e:
        assert NON_TENSOR_AUX_ERROR_MSG in str(e)

    # Check CatchAuxError works for NO_AUX_ERROR_MSG
    with pytest.raises(RuntimeError) as e:
        with CatchAuxError():
            torch.func.grad(func, has_aux=True)(torch.tensor(1.0))

    assert "Auxiliary output not found" in str(e)

    with pytest.raises(RuntimeError) as e:
        with torch.no_grad(), CatchAuxError():
            torch.func.grad(func, has_aux=True)(torch.tensor(1.0))

    assert "Auxiliary output not found" in str(e)

    # Check CatchAuxError works for NON_TENSOR_AUX_ERROR_MSG
    with pytest.raises(RuntimeError) as e:
        with CatchAuxError():
            torch.func.grad(func_aux_none, has_aux=True)(torch.tensor(1.0))

    assert "Auxiliary output should be a TensorTree" in str(e)

    with pytest.raises(RuntimeError) as e:
        with torch.no_grad(), CatchAuxError():
            torch.func.grad(func_aux_none, has_aux=True)(torch.tensor(1.0))

    assert "Auxiliary output should be a TensorTree" in str(e)

    # Check CatchAuxError works for correct func_aux
    with CatchAuxError():
        torch.func.grad(func_aux, has_aux=True)(torch.tensor(1.0))

    with torch.no_grad(), CatchAuxError():
        torch.func.grad(func_aux, has_aux=True)(torch.tensor(1.0))


def test_model_to_function():
    model = TestModel()

    xs = torch.randn(100, 10)

    func_model = model_to_function(model)

    output = model(xs)
    func_output = func_model(dict(model.named_parameters()), xs)

    assert torch.allclose(output, func_output)

    lm = TestLanguageModel()

    input_ids = torch.randint(1000, (10, 10))
    attention_mask = torch.ones_like(input_ids)

    func_lm = model_to_function(lm)

    output = lm(input_ids, attention_mask)

    func_output1 = func_lm(
        dict(lm.named_parameters()), input_ids=input_ids, attention_mask=attention_mask
    )

    func_output2 = func_lm(dict(lm.named_parameters()), input_ids, attention_mask)

    assert type(output) is type(func_output1)
    assert type(output) is type(func_output2)
    assert torch.allclose(output["logits"], func_output1["logits"])
    assert torch.allclose(output["logits"], func_output2["logits"])


def test_linearized_forward_diag():
    vocab_size = 50
    batch_size = 3
    lm = TestLanguageModel(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)

    lm_functional = model_to_function(lm)

    input_ids = torch.randint(vocab_size, (batch_size, batch_size))
    attention_mask = torch.ones_like(input_ids)
    params = dict(lm.named_parameters())

    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_func(p, batch):
        output = lm_functional(p, **batch)
        next_token_logits = output["logits"][:, -1, :]
        return next_token_logits, output

    sd_diag = tree_map(lambda x: torch.randn_like(x).abs(), params)

    mean, lin_cov_chol, output = linearized_forward_diag(
        forward_func, params, batch, sd_diag
    )

    assert mean.shape == (batch_size, vocab_size)
    assert lin_cov_chol.shape == (batch_size, vocab_size, vocab_size)
    assert torch.allclose(output["logits"][:, -1, :], mean)

    lin_cov = lin_cov_chol @ lin_cov_chol.transpose(-1, -2)

    jac = torch.func.jacrev(forward_func, has_aux=True)(params, batch)[0]
    jac_flat = tree_flatten(jac)[0]
    jac_flat = torch.cat([x.flatten(start_dim=2) for x in jac_flat], dim=2)

    sd_diag_flat = tree_flatten(sd_diag)[0]
    sd_diag_flat = torch.cat([x.flatten() for x in sd_diag_flat])

    for i in range(batch_size):
        jac_i = jac_flat[i]
        delta = jac_i @ torch.diag(sd_diag_flat**2) @ jac_i.T
        assert torch.allclose(lin_cov[i], delta, atol=1e-5)


def test_hvp():
    def func(x):
        return (x**5).sum()

    x = torch.arange(1.0, 6.0)
    v = torch.ones_like(x)
    hvp_result = hvp(func, (x,), (v,))
    expected = torch.func.hessian(func)(x) @ v
    assert torch.allclose(hvp_result[0], torch.func.grad(func)(x))
    assert torch.allclose(hvp_result[1], expected)

    def func_and_aux(x):
        return (x**5).sum(), x**2

    hvp_result_aux = hvp(func_and_aux, (x,), (v,), has_aux=True)

    assert torch.allclose(hvp_result_aux[0], torch.func.grad(func)(x))
    assert torch.allclose(hvp_result_aux[1], expected)
    assert torch.allclose(hvp_result_aux[2], x**2)


def test_fvp():
    def func(x):
        return torch.stack([(x**5).sum(), (x**3).sum()])

    x = torch.arange(1.0, 6.0)
    v = torch.ones_like(x)

    # Test not normalized
    output, fvp_result = fvp(func, (x,), (v,))

    jac = torch.func.jacrev(func)(x)
    fisher = jac.T @ jac
    expected = fisher @ v
    assert torch.allclose(fvp_result, expected)
    assert torch.allclose(output, func(x))

    # Test normalize
    output_norm, fvp_result_norm = fvp(func, (x,), (v,), normalize=True)
    assert torch.allclose(fvp_result_norm, expected / 2)
    assert torch.allclose(output_norm, func(x))

    # Test aux, not normalized
    def func_aux(x):
        return torch.stack([(x**5).sum(), (x**3).sum()]), x

    output_aux, fvp_aux_result, aux = fvp(
        func_aux, (x,), (v,), has_aux=True, normalize=False
    )
    assert torch.allclose(fvp_aux_result, expected)
    assert torch.allclose(output_aux, func(x))
    assert torch.allclose(aux, x)

    # Test aux, normalized
    output_aux_norm, fvp_aux_result_norm, aux_norm = fvp(
        func_aux, (x,), (v,), has_aux=True, normalize=True
    )
    assert torch.allclose(fvp_aux_result_norm, expected / 2)
    assert torch.allclose(output_aux_norm, func(x))
    assert torch.allclose(aux_norm, x)

    # Test model
    model = TestModel()
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3,)).unsqueeze(-1)
    batch_spec = {"inputs": batch_inputs, "labels": batch_labels}

    def log_likelihood(params, batch):
        output = torch.func.functional_call(model, params, batch["inputs"])
        return -torch.nn.BCEWithLogitsLoss()(output, batch["labels"].float())

    log_likelihood_per_sample = per_samplify(log_likelihood)

    v = tree_map(lambda x: torch.randn_like(x), params)

    fvp_result = fvp(
        partial(log_likelihood_per_sample, batch=batch_spec),
        (params,),
        (v,),
    )
    assert len(fvp_result) == 2
    assert torch.allclose(fvp_result[0], log_likelihood_per_sample(params, batch_spec))


def test_empirical_fisher():
    # Test no aux
    def f(params, batch):
        return torch.squeeze(batch[1] - batch[0] @ params["weights"] - params["bias"])

    f_per_sample = torch.vmap(f, in_dims=(None, 0))

    num_samples = 2
    num_features = 4

    x = torch.randn(num_samples, num_features)
    y = torch.randn(
        num_samples,
    )
    params = {
        "weights": torch.randn(num_features, 1, requires_grad=True),
        "bias": torch.randn(1, requires_grad=True),
    }

    # Test not normalized
    batch = (x, y)
    fisher = empirical_fisher(lambda p: f_per_sample(p, batch))(params)
    expected_fisher = torch.zeros((num_features + 1, num_features + 1))
    for xs, ys in zip(x, y):
        g = torch.func.grad(f)(params, (xs, ys))
        g = tree_ravel(g)[0]
        expected_fisher += torch.outer(g, g)

    assert torch.allclose(fisher, expected_fisher, rtol=1e-5)

    # Test normalized
    fisher_norm = empirical_fisher(lambda p: f_per_sample(p, batch), normalize=True)(
        params
    )
    assert torch.allclose(fisher_norm, expected_fisher / num_samples, rtol=1e-5)

    # Test aux, not normalized
    def f_aux(params, batch):
        return f(params, batch), params

    f_aux_per_sample = torch.vmap(f_aux, in_dims=(None, 0))

    fisher_aux, _ = empirical_fisher(
        lambda p: f_aux_per_sample(p, batch), has_aux=True, normalize=False
    )(params)
    assert torch.allclose(fisher_aux, expected_fisher, rtol=1e-5)

    # Test aux, normalized
    fisher_aux_norm, _ = empirical_fisher(
        lambda p: f_aux_per_sample(p, batch), has_aux=True, normalize=True
    )(params)
    assert torch.allclose(fisher_aux_norm, expected_fisher / num_samples, rtol=1e-5)

    # Test matches fvp
    v = tree_map(lambda x: torch.randn_like(x), params)
    fvp_result = fvp(
        lambda p: f_per_sample(p, batch), (params,), (v,), normalize=False
    )[1]
    fvp_result = tree_ravel(fvp_result)[0]
    fisher_fvp = fisher @ tree_ravel(v)[0]
    assert torch.allclose(fvp_result, fisher_fvp, rtol=1e-5)

    # Test model
    model = TestModel()
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3,)).unsqueeze(-1)
    batch = {"inputs": batch_inputs, "labels": batch_labels}

    def log_likelihood(params, batch):
        output = torch.func.functional_call(model, params, batch["inputs"])
        return -torch.nn.BCEWithLogitsLoss()(output, batch["labels"].float())

    log_likelihood_per_sample = per_samplify(log_likelihood)

    ef_result = empirical_fisher(log_likelihood_per_sample)(params, batch)

    flat_params, unravel_params = tree_ravel(params)

    def log_likelihood_flat(flat_p, batch):
        return log_likelihood_per_sample(unravel_params(flat_p), batch)

    jac = torch.func.jacrev(log_likelihood_flat)(flat_params, batch)
    expected = jac.T @ jac
    assert torch.allclose(ef_result, expected, rtol=1e-5)


def test_ggnvp():
    # Batchsize=2, dz=5
    def forward(x):
        zs1 = torch.vmap(lambda a: a ** torch.arange(1, 6))(x).sum(0)
        zs2 = torch.vmap(lambda a: (a / 2) ** torch.arange(1, 6))(x).sum(0)

        return torch.stack([zs1, zs2])

    def loss(z):
        return torch.nn.functional.log_softmax(z, dim=0)[..., 0].sum()

    x = torch.randn(10)
    v = torch.randn(10)

    z = forward(x)
    Jac = torch.func.jacrev(forward)(x).flatten(end_dim=-2)
    Hes = torch.func.hessian(loss)(z).flatten(start_dim=2).flatten(end_dim=1)
    expected = Jac.T @ Hes @ Jac

    # Test unnormalised
    ggnvp_result = ggnvp(forward, loss, (x,), (v,))
    assert torch.allclose(ggnvp_result[1], expected @ v, rtol=1e-5)
    assert len(ggnvp_result) == 2
    assert len(ggnvp_result[0]) == 2
    assert torch.allclose(ggnvp_result[0][0], z)
    assert torch.allclose(ggnvp_result[0][1], torch.func.grad(loss)(z))

    # Test normalised
    ggnvp_result_norm = ggnvp(forward, loss, (x,), (v,), normalize=True)
    assert torch.allclose(ggnvp_result_norm[1], expected @ v / 2, rtol=1e-5)
    assert len(ggnvp_result_norm) == 2
    assert len(ggnvp_result_norm[0]) == 2
    assert torch.allclose(ggnvp_result_norm[0][0], z)
    assert torch.allclose(ggnvp_result_norm[0][1], torch.func.grad(loss)(z))

    # Test forward aux
    def forward_aux(x):
        return forward(x), x

    ggnvp_result_faux = ggnvp(forward_aux, loss, (x,), (v,), forward_has_aux=True)
    assert torch.allclose(ggnvp_result_faux[1], expected @ v, rtol=1e-5)
    assert len(ggnvp_result_faux) == 3
    assert len(ggnvp_result_faux[0]) == 2
    assert torch.allclose(ggnvp_result_faux[0][0], z)
    assert torch.allclose(ggnvp_result_faux[0][1], torch.func.grad(loss)(z))
    assert torch.allclose(ggnvp_result_faux[2], x)

    # Test loss aux
    def loss_aux(z):
        return loss(z), z

    ggnvp_result_laux = ggnvp(forward, loss_aux, (x,), (v,), loss_has_aux=True)
    assert torch.allclose(ggnvp_result_laux[1], expected @ v, rtol=1e-5)
    assert len(ggnvp_result_laux) == 3
    assert len(ggnvp_result_laux[0]) == 2
    assert torch.allclose(ggnvp_result_laux[0][0], z)
    assert torch.allclose(ggnvp_result_laux[0][1], torch.func.grad(loss)(z))
    assert torch.allclose(ggnvp_result_laux[2], z)

    # Test both aux
    ggnvp_result_flaux = ggnvp(
        forward_aux, loss_aux, (x,), (v,), forward_has_aux=True, loss_has_aux=True
    )
    assert torch.allclose(ggnvp_result_flaux[1], expected @ v, rtol=1e-5)
    assert len(ggnvp_result_flaux) == 4
    assert len(ggnvp_result_flaux[0]) == 2
    assert torch.allclose(ggnvp_result_flaux[0][0], z)
    assert torch.allclose(ggnvp_result_flaux[0][1], torch.func.grad(loss)(z))
    assert torch.allclose(ggnvp_result_faux[2], x)
    assert torch.allclose(ggnvp_result_flaux[3], z)

    # Test model
    model = TestModel()
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3,)).unsqueeze(-1)

    def forward(params, inputs):
        return torch.func.functional_call(model, params, inputs)

    def loss(logits, labels):
        return torch.nn.BCEWithLogitsLoss()(logits, labels.float())

    v = tree_map(lambda x: torch.randn_like(x), params)
    ggnvp_result = ggnvp(
        partial(forward, inputs=batch_inputs),
        partial(loss, labels=batch_labels),
        (params,),
        (v,),
    )
    assert len(ggnvp_result) == 2
    assert torch.allclose(ggnvp_result[0][0], forward(params, batch_inputs))
    assert torch.allclose(
        ggnvp_result[0][1],
        torch.func.grad(partial(loss, labels=batch_labels))(
            forward(params, batch_inputs)
        ),
    )


def test_ggn():
    # Batchsize=2, dz=5
    def forward(x):
        zs1 = torch.vmap(lambda a: a ** torch.arange(1, 6))(x).sum(0)
        zs2 = torch.vmap(lambda a: (a / 2) ** torch.arange(1, 6))(x).sum(0)
        return torch.stack([zs1, zs2])

    def loss(z):
        return torch.nn.functional.log_softmax(z, dim=0)[..., 0].sum()

    x = torch.randn(10)
    v = torch.randn(10)

    z = forward(x)
    Jac = torch.func.jacrev(forward)(x).flatten(end_dim=-2)
    Hes = torch.func.hessian(loss)(z).flatten(start_dim=2).flatten(end_dim=1)
    expected = Jac.T @ Hes @ Jac

    # Test unnormalised
    ggn_result = ggn(forward, loss)(x)
    assert torch.allclose(ggn_result, expected, rtol=1e-5)
    assert torch.allclose(
        ggn_result @ v, ggnvp(forward, loss, (x,), (v,))[1], rtol=1e-5
    )

    # Test normalised
    ggn_result_norm = ggn(forward, loss, normalize=True)(x)
    assert torch.allclose(ggn_result_norm, expected / 2, rtol=1e-5)

    # Test forward aux
    def forward_aux(x):
        return forward(x), x

    ggn_result_faux = ggn(forward_aux, loss, forward_has_aux=True)(x)
    assert torch.allclose(ggn_result_faux[0], expected, rtol=1e-5)
    assert len(ggn_result_faux) == 2
    assert torch.allclose(ggn_result_faux[1], x)

    # Test loss aux
    def loss_aux(z):
        return loss(z), z

    ggn_result_laux = ggn(forward, loss_aux, loss_has_aux=True)(x)
    assert torch.allclose(ggn_result_laux[0], expected, rtol=1e-5)
    assert len(ggn_result_laux) == 2
    assert torch.allclose(ggn_result_laux[1], z)

    # Test both aux
    ggn_result_flaux = ggn(
        forward_aux, loss_aux, forward_has_aux=True, loss_has_aux=True
    )(x)
    assert torch.allclose(ggn_result_flaux[0], expected, rtol=1e-5)
    assert len(ggn_result_flaux) == 3
    assert torch.allclose(ggn_result_flaux[1], x)
    assert torch.allclose(ggn_result_flaux[2], z)

    # Test model
    model = TestModel()
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3,)).unsqueeze(-1)

    def forward(params, inputs):
        return torch.func.functional_call(model, params, inputs)

    def loss(logits, labels):
        return torch.nn.BCEWithLogitsLoss()(logits, labels.float())

    ggn_result = ggn(
        partial(forward, inputs=batch_inputs), partial(loss, labels=batch_labels)
    )(params)

    flat_params, unravel_params = tree_ravel(params)

    def forward_flat(p):
        return forward(unravel_params(p), batch_inputs)[:, 0]

    z = forward_flat(flat_params)
    jac = torch.func.jacrev(forward_flat)(flat_params)
    hess = torch.func.hessian(partial(loss, labels=batch_labels[:, 0]))(z)
    expected = jac.T @ hess @ jac
    assert torch.allclose(ggn_result, expected, rtol=1e-5)


def test_ggndiag():
    # Batchsize=2, dz=5
    def forward(x):
        zs1 = torch.vmap(lambda a: a ** torch.arange(1, 6))(x["a"]).sum(0)
        zs2 = torch.vmap(lambda a: (a / 2) ** torch.arange(1, 6))(x["a"]).sum(0)
        return torch.stack([zs1, zs2])

    def loss(z):
        return torch.nn.functional.log_softmax(z, dim=0)[..., 0].sum()

    x = {"a": torch.randn(10)}

    z = forward(x)
    Jac = torch.func.jacrev(forward)(x)["a"].flatten(end_dim=-2)
    Hes = torch.func.hessian(loss)(z).flatten(start_dim=2).flatten(end_dim=1)
    expected_full = Jac.T @ Hes @ Jac
    expected_diag = torch.diagonal(expected_full)
    expected = {"a": expected_diag}

    # Test unnormalised
    ggndiag_result = diag_ggn(forward, loss)(x)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result[k], expected[k], rtol=1e-5)

    # Test normalised
    ggndiag_result_norm = diag_ggn(forward, loss, normalize=True)(x)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result_norm[k], expected[k] / 2, rtol=1e-5)

    # Test forward aux
    def forward_aux(x):
        return forward(x), x

    ggndiag_result_faux = diag_ggn(forward_aux, loss, forward_has_aux=True)(x)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result_faux[0][k], expected[k], rtol=1e-5)
        assert torch.allclose(ggndiag_result_faux[1][k], x[k])
    assert len(ggndiag_result_faux) == 2

    # Test loss aux
    def loss_aux(z):
        return loss(z), z

    ggndiag_result_laux = diag_ggn(forward, loss_aux, loss_has_aux=True)(x)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result_laux[0][k], expected[k], rtol=1e-5)
    assert len(ggndiag_result_laux) == 2
    assert torch.allclose(ggndiag_result_laux[1], z)

    # Test both aux
    ggndiag_result_flaux = diag_ggn(
        forward_aux, loss_aux, forward_has_aux=True, loss_has_aux=True
    )(x)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result_flaux[0][k], expected[k], rtol=1e-5)
        assert torch.allclose(ggndiag_result_flaux[1][k], x[k])
    assert len(ggndiag_result_flaux) == 3
    assert torch.allclose(ggndiag_result_flaux[2], z)

    # Test model
    model = TestModel()
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3,)).unsqueeze(-1)

    def forward(params, inputs):
        return torch.func.functional_call(model, params, inputs)

    def loss(logits, labels):
        return torch.nn.BCEWithLogitsLoss()(logits, labels.float())

    ggndiag_result = diag_ggn(
        partial(forward, inputs=batch_inputs), partial(loss, labels=batch_labels)
    )(params)

    flat_params, unravel_params = tree_ravel(params)

    def forward_flat(p):
        return forward(unravel_params(p), batch_inputs)[:, 0]

    z = forward_flat(flat_params)
    jac = torch.func.jacrev(forward_flat)(flat_params)
    hess = torch.func.hessian(partial(loss, labels=batch_labels[:, 0]))(z)
    expected_dense = jac.T @ hess @ jac
    expected_diag = torch.diagonal(expected_dense)
    expected = unravel_params(expected_diag)
    for k in expected.keys():
        assert torch.allclose(ggndiag_result[k], expected[k], rtol=1e-5)


def test_cg():
    # simple function with tensor parameters
    def func(x):
        return torch.stack([(x**5).sum(), (x**3).sum()])

    def partial_fvp(v):
        return fvp(func, (x,), (v,), normalize=False)[1]

    x = torch.arange(1.0, 6.0)
    v = torch.ones_like(x)

    jac = torch.func.jacrev(func)(x)
    fisher = jac.T @ jac
    damping = 100

    sol = torch.linalg.solve(fisher + damping * torch.eye(fisher.shape[0]), v)
    sol_cg, _ = cg(partial_fvp, v, x0=None, damping=damping, maxiter=10000, tol=1e-10)
    assert torch.allclose(sol, sol_cg, rtol=1e-3)

    # simple complex number example
    A = torch.tensor([[0, -1j], [1j, 0]])

    def mvp(x):
        return A @ x

    b = torch.randn(2, dtype=torch.cfloat)

    sol = torch.linalg.solve(A, b)
    sol_cg, _ = cg(mvp, b, x0=None, tol=1e-10)

    assert torch.allclose(sol, sol_cg, rtol=1e-1)

    # function with parameters as a TensorTree
    model = TestModel()

    func_model = model_to_function(model)
    f_per_sample = torch.vmap(func_model, in_dims=(None, 0))

    xs = torch.randn(100, 10)

    def partial_fvp(v):
        return fvp(lambda p: func_model(p, xs), (params,), (v,), normalize=False)[1]

    params = dict(model.named_parameters())
    fisher = empirical_fisher(lambda p: f_per_sample(p, xs), normalize=False)(params)
    damping = 0

    v, _ = tree_ravel(params)
    sol = torch.linalg.solve(fisher + damping * torch.eye(fisher.shape[0]), v)
    sol_cg, _ = cg(partial_fvp, params, x0=None, damping=damping, tol=1e-10)

    assert torch.allclose(sol, tree_ravel(sol_cg)[0], rtol=1e-3)


def test_diag_normal_log_prob():
    # Test tree mean and tree sd
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = {"a": torch.tensor([0.1, 0.2]), "b": torch.tensor([0.3, 0.4])}
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}

    mean_flat = torch.stack([mean["a"], mean["b"]]).flatten()
    sd_flat = torch.stack([sd_diag["a"], sd_diag["b"]]).flatten()
    x_flat = torch.stack([x["a"], x["b"]]).flatten()

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = torch.distributions.Normal(mean_flat, sd_flat).log_prob(x_flat).sum()

    assert torch.allclose(result, expected)

    # Test float mean and tree sd
    mean = 1.0
    mean_tree = tree_map(lambda t: torch.ones_like(t) * mean, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean_tree, sd_diag)

    assert torch.allclose(result, expected)

    # Test tree mean and float sd
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = 0.1

    sd_tree = tree_map(lambda t: torch.ones_like(t) * sd_diag, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean, sd_tree)

    assert torch.allclose(result, expected)

    # Test float mean and float sd
    mean = 1.0
    sd_diag = 0.1

    mean_tree = tree_map(lambda t: torch.ones_like(t) * mean, x)
    sd_tree = tree_map(lambda t: torch.ones_like(t) * sd_diag, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean_tree, sd_tree)

    assert torch.allclose(result, expected)

    # Test unnormalized
    result = diag_normal_log_prob(x, mean, sd_diag, normalize=False)
    expected = tree_reduce(
        torch.add,
        tree_map(
            lambda v, m, sd: -((v - m) ** 2 / (2 * sd**2)).sum(), x, mean_tree, sd_tree
        ),
    )
    assert torch.allclose(result, expected)

    # Test unnormalised gradient
    result_grad = torch.func.grad(diag_normal_log_prob)(x, mean, sd_diag)
    expected_grad = tree_map(lambda v, m, sd: -(v - m) / (sd**2), x, mean_tree, sd_tree)

    for key in result_grad.keys():
        assert torch.allclose(result_grad[key], expected_grad[key])


def test_diag_normal_sample():
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = {"a": torch.tensor([0.1, 0.2]), "b": torch.tensor([0.3, 0.4])}

    torch.manual_seed(42)
    result = diag_normal_sample(mean, sd_diag)
    torch.manual_seed(42)
    expected = tree_map(
        lambda m, sd: torch.distributions.Normal(m, sd).sample(), mean, sd_diag
    )

    for key in result:
        assert torch.equal(result[key], expected[key])

    n_samps = 1000
    result = diag_normal_sample(mean, sd_diag, sample_shape=(n_samps,))
    result_mean = tree_map(lambda v: v.mean(dim=0), result)
    result_std = tree_map(lambda v: v.std(dim=0), result)

    for key in result_mean:
        assert torch.allclose(result_mean[key], mean[key], atol=1e-1)
        assert torch.allclose(result_std[key], sd_diag[key], atol=1e-1)

    # Test float sd_diag
    sd_diag = 0.1
    torch.manual_seed(42)
    result = diag_normal_sample(mean, sd_diag)
    torch.manual_seed(42)
    expected = tree_map(lambda m: torch.distributions.Normal(m, sd_diag).sample(), mean)

    for key in result:
        assert torch.equal(result[key], expected[key])


@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("shape", [(), (1,)])
@pytest.mark.parametrize("structured", ["mean", "sd", "both"])
@pytest.mark.parametrize("normalize", [False, True])
def test_diag_normal_log_prob_singleton_tree(as_tuple, shape, structured, normalize):
    def wrap(value):
        return (value,) if as_tuple else {"weight": value}

    value = torch.full(shape, 1.5)
    mean = torch.full(shape, 0.5)
    sd = torch.full(shape, 2.0)
    result = diag_normal_log_prob(
        wrap(value),
        wrap(mean) if structured in ("mean", "both") else 0.5,
        wrap(sd) if structured in ("sd", "both") else 2.0,
        normalize=normalize,
    )
    expected = (
        torch.distributions.Normal(mean, sd).log_prob(value).sum()
        if normalize
        else (-0.5 * ((value - mean) / sd) ** 2).sum()
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("shape", [(), (1,)])
def test_diag_normal_sample_singleton_tree(as_tuple, shape):
    mean = torch.full(shape, 0.5)
    sd = torch.full(shape, 2.0)
    wrap = (lambda value: (value,)) if as_tuple else (lambda value: {"weight": value})

    torch.manual_seed(42)
    result = diag_normal_sample(wrap(mean), wrap(sd), sample_shape=(2, 3))
    samples = result[0] if as_tuple else result["weight"]
    torch.manual_seed(42)
    expected = torch.distributions.Normal(mean, sd).sample((2, 3))

    assert samples.shape == (2, 3) + shape
    torch.testing.assert_close(samples, expected)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_diag_normal_singleton_tree_parameter_gradients(as_tuple):
    def wrap(value):
        return (value,) if as_tuple else {"weight": value}

    value = torch.tensor(1.5, requires_grad=True)
    mean = torch.tensor(0.5, requires_grad=True)
    sd = torch.tensor(2.0, requires_grad=True)
    result = diag_normal_log_prob(wrap(value), wrap(mean), wrap(sd))
    actual = torch.autograd.grad(result, (mean, sd))
    expected = torch.autograd.grad(
        torch.distributions.Normal(mean, sd).log_prob(value), (mean, sd)
    )
    tree_map(torch.testing.assert_close, actual, expected)


def test_per_samplify():
    def func(p, b):
        return p + b, p

    p = torch.tensor(1.0)
    b = torch.tensor([1.0, 2.0])

    func1 = per_samplify(func)
    ra, rb = func1(p, b)
    expected_a = torch.tensor([[2.0], [3.0]])
    expected_b = torch.tensor([1.0, 1.0])
    assert torch.allclose(ra, expected_a)
    assert torch.allclose(rb, expected_b)

    # Test model
    model = TestModel()  # From tests.scenarios
    params = dict(model.named_parameters())
    batch_inputs = torch.randn(3, 10)
    batch_labels = torch.randint(2, (3, 1))
    batch_spec = {"inputs": batch_inputs, "labels": batch_labels}

    def log_likelihood(params, batch):
        output = torch.func.functional_call(model, params, batch["inputs"])
        return -torch.nn.BCEWithLogitsLoss()(output, batch["labels"].float())

    log_likelihood_per_sample = per_samplify(log_likelihood)

    expected = torch.tensor(
        [
            log_likelihood(
                params, {"inputs": inp.unsqueeze(0), "labels": lab.unsqueeze(0)}
            )
            for inp, lab in zip(batch_inputs, batch_labels)
        ]
    )

    eval = log_likelihood_per_sample(params, batch_spec)
    eval_p = partial(log_likelihood_per_sample, batch=batch_spec)(params)
    assert torch.allclose(expected, eval)
    assert torch.allclose(expected, eval_p)


def test_is_scalar():
    assert is_scalar(1)
    assert is_scalar(1.0)
    assert is_scalar(torch.ones(1))
    assert is_scalar(torch.tensor(1.0))
    assert is_scalar(torch.ones(1, 1))
    assert not is_scalar(torch.ones(2))


def test_L_from_flat():
    expected_L = torch.Tensor(
        [
            [1.0, 0.0, 0.0],
            [-4.1, 2.2, 0.0],
            [-1.7, 4.4, -5.5],
        ]
    )
    L_flat = torch.tensor([1.0, -4.1, 2.2, -1.7, 4.4, -5.5])
    L = L_from_flat(L_flat)
    assert torch.allclose(expected_L, L)


def test_L_to_flat():
    L = torch.Tensor(
        [
            [1.0, 0.0, 0.0],
            [-4.1, 2.2, 0.0],
            [-1.7, 4.4, -5.5],
        ]
    )
    expected_L_flat = torch.tensor([1.0, -4.1, 2.2, -1.7, 4.4, -5.5])
    L_flat = L_to_flat(L)
    assert torch.allclose(expected_L_flat, L_flat)


def test_cumulative_mean_and_cov():
    xs = torch.randn(30, 5)
    cum_means, cum_covs = cumulative_mean_and_cov(xs)

    assert cum_means.shape == xs.shape
    assert cum_covs.shape == xs.shape + (xs.shape[1],)

    for i in range(xs.shape[0]):
        assert torch.allclose(cum_means[i], xs[: i + 1].mean(0), atol=1e-7)
        if i > 0:
            assert torch.allclose(cum_covs[i], xs[: i + 1].T.cov(), atol=1e-7)
