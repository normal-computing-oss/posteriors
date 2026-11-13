from typing import Callable, Any, Tuple, Sequence
import operator
from functools import partial, wraps
import contextlib
import torch
from torch.func import grad, jvp, vjp, functional_call, jacrev, jacfwd
from torch.distributions import Normal
from optree import tree_map, tree_reduce, tree_flatten, tree_leaves
from optree.integrations.torch import tree_ravel

from posteriors.types import TensorTree, ForwardFn, Tensor
from posteriors.tree_utils import tree_size


NO_AUX_ERROR_MSG = "should be a tuple: (output, aux) if has_aux is True"
NON_TENSOR_AUX_ERROR_MSG = "Expected tensors, got unsupported type"


class CatchAuxError(contextlib.AbstractContextManager):
    """Context manager to catch errors when auxiliary output is not found."""

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if NO_AUX_ERROR_MSG in str(exc_value):
                raise RuntimeError(
                    "Auxiliary output not found. Perhaps you have forgotten to return "
                    "the aux output?\n"
                    "\tIf you don't have any auxiliary info, simply amend to e.g. "
                    "log_posterior(params, batch) -> Tuple[float, torch.tensor([])].\n"
                    "\tMore info at https://normal-computing-oss.github.io/posteriors/log_posteriors"
                )
            elif NON_TENSOR_AUX_ERROR_MSG in str(exc_value):
                raise RuntimeError(
                    "Auxiliary output should be a TensorTree. If you don't have any "
                    "auxiliary info, simply amend to e.g. "
                    "log_posterior(params, batch) -> Tuple[float, torch.tensor([])].\n"
                    "\tMore info at https://normal-computing-oss.github.io/posteriors/log_posteriors"
                )
        return False


def model_to_function(model: torch.nn.Module) -> Callable[[TensorTree, Any], Any]:
    """Converts a model into a function that maps parameters and inputs to outputs.

    Convenience wrapper around [torch.functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html).

    Args:
        model: torch.nn.Module with parameters stored in .named_parameters().

    Returns:
        Function that takes a PyTree of parameters as well as any input
            arg or kwargs and returns the output of the model.
    """

    def func_model(p_dict, *args, **kwargs):
        return functional_call(model, p_dict, args=args, kwargs=kwargs)

    return func_model


def linearized_forward_diag(
    forward_func: ForwardFn, params: TensorTree, batch: TensorTree, sd_diag: TensorTree
) -> Tuple[TensorTree, Tensor, TensorTree]:
    """Compute the linearized forward mean and its square root covariance, assuming
    posterior covariance over parameters is diagonal.

    $$
    f(x | θ) \\sim N(x | f(x | θₘ), J(x | θₘ) \\Sigma J(x | θₘ)^T)
    $$
    where $θₘ$ is the MAP estimate, $\\Sigma$ is the diagonal covariance approximation
    at the MAP and $J(x | θₘ)$ is the Jacobian of the forward function $f(x | θₘ)$ with
    respect to $θₘ$.

    For more info on linearized models see [Foong et al, 2019](https://arxiv.org/abs/1906.11537).

    Args:
        forward_func: A function that takes params and batch and returns the forward
            values and any auxiliary information. Forward values must be a dim=2 Tensor
            with batch dimension in its first axis.
        params: PyTree of tensors.
        batch: PyTree of tensors.
        sd_diag: PyTree of tensors of same shape as params.

    Returns:
        A tuple of (forward_vals, chol, aux) where forward_vals is the output of the
            forward function (mean), chol is the tensor square root of the covariance
            matrix (non-diagonal) and aux is auxiliary info from the forward function.
    """
    forward_vals, aux = forward_func(params, batch)

    with torch.no_grad(), CatchAuxError():
        jac, _ = jacrev(forward_func, has_aux=True)(params, batch)

    # Convert Jacobian to be flat in parameter dimension
    jac = tree_flatten(jac)[0]
    jac = torch.cat([x.flatten(start_dim=2) for x in jac], dim=2)

    # Flatten the diagonal square root covariance
    sd_diag = tree_flatten(sd_diag)[0]
    sd_diag = torch.cat([x.flatten() for x in sd_diag])

    # Cholesky of J @ Σ @ J^T
    linearised_chol = torch.linalg.cholesky((jac * sd_diag**2) @ jac.transpose(-1, -2))

    return forward_vals, linearised_chol, aux


def hvp(
    f: Callable, primals: tuple, tangents: tuple, has_aux: bool = False
) -> Tuple[float, TensorTree] | Tuple[float, TensorTree, Any]:
    """Hessian vector product.

    H(primals) @ tangents

    where H(primals) is the Hessian of f evaluated at primals.

    Taken from [jacobians_hessians.html](https://pytorch.org/functorch/nightly/notebooks/jacobians_hessians.html).
    Follows API from [`torch.func.jvp`](https://pytorch.org/docs/stable/generated/torch.func.jvp.html).

    Args:
        f: A function with scalar output.
        primals: Tuple of e.g. tensor or dict with tensor values to evalute f at.
        tangents: Tuple matching structure of primals.
        has_aux: Whether f returns auxiliary information.

    Returns:
        Returns a (gradient, hvp_out) tuple containing the gradient of func evaluated at
            primals and the Hessian-vector product. If has_aux is True, then instead
            returns a (gradient, hvp_out, aux) tuple.
    """
    return jvp(grad(f, has_aux=has_aux), primals, tangents, has_aux=has_aux)


def fvp(
    f: Callable,
    primals: tuple,
    tangents: tuple,
    has_aux: bool = False,
    normalize: bool = False,
) -> Tuple[float, TensorTree] | Tuple[float, TensorTree, Any]:
    """Empirical Fisher vector product.

    F(primals) @ tangents

    where F(primals) is the empirical Fisher of f evaluated at primals.

    The empirical Fisher is defined as:
    $$
    F(θ) = J_f(θ) J_f(θ)^T
    $$
    where typically $f_θ$ is the per-sample log likelihood (with elements
    $\\log p(y_i | x_i, θ)$ for a model with `primals` $θ$ given inputs $x_i$ and
    labels $y_i$).

    If `normalize=True`, then $F(θ)$ is divided by the number of outputs from f
    (i.e. batchsize).

    Follows API from [`torch.func.jvp`](https://pytorch.org/docs/stable/generated/torch.func.jvp.html).

    More info on empirical Fisher matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).

    Examples:
        ```python
        from functools import partial
        from optree import tree_map
        import torch
        from posteriors import fvp

        # Load model that outputs logits
        # Load batch = {'inputs': ..., 'labels': ...}

        def log_likelihood_per_sample(params, batch):
            output = torch.func.functional_call(model, params, batch["inputs"])
            return -torch.nn.functional.cross_entropy(
                output, batch["labels"], reduction="none"
            )

        params = dict(model.parameters())
        v = tree_map(lambda x: torch.randn_like(x), params)
        fvp_result = fvp(
            partial(log_likelihood_per_sample, batch=batch),
            (params,),
            (v,)
        )
        ```

    Args:
        f: A function with tensor output.
            Typically this is the [per-sample log likelihood of a model](https://pytorch.org/tutorials/intermediate/per_sample_grads.html).
        primals: Tuple of e.g. tensor or dict with tensor values to evaluate f at.
        tangents: Tuple matching structure of primals.
        has_aux: Whether f returns auxiliary information.
        normalize: Whether to normalize, divide by the dimension of the output from f.

    Returns:
        Returns a (output, fvp_out) tuple containing the output of func evaluated at
            primals and the empirical Fisher-vector product. If has_aux is True, then
            instead returns a (output, fvp_out, aux) tuple.
    """
    jvp_output = jvp(f, primals, tangents, has_aux=has_aux)
    Jv = jvp_output[1]
    f_vjp = vjp(f, *primals, has_aux=has_aux)[1]
    Fv = f_vjp(Jv)[0]

    if normalize:
        output_dim = tree_flatten(jvp_output[0])[0][0].shape[0]
        Fv = tree_map(lambda x: x / output_dim, Fv)

    return jvp_output[0], Fv, *jvp_output[2:]


def empirical_fisher(
    f: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    normalize: bool = False,
) -> Callable:
    """
    Constructs function to compute the empirical Fisher information matrix of a function
    f with respect to its parameters, defined as (unnormalized):
    $$
    F(θ) = J_f(θ) J_f(θ)^T
    $$
    where typically $f_θ$ is the per-sample log likelihood (with elements
    $\\log p(y_i | x_i, θ)$ for a model with `primals` $θ$ given inputs $x_i$ and
    labels $y_i$).

    If `normalize=True`, then $F(θ)$ is divided by the number of outputs from f
    (i.e. batchsize).

    The empirical Fisher will be provided as a square tensor with respect to the
    ravelled parameters.
    `flat_params, params_unravel = optree.tree_ravel(params)`.

    Follows API from [`torch.func.jacrev`](https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    More info on empirical Fisher matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).

    Examples:
        ```python
        import torch
        from posteriors import empirical_fisher, per_samplify

        # Load model that outputs logits
        # Load batch = {'inputs': ..., 'labels': ...}

        def log_likelihood(params, batch):
            output = torch.func.functional_call(model, params, batch['inputs'])
            return -torch.nn.functional.cross_entropy(output, batch['labels'])

        likelihood_per_sample = per_samplify(log_likelihood)
        params = dict(model.parameters())
        ef_result = empirical_fisher(log_likelihood_per_sample)(params, batch)
        ```

    Args:
        f:  A Python function that takes one or more arguments, one of which must be a
            Tensor, and returns one or more Tensors.
            Typically this is the [per-sample log likelihood of a model](https://pytorch.org/tutorials/intermediate/per_sample_grads.html).
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to.
        has_aux: Whether f returns auxiliary information.
        normalize: Whether to normalize, divide by the dimension of the output from f.

    Returns:
        A function with the same arguments as f that returns the empirical Fisher, F.
            If has_aux is True, then the function instead returns a tuple of (F, aux).
    """

    def f_to_flat(*args, **kwargs):
        f_out = f(*args, **kwargs)
        f_out_val = f_out[0] if has_aux else f_out
        f_out_val = tree_ravel(f_out_val)[0]
        return (f_out_val, f_out[1]) if has_aux else f_out_val

    def fisher(*args, **kwargs):
        jac_output = jacrev(f_to_flat, argnums=argnums, has_aux=has_aux)(
            *args, **kwargs
        )
        jac = jac_output[0] if has_aux else jac_output

        # Convert Jacobian to tensor, flat in parameter dimension
        jac = torch.vmap(lambda x: tree_ravel(x)[0])(jac)

        rescale = 1 / jac.shape[0] if normalize else 1

        if has_aux:
            return jac.T @ jac * rescale, jac_output[1]
        else:
            return jac.T @ jac * rescale

    return fisher


def ggnvp(
    forward: Callable,
    loss: Callable,
    primals: tuple,
    tangents: tuple,
    forward_has_aux: bool = False,
    loss_has_aux: bool = False,
    normalize: bool = False,
) -> (
    Tuple[float, TensorTree]
    | Tuple[float, TensorTree, Any]
    | Tuple[float, TensorTree, Any, Any]
):
    """Generalised Gauss-Newton vector product.

    Equivalent to the (non-empirical) Fisher vector product when `loss` is the negative
    log likelihood of an exponential family distribution as a function of its natural
    parameter.

    Defined as
    $$
    G(θ) = J_f(θ) H_l(z) J_f(θ)^T
    $$
    where $z = f(θ)$ is the output of the forward function $f$ and $l(z)$
    is a loss function with scalar output.

    Thus $J_f(θ)$ is the Jacobian of the forward function $f$ evaluated
    at `primals` $θ$, with dimensions `(dz, dθ)`.
    And $H_l(z)$ is the Hessian of the loss function $l$ evaluated at `z = f(θ)`, with
    dimensions `(dz, dz)`.

    Follows API from [`torch.func.jvp`](https://pytorch.org/docs/stable/generated/torch.func.jvp.html).

    More info on Fisher and GGN matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).

    Examples:
        ```python
        from functools import partial
        from optree import tree_map
        import torch
        from posteriors import ggnvp

        # Load model that outputs logits
        # Load batch = {'inputs': ..., 'labels': ...}

        def forward(params, inputs):
            return torch.func.functional_call(model, params, inputs)

        def loss(logits, labels):
            return torch.nn.functional.cross_entropy(logits, labels)

        params = dict(model.parameters())
        v = tree_map(lambda x: torch.randn_like(x), params)
        ggnvp_result = ggnvp(
            partial(forward, inputs=batch['inputs']),
            partial(loss, labels=batch['labels']),
            (params,),
            (v,),
        )
        ```

    Args:
        forward: A function with tensor output.
        loss: A function that maps the output of forward to a scalar output.
        primals: Tuple of e.g. tensor or dict with tensor values to evaluate f at.
        tangents: Tuple matching structure of primals.
        forward_has_aux: Whether forward returns auxiliary information.
        loss_has_aux: Whether loss returns auxiliary information.
        normalize: Whether to normalize, divide by the first dimension of the output
            from f.

    Returns:
        Returns a (output, ggnvp_out) tuple, where output is a tuple of
            `(forward(primals), grad(loss)(forward(primals)))`.
            If forward_has_aux or loss_has_aux is True, then instead returns a
            (output, ggnvp_out, aux) or
            (output, ggnvp_out, forward_aux, loss_aux) tuple accordingly.
    """

    jvp_output = jvp(forward, primals, tangents, has_aux=forward_has_aux)
    z = jvp_output[0]
    Jv = jvp_output[1]
    HJv_output = hvp(loss, (z,), (Jv,), has_aux=loss_has_aux)
    HJv = HJv_output[1]

    if normalize:
        output_dim = tree_flatten(jvp_output[0])[0][0].shape[0]
        HJv = tree_map(lambda x: x / output_dim, HJv)

    forward_vjp = vjp(forward, *primals, has_aux=forward_has_aux)[1]
    JTHJv = forward_vjp(HJv)[0]

    return (jvp_output[0], HJv_output[0]), JTHJv, *jvp_output[2:], *HJv_output[2:]


def _hess_and_jac_for_ggn(
    flat_params_to_forward,
    loss,
    argnums,
    forward_has_aux,
    loss_has_aux,
    normalize,
    flat_params,
) -> Tuple[Tensor, Tensor, list]:
    jac_output = jacrev(
        flat_params_to_forward, argnums=argnums, has_aux=forward_has_aux
    )(flat_params)
    jac = jac_output[0] if forward_has_aux else jac_output  # (..., dθ)
    jac = torch.stack(tree_leaves(jac))[
        0
    ]  # convert to tensor (assumes jac has tensor output)
    rescale = 1 / jac.shape[0] if normalize else 1  #  maybe normalize by batchsize
    jac = jac.flatten(end_dim=-2)  # (d, dθ)

    z = flat_params_to_forward(flat_params)
    z = z[0] if forward_has_aux else z

    hess_output = jacfwd(jacrev(loss, has_aux=loss_has_aux), has_aux=loss_has_aux)(z)
    hess = hess_output[0] if loss_has_aux else hess_output
    hess = torch.stack(tree_leaves(hess))[
        0
    ]  # convert to tensor (assumes loss has tensor input)
    z_ndim = hess.ndim // 2
    hess = hess.flatten(start_dim=z_ndim).flatten(
        end_dim=-z_ndim
    )  # flatten to square tensor

    hess *= rescale

    # Collect aux outputs
    aux = []
    if forward_has_aux:
        aux.append(jac_output[1])
    if loss_has_aux:
        aux.append(loss(z)[1])

    return jac, hess, aux


def ggn(
    forward: Callable,
    loss: Callable,
    argnums: int | Sequence[int] = 0,
    forward_has_aux: bool = False,
    loss_has_aux: bool = False,
    normalize: bool = False,
) -> Callable:
    """
    Constructs function to compute the Generalised Gauss-Newton matrix.

    Equivalent to the (non-empirical) Fisher when `loss` is the negative
    log likelihood of an exponential family distribution as a function of its natural
    parameter.

    Defined as
    $$
    G(θ) = J_f(θ) H_l(z) J_f(θ)^T
    $$
    where $z = f(θ)$ is the output of the forward function $f$ and $l(z)$
    is a loss function with scalar output.

    Thus $J_f(θ)$ is the Jacobian of the forward function $f$ evaluated
    at `primals` $θ$. And $H_l(z)$ is the Hessian of the loss function $l$ evaluated
    at `z = f(θ)`.

    Requires output from `forward` to be a tensor and therefore `loss` takes a tensor as
    input. Although both support `aux` output.

    If `normalize=True`, then $G(θ)$ is divided by the size of the leading dimension of
    outputs from `forward` (i.e. batchsize).

    The GGN will be provided as a square tensor with respect to the
    ravelled parameters.
    `flat_params, params_unravel = optree.tree_ravel(params)`.

    Follows API from [`torch.func.jacrev`](https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    More info on Fisher and GGN matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).

    Examples:
        ```python
        from functools import partial
        import torch
        from posteriors import ggn

        # Load model that outputs logits
        # Load batch = {'inputs': ..., 'labels': ...}

        def forward(params, inputs):
            return torch.func.functional_call(model, params, inputs)

        def loss(logits, labels):
            return torch.nn.functional.cross_entropy(logits, labels)

        params = dict(model.parameters())
        ggn_result = ggn(
            partial(forward, inputs=batch['inputs']),
            partial(loss, labels=batch['labels']),
        )(params)
        ```

    Args:
        forward: A function with tensor output.
        loss: A function that maps the output of forward to a scalar output.
            Takes a single input and returns a scalar (and possibly aux).
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate `forward` with respect to.
        forward_has_aux: Whether forward returns auxiliary information.
        loss_has_aux: Whether loss returns auxiliary information.
        normalize: Whether to normalize, divide by the first dimension of the output
            from f.

    Returns:
        A function with the same arguments as f that returns the tensor GGN.
            If has_aux is True, then the function instead returns a tuple of (F, aux).
    """
    assert argnums == 0, "Only argnums=0 is supported for now."

    def internal_ggn(params):
        flat_params, params_unravel = tree_ravel(params)

        def flat_params_to_forward(fps):
            return forward(params_unravel(fps))

        jac, hess, aux = _hess_and_jac_for_ggn(
            flat_params_to_forward,
            loss,
            argnums,
            forward_has_aux,
            loss_has_aux,
            normalize,
            flat_params,
        )

        if aux:
            return jac.T @ (hess @ jac), *aux
        else:
            return jac.T @ (hess @ jac)

    return internal_ggn


def diag_ggn(
    forward: Callable,
    loss: Callable,
    argnums: int | Sequence[int] = 0,
    forward_has_aux: bool = False,
    loss_has_aux: bool = False,
    normalize: bool = False,
) -> Callable:
    """
    Constructs function to compute the diagonal of the Generalised Gauss-Newton matrix.

    Equivalent to the (non-empirical) diagonal Fisher when `loss` is the negative
    log likelihood of an exponential family distribution as a function of its natural
    parameter.

    The GGN is defined as
    $$
    G(θ) = J_f(θ) H_l(z) J_f(θ)^T
    $$
    where $z = f(θ)$ is the output of the forward function $f$ and $l(z)$
    is a loss function with scalar output.

    Thus $J_f(θ)$ is the Jacobian of the forward function $f$ evaluated
    at `primals` $θ$. And $H_l(z)$ is the Hessian of the loss function $l$ evaluated
    at `z = f(θ)`.

    Requires output from `forward` to be a tensor and therefore `loss` takes a tensor as
    input. Although both support `aux` output.

    If `normalize=True`, then $G(θ)$ is divided by the size of the leading dimension of
    outputs from `forward` (i.e. batchsize).

    Unlike `posteriors.ggn`, the output will be in PyTree form matching the input.

    Follows API from [`torch.func.jacrev`](https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    More info on Fisher and GGN matrices can be found in
    [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).

    Examples:
        ```python
        from functools import partial
        import torch
        from posteriors import diag_ggn

        # Load model that outputs logits
        # Load batch = {'inputs': ..., 'labels': ...}

        def forward(params, inputs):
            return torch.func.functional_call(model, params, inputs)

        def loss(logits, labels):
            return torch.nn.functional.cross_entropy(logits, labels)

        params = dict(model.parameters())
        ggndiag_result = diag_ggn(
            partial(forward, inputs=batch['inputs']),
            partial(loss, labels=batch['labels']),
        )(params)
        ```

    Args:
        forward: A function with tensor output.
        loss: A function that maps the output of forward to a scalar output.
            Takes a single input and returns a scalar (and possibly aux).
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate `forward` with respect to.
        forward_has_aux: Whether forward returns auxiliary information.
        loss_has_aux: Whether loss returns auxiliary information.
        normalize: Whether to normalize, divide by the first dimension of the output
            from f.

    Returns:
        A function with the same arguments as f that returns the diagonal GGN.
            If has_aux is True, then the function instead returns a tuple of (F, aux).
    """
    assert argnums == 0, "Only argnums=0 is supported for now."

    def internal_ggn(params):
        flat_params, params_unravel = tree_ravel(params)

        def flat_params_to_forward(fps):
            return forward(params_unravel(fps))

        jac, hess, aux = _hess_and_jac_for_ggn(
            flat_params_to_forward,
            loss,
            argnums,
            forward_has_aux,
            loss_has_aux,
            normalize,
            flat_params,
        )

        G_diag = torch.einsum("ji,jk,ki->i", jac, hess, jac)
        G_diag = params_unravel(G_diag)

        if aux:
            return G_diag, *aux
        else:
            return G_diag

    return internal_ggn


def _vdot_real_part(x: Tensor, y: Tensor) -> float:
    """Vector dot-product guaranteed to have a real valued result despite
    possibly complex input. Thus neglects the real-imaginary cross-terms.

    Args:
        x: First tensor in the dot product.
        y: Second tensor in the dot product.

    Returns:
        The result vector dot-product, a real float
    """
    # all our uses of vdot() in CG are for computing an operator of the form
    #  z^H M z
    #  where M is positive definite and Hermitian, so the result is
    # real valued:
    # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
    real_part = torch.vdot(x.real.flatten(), y.real.flatten())
    if torch.is_complex(x) or torch.is_complex(y):
        imag_part = torch.vdot(x.imag.flatten(), y.imag.flatten())
        return real_part + imag_part
    return real_part


def _vdot_real_tree(x, y) -> TensorTree:
    return sum(tree_leaves(tree_map(_vdot_real_part, x, y)))


def _mul(scalar, tree) -> TensorTree:
    return tree_map(partial(operator.mul, scalar), tree)


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)


def _identity(x):
    return x


def cg(
    A: Callable,
    b: TensorTree,
    x0: TensorTree = None,
    *,
    maxiter: int = None,
    damping: float = 0.0,
    tol: float = 1e-5,
    atol: float = 0.0,
    M: Callable = _identity,
) -> Tuple[TensorTree, Any]:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.
    ``A`` is supplied as a function instead of a matrix.

    Adapted from [`jax.scipy.sparse.linalg.cg`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html).

    Args:
        A:  Callable that calculates the linear map (matrix-vector
            product) ``Ax`` when called like ``A(x)``. ``A`` must represent
            a hermitian, positive definite matrix, and must return array(s) with the
            same structure and shape as its argument.
        b:  Right hand side of the linear system representing a single vector.
        x0: Starting guess for the solution. Must have the same structure as ``b``.
        maxiter: Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        damping: damping term for the mvp function. Acts as regularization.
        tol: Tolerance for convergence.
        atol: Tolerance for convergence. ``norm(residual) <= max(tol*norm(b), atol)``.
            The behaviour will differ from SciPy unless you explicitly pass
            ``atol`` to SciPy's ``cg``.
        M: Preconditioner for A. Maps a residual to an approximate solution of
            the damped linear system. Applied to the initial residual and each update.
            See [the preconditioned CG method.](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method)

    Returns:
        x : The converged solution. Has the same structure as ``b``.
        info : Placeholder for convergence information.
    """
    if x0 is None:
        x0 = tree_map(torch.zeros_like, b)

    if maxiter is None:
        maxiter = 10 * tree_size(b)  # copied from scipy

    tol *= torch.tensor([1.0])
    atol *= torch.tensor([1.0])

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = torch.maximum(torch.square(tol) * bs, torch.square(atol))

    def A_damped(p):
        return _add(A(p), _mul(damping, p))

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A_damped(p)
        alpha = gamma / _vdot_real_tree(p, Ap)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1

    r0 = _sub(b, A_damped(x0))
    p0 = z0 = M(r0)
    gamma0 = _vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0, 0)

    value = initial_value

    while cond_fun(value):
        value = body_fun(value)

    x_final, r, gamma, _, k = value
    # compute the final error and whether it has converged.
    rs = gamma if M is _identity else _vdot_real_tree(r, r)
    converged = rs <= atol2

    # additional info output structure
    info = {"error": rs, "converged": converged, "niter": k}

    return x_final, info


def diag_normal_log_prob(
    x: TensorTree,
    mean: float | TensorTree = 0.0,
    sd_diag: float | TensorTree = 1.0,
    normalize: bool = True,
) -> float:
    """Evaluate multivariate normal log probability for a diagonal covariance matrix.

    If either mean or sd_diag are scalars, they will be broadcast to the same shape as x
    (in a memory efficient manner).

    Args:
        x: Value to evaluate log probability at.
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.
        normalize: Whether to compute normalized log probability.
            If False the elementwise log prob is -0.5 * ((x - mean) / sd_diag)**2.

    Returns:
        Scalar log probability.
    """
    if tree_size(mean) == 1:
        mean = tree_map(lambda t: torch.tensor(mean, device=t.device), x)
    if tree_size(sd_diag) == 1:
        sd_diag = tree_map(lambda t: torch.tensor(sd_diag, device=t.device), x)

    if normalize:

        def univariate_norm_and_sum(v, m, sd):
            return Normal(m, sd, validate_args=False).log_prob(v).sum()
    else:

        def univariate_norm_and_sum(v, m, sd):
            return (-0.5 * ((v - m) / sd) ** 2).sum()

    log_probs = tree_map(
        univariate_norm_and_sum,
        x,
        mean,
        sd_diag,
    )
    log_prob = tree_reduce(torch.add, log_probs)
    return log_prob


def diag_normal_sample(
    mean: TensorTree,
    sd_diag: float | TensorTree,
    sample_shape: torch.Size = torch.Size([]),
) -> dict:
    """Sample from multivariate normal with diagonal covariance matrix.

    If sd_diag is scalar, it will be broadcast to the same shape as mean
    (in a memory efficient manner).

    Args:
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.
        sample_shape: Shape of the sample.

    Returns:
        Sample(s) from normal distribution with the same structure as mean and sd_diag.
    """
    if tree_size(sd_diag) == 1:
        sd_diag = tree_map(lambda t: torch.tensor(sd_diag, device=t.device), mean)

    return tree_map(
        lambda m, sd: m + torch.randn(sample_shape + m.shape, device=m.device) * sd,
        mean,
        sd_diag,
    )


def per_samplify(
    f: Callable[[TensorTree, TensorTree], Any],
) -> Callable[[TensorTree, TensorTree], Any]:
    """Converts a function that takes params and batch into one that provides an output
    for each batch sample.

    ```
    output = f(params, batch)
    per_sample_output = per_samplify(f)(params, batch)
    ```

    For more info see [per_sample_grads.html](https://pytorch.org/tutorials/intermediate/per_sample_grads.html)

    Args:
        f: A function that takes params and batch provides an output with size
            independent of batchsize (i.e. averaged).

    Returns:
        A new function that provides an output for each batch sample.
            `per_sample_output  = per_samplify(f)(params, batch)`
    """

    @partial(torch.vmap, in_dims=(None, 0))
    def f_per_sample(params, batch):
        batch = tree_map(lambda x: x.unsqueeze(0), batch)
        return f(params, batch)

    @wraps(f)
    def f_per_sample_ensure_no_kwargs(params, batch):
        return f_per_sample(params, batch)  # vmap in_dims requires no kwargs

    return f_per_sample_ensure_no_kwargs


def is_scalar(x: Any) -> bool:
    """Returns True if x is a scalar (int, float, bool) or a tensor with a single element.

    Args:
        x: Any object.

    Returns:
        True if x is a scalar.
    """
    return isinstance(x, (int, float)) or (torch.is_tensor(x) and x.numel() == 1)


def L_from_flat(L_flat: torch.Tensor) -> torch.Tensor:
    """Returns lower triangular matrix from a flat representation of its nonzero elements.

    Args:
        L_flat: Flat representation of nonzero lower triangular matrix elements.

    Returns:
        Lower triangular matrix.
    """
    k = torch.tensor(L_flat.shape[0], dtype=L_flat.dtype, device=L_flat.device)
    n = (-1 + (1 + 8 * k).sqrt()) / 2
    num_params = round(n.item())

    tril_indices = torch.tril_indices(num_params, num_params)
    L = torch.zeros((num_params, num_params), device=L_flat.device)
    L[tril_indices[0], tril_indices[1]] = L_flat
    return L


def L_to_flat(L: torch.Tensor) -> torch.Tensor:
    """Returns flat representation of the nonzero elements of a lower triangular matrix.

    Args:
        L: Lower triangular matrix.

    Returns:
        Flat representation of the nonzero lower triangular matrix elements.
    """

    num_params = L.shape[0]
    tril_indices = torch.tril_indices(num_params, num_params)
    L_flat = L[tril_indices[0], tril_indices[1]].clone()
    return L_flat


def cumulative_mean_and_cov(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the cumulative mean and covariance of a sequence of tensors.

    This is a numerically efficient way to calculate the cumulative covariances
    in particular. It costs O(n d^2) time compared to a naive O(n^2 d^2)
    implementation.

    Args:
        xs: Sequence of tensors.

    Returns:
        Tuple of cumulative mean and covariance.
    """
    n, d = xs.shape
    out_means = torch.zeros((n, d))
    out_covs = torch.zeros((n, d, d))

    out_means[0] = xs[0]
    out_covs[0] = torch.eye(d)

    for i in range(1, n):
        n = i + 1
        # Update mean
        out_means[i] = out_means[i - 1] * (n - 1) / n + xs[i] / n

        # Update covariance
        delta_n = xs[i] - out_means[i - 1]
        out_covs[i] = (
            out_covs[i - 1] * (n - 2) / (n - 1) + torch.outer(delta_n, delta_n) / n
        )

    return out_means, out_covs
