## Installation
Install from [PyPI](https://pypi.org/project/posteriors/) with `pip`:

```bash
pip install posteriors
```

## Why UQ?

Uncertainty quantification allows for informed decision making by averaging over
multiple plausible model configurations rather than relying on a single point estimate.
Thus providing a coherent framework for detecting [**out of distribution**](https://github.com/normal-computing-oss/posteriors/tree/main/examples/yelp)
inputs and [**continual learning**](https://github.com/normal-computing-oss/posteriors/tree/main/examples/continual_lora).

For more info on the utility of UQ, check out our [blog post introducing `posteriors`](https://www.normalcomputing.com/blog/posteriors-normal-computings-library-for-uncertainty-aware-llms-3)!


## Quick Start

`posteriors` is a Python library for uncertainty quantification and machine learning 
that is designed to be easy to use, flexible and extensible. It is built on top 
of [PyTorch](https://pytorch.org/docs/stable/index.html) and provides a range of 
tools for probabilistic modelling, Bayesian inference, and online learning.

Enough smalltalk, let's train a simple Bayesian neural network using `posteriors`:

```py
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import nn, utils, func
import torchopt
import posteriors

dataset = MNIST(root="./data", transform=ToTensor(), download=True)
train_loader = utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
num_data = len(dataset)

classifier = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))
params = dict(classifier.named_parameters())


def log_posterior(params, batch):
    images, labels = batch
    images = images.view(images.size(0), -1)
    output = func.functional_call(classifier, params, images)
    log_post_val = (
        -nn.functional.cross_entropy(output, labels)
        + posteriors.diag_normal_log_prob(params) / num_data
    )
    return log_post_val, output


transform = posteriors.vi.diag.build(
    log_posterior, torchopt.adam(), temperature=1 / num_data
)  # Can swap out for any posteriors algorithm

state = transform.init(params)

for batch in train_loader:
    state, aux = transform.update(state, batch)
```
Here:

- `build` is a function that loads `config_args` into the `init` and `update` functions
 and stores them within the `transform` instance. The `init` and `update` 
 functions then conform to a preset signature allowing for easy switching between algorithms.
- `state` is a [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)
    encoding the state of the algorithm, including `params` and `aux` attributes.
- `init` constructs the iteration-varying `state` based on the model parameters `params`.
- `update` updates the `state` based on a new `batch` of data and also returns
   any auxiliary information from the model call.


We've here used `posteriors.vi.diag` but we could easily swap to any of the other
`posteriors` algorithms such as `posteriors.laplace.diag_fisher` or
`posteriors.sgmcmc.sghmc`


!!! example "I want more!"

    The [Visualizing VI and SGHMC](tutorials/visualizing_vi_sghmc.md) tutorial provides
    a walkthrough for a simple example demonstrating how to use `posteriors` and easily
    switch between algorithms.

    `posteriors` expects `log_posterior` to take a certain form, learn more in the 
    [constructing log posteriors page](log_posteriors.md).

    Our [API documentation](api/index.md) provides detailed descriptions for all
    of the `posteriors` algorithms and utilities.


## PyTrees

The internals of `posteriors` rely on [`optree`](https://optree.readthedocs.io/en/latest/) to
apply functions across arbitrary PyTrees of tensors (i.e. TensorTrees). For example:
```py
params_squared = optree.tree_map(lambda x: x**2, params)
```
will square all the tensors in the `params`, where `params` can be a 
`dict`, `list`, `tuple`, or any other [PyTree](https://github.com/metaopt/optree?tab=readme-ov-file#built-in-pytree-node-types).

`posteriors` also provides a [`posteriors.flexi_tree_map`][] function that allows for in-place support:
```py
params_squared = optree.flexi_tree_map(lambda x: x**2, params, inplace=True)
```
In this case, the tensors of params are modified in-place, without assigning extra memory.


## [`torch.func`](https://pytorch.org/docs/stable/func.html)

Instead of using `torch`'s more common `loss.backward()` style automatic differentiation,
`posteriors` uses a functional approach, via `torch.func.grad` and friends. The functional 
approach is easier to test, composes better with other tools and importantly for `posteriors` 
it makes for code that is closer to the mathematical notation.

For example, the gradient of a function `f` with respect to `x` can be computed as:
```py
grad_f_x = torch.func.grad(f)(x)
```
where `f` is a function that takes `x` as input and returns a scalar output. Again, 
`x` can be a `dict`, `list`, `tuple`, or any other PyTree with `torch.Tensor` leaves.


## Friends

Compose `posteriors` with wonderful tools from the `torch` ecosystem

- Define priors and likelihoods with [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html).  
    Remember to set [`validate_args=False`](gotchas.md#validate_argsfalse-in-torchdistributions)
    and [construct the log posterior](log_posteriors.md) accordingly.
- [`torchopt`](https://github.com/metaopt/torchopt) for functional optimizers.
- [`transfomers`](https://huggingface.co/docs/transformers/en/index) for open source models.
- [`lightning`](https://pytorch-lightning.readthedocs.io/en/latest/) for logging and device management.  
    Check out the [`lightning` integration tutorial](tutorials/lightning_autoencoder.md).

Additionally, the functional transform interface used in `posteriors` is strongly
inspired by frameworks such as [`optax`](https://github.com/google-deepmind/optax) and
[`blackjax`](https://github.com/blackjax-devs/blackjax).

As well as other UQ libraries [`fortuna`](https://github.com/awslabs/fortuna),
[`laplace`](https://github.com/aleximmer/Laplace), [`numpyro`](https://github.com/pyro-ppl/numpyro),
[`pymc`](https://github.com/pymc-devs/pymc) and [`uncertainty-baselines`](https://github.com/google/uncertainty-baselines).
