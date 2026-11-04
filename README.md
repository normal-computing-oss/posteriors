<div align="center">
<img src="https://storage.googleapis.com/posteriors/logo_with_text.png" alt="logo"></img>
</div>

[**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Methods**](#methods)
| [**Friends**](#friends)
| [**Contributing**](#contributing)
| [**Citation**](#citation)
| [**Documentation**](https://normal-computing-oss.github.io/posteriors/)
| [**Paper**](https://arxiv.org/abs/2406.00104)

## What is `posteriors`?

General purpose python library for uncertainty quantification with [`PyTorch`](https://github.com/pytorch/pytorch).

- [x] **Composable**: Use with [`transformers`](https://huggingface.co/docs/transformers/en/index), [`lightning`](https://lightning.ai/), [`torchopt`](https://github.com/metaopt/torchopt), [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html), [`pyro`](https://pyro.ai/) and more!
- [x] **Extensible**: Add new methods! Add new models!
- [x] **Functional**: Easier to test, closer to mathematics!
- [x] **Scalable**: Big model? Big data? No problem!
- [x] **Swappable**: Swap between algorithms with ease!


## Installation

`posteriors` is available on [PyPI](https://pypi.org/project/posteriors/) and can be installed via `pip`:

```bash
pip install posteriors
```

## Quickstart

`posteriors` is functional first and aims to be easy to use and extend. Let's try it out
by training a simple model with variational inference:
```python
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import nn, utils, func
import torchopt
import posteriors

dataset = MNIST(root="./data", transform=ToTensor())
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

Observe that `posteriors` recommends specifying `log_posterior` and `temperature` such that 
`log_posterior` remains on the same scale for different batch sizes. `posteriors` 
algorithms are designed to be stable as `temperature` goes to zero.

Further, the output of `log_posterior` is a tuple containing the evaluation 
(single-element Tensor) and an additional argument (TensorTree) containing any 
auxiliary information we'd like to retain from the model call, here the model predictions.
If you have no auxiliary information, you can simply return `torch.tensor([])` as
the second element. For more info see [`torch.func.grad`](https://pytorch.org/docs/stable/generated/torch.func.grad.html) 
(with `has_aux=True`) or the [documentation](https://normal-computing-oss.github.io/posteriors/log_posteriors).

Check out the [tutorials](https://normal-computing-oss.github.io/posteriors/tutorials) for more detailed usage!

## Methods

`posteriors` supports a variety of methods for uncertainty quantification, including:

- [**Extended Kalman filter**](posteriors/ekf/)
- [**Laplace approximation**](posteriors/laplace/)
- [**Stochastic gradient MCMC**](posteriors/sgmcmc/)
- [**Variational inference**](posteriors/vi/)

With full details available in the [API documentation](https://normal-computing-oss.github.io/posteriors/api).

`posteriors` is designed to be easily extensible, if you're favorite method is not listed above,
[raise an issue](https://github.com/normal-computing-oss/posteriors/issues) and we'll see what we can do!


## Friends

Interfaces seamlessly with:

- [`torch`](https://github.com/pytorch/pytorch) and in particular [`torch.func`](https://pytorch.org/docs/stable/func.html).
- [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) for distributions and sampling, (note that it's typically required to set `validate_args=False` to conform with the control flows in [`torch.func`](https://pytorch.org/docs/stable/func.html)).
- Functional and flexible torch optimizers from [`torchopt`](https://github.com/metaopt/torchopt).
- [`transformers`](https://github.com/huggingface/transformers) for pre-trained models.
- [`lightning`](https://github.com/Lightning-AI/lightning) for convenient training and logging, see [examples/lightning_autoencoder.py](examples/lightning_autoencoder.py).

The functional transform interface is strongly inspired by frameworks such as 
[`optax`](https://github.com/google-deepmind/optax) and [`blackjax`](https://github.com/blackjax-devs/blackjax).

As well as other UQ libraries [`fortuna`](https://github.com/awslabs/fortuna),
[`laplace`](https://github.com/aleximmer/Laplace), [`numpyro`](https://github.com/pyro-ppl/numpyro),
[`pymc`](https://github.com/pymc-devs/pymc) and [`uncertainty-baselines`](https://github.com/google/uncertainty-baselines).


## Contributing

You can report a bug or request a feature by [creating a new issue on GitHub](https://github.com/normal-computing-oss/posteriors/issues).


If you want to contribute code, please check the [contributing guide](https://normal-computing-oss.github.io/posteriors/contributing).


## Citation

If you use `posteriors` in your research, please cite the library using the following BibTeX entry:

```bibtex
@inproceedings{duffield2025scalable,
    title={Scalable Bayesian Learning with posteriors},
    author={Samuel Duffield and Kaelan Donatella and Johnathan Chiu and Phoebe Klett and Daniel Simpson},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=fifXzmzeGy}
}
```
