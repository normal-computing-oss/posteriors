# Transformer UQ with Yelp

Here we demonstrate how to use `posteriors` methods in training a transformer model
 on the [Yelp dataset](https://huggingface.co/datasets/yelp_review_full).

We demonstrate that Bayesian UQ techniques for the last layer of a transformer 
can provide improved predictions as well as giving us valuable information about
what it doesn't know in the face of out of distribution data.


## Methods

We start by fine-tuning the [bert-base-cased](https://huggingface.co/bert-base-cased) 
model on a subset (1000 reviews) from the Yelp dataset (following the 
[Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch)). 
We then use `posteriors` to get uncertainty quantification over the parameters of the last 
layer and then investigate how this affects predictions. The methods we compare are:

- **map**: Maximum a posteriori (MAP) estimate of the parameters. Simple optimization 
using [AdamW](https://arxiv.org/abs/1711.05101) via `posteriors.torchopt`.
- **sghmc**: [Stochastic gradient Hamiltonian Monte Carlo](https://proceedings.mlr.press/v32/cheni14.pdf) 
using `posteriors.sgmcmc.sghmc`. SGHMC approximately simulates a trajectory that has the 
Bayesian posterior as its stationary distribution, meaning that taking spaced samples 
along the trajectory gives approximate samples from the posterior. We run for 100 epochs 
and take 14 samples from the trajectory after a burnin.
- **sghmc parallel**: Also with `posteriors.sgmcmc.sghmc` but this time we run 10 
trajectories in parallel and take the last value from each as our approximate posterior 
samples - [deep ensemble](https://arxiv.org/abs/1612.01474) style.
- **vi**: Variational inference using `posteriors.vi.diag` to get a diagonal Gaussian 
approximation to the posterior. Test predictions are calculated by generating 10 samples
from the approximate posterior running each through the last layer and then averaging.
- **vi linearised**: Training is the same as above with `posteriors.vi.diag`, however 
predictions use `posteriors.linearized_forward_diag` to get uncertainty directly in the logit
space, for more info on linearized neural networks see section 2 of 
[Laplace Redux](https://arxiv.org/abs/2106.14806).
Since the logit space is much smaller we can use a much larger 1000 samples .


## Results

We first show the test loss for each method in Figure 1. We can see that the 
uncertainty methods all have a lower cross entropy loss (lower loss is better) than the 
MAP estimate which does not use UQ. In this setting of relatively little training data 
the uncertainty methods are able to provide more robust predictions by averaging over 
multiple plausible parameter configurations.

<p align="center">
  <img src="https://storage.googleapis.com/posteriors/yelp_loss.png" width=65%">
  <br>
  <em>Figure 1. Test loss on Yelp data (lower is better).</em>
</p>

In particular, the linearized VI method is particularly efficient as in this case the 
last layer of the transformer is exactly linear. In cases, where the logit space is 
larger and the prediction function is non-linear we might expect SGHMC to 
become more efficient as it makes no linear or Gaussian assumptions.


We then turn to the case of out of distribution (OOD) data, by testing how the methods 
handle Spanish reviews. In this case, we do not expect the model to perform particularly
well without seeing a lot of Spanish in its training data. However, we would like some 
indication that the model is struggling more with these reviews than the in distribution
English ones.


<p align="center">
    <img src="https://storage.googleapis.com/posteriors/yelp_uncertainty.png" width=45%">
    <img src="https://storage.googleapis.com/posteriors/yelp_spanish_uncertainty.png" width=45%">
    <br>
    <em>Figure 2. Left: Uncertainty on English test data. Right: Uncertainty on Spanish test data (OOD). <br />
     We'd like low epistemic uncertainty on familiar topics e.g. English
     and higher epistemic uncertainty on unfamiliar topics e.g. Spanish. Aleatoric uncertainty represents natural
     ambiguity also present in the training data which we do not desire to be high or low, only to fairly represent
     natural uncertainty in the reviews.</em>
</p>

In Figure 2 we show the uncertainty of the methods on the English test 
data and the Spanish test data. Total uncertainty is the entropy of the 5-class predictive 
distribution (averaged over each input in the test set). Given a distribution over the 
logits, total uncertainty can be broken
down into two components: aleatoric uncertainty (intrinsic to the data) and epistemic
uncertainty (due to lack of knowledge).

Aleatoric uncertainty might arise from a review that is genuinely ambiguous such as 
"The food was amazing! But the service was horrendous!". Even with an infinite amount of
training data there would still be uncertainty in predicting the label. Epistemic 
uncertainty is the difference between the total uncertainty and the aleatoric
uncertainty and converges to zero as the amount of training data goes to infinity.

Baseline optimisation methods (MAP) do not provide a distribution over logits and 
therefore lack the ability to capture epistemic and aleatoric uncertainty. For detailed 
information on the breakdown of uncertainty for machine learning, as well as the 
caveats, see [Wimmer et al](https://arxiv.org/abs/2209.03302).

In Figure 2, we can see that the approximate Bayesian methods capture both types of 
uncertainty. In particular, we see that for 
the Spanish reviews the total uncertainty increases (as we would expect) and that
this is mostly due to an increase in epistemic uncertainty. This is great! The model
is telling us that it doesn't know the answer here and would like to receive more
training data on Spanish reviews.


### Laplace approximation?

`posteriors` indeed also provides the tools to seamlessly apply a Laplace approximation using 
the empirical Fisher information matrix. However, in this smaller data, 
overparameterized setting, the MAP estimate fits very well and essentially all of the 
gradients in the training data are close to zero, that is 
$\nabla_\theta \log p(y_n \mid x_n, \theta) \approx 0$ for all input, output pairs 
$(x_n, y_n)$ in the training data. This means that the empirical Fisher 
information matrix, $\hat{F} = \sum_{n} \nabla_\theta \log p(y_n \mid x_n, \theta) \nabla_\theta \log p(y_n \mid x_n, \theta)^T$, is extremely small and fails to provide useful uncertainty quantification. In 
our case, all values of the diagonal empirical Fisher information matrix
were less than $10^{-22}$. For more information on the empirical Fisher information 
matrix and how its use is only advised in large data settings, see [Kunstner et al](https://arxiv.org/abs/1905.12558).
Additionally, the [continual lora example](https://github.com/normal-computing-oss/posteriors/tree/main/examples/continual_lora)
provides an in depth example of how the empirical Fisher Laplace approximation can be 
used in a practical continual learning setting.


## Data

Both the Yelp and Spanish reviews are available from the Hugging Face datasets library 
at [yelp_review_full](https://huggingface.co/datasets/yelp_review_full) and 
[beltrewilton/punta-cana-spanish-reviews](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews).



## Model
We use the [bert-base-cased](https://huggingface.co/bert-base-cased) model adapted from
the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).



## Code structure

- `load.py`: Functions to load the Yelp and Spanish reviews. As well as the model and an
    associated `posteriors` log posterior function.
- `train.py`: Small and general script for training the model using `posteriors` methods 
    which can easily be swapped.
- `configs/` : Configuration files (python files) for the different methods.
- `combine_states.py`: Combines single parameter states (of the serial or parallel 
    SGHMC runs).
- `test.py`: Calculate metrics on either yelp or Spanish test data for a given method.
- `plot.py`: Plot the results.
- `utils.py`: Small utility functions for logging.


