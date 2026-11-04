# Tutorials

We list here a view small examples, that are easy to run locally on a CPU, 
to show how to use `posteriors`:


- [Visualizing VI and SGHMC](visualizing_vi_sghmc.md)
[<img style="float: right; width: 5em;" src="https://storage.googleapis.com/posteriors/double_well_compare.png">](visualizing_vi_sghmc.md)
<br><br><br>
- [EKF for Premier League football](ekf_premier_league.md)
[<img style="float: right; width: 5em;" src="https://em-content.zobj.net/source/telegram/386/soccer-ball_26bd.webp">](ekf_premier_league.md)
<br><br><br>
- [Autoencoder with Lightning](lightning_autoencoder.md)
[<img style="float: right; width: 7em" src="https://storage.googleapis.com/posteriors/lightning_posteriors.png">](lightning_autoencoder.md)




Additionally, we provide more in depth examples in the [examples directory on GitHub](https://github.com/normal-computing-oss/posteriors/tree/main/examples),
many of which use larger models that require a GPU to run:

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/bayes_llama_uncertainties_appendix.png">](https://github.com/normal-computing-oss/posteriors/tree/main/examples/bayes_llama3)

- [`bayes_llama3`](https://github.com/normal-computing-oss/posteriors/tree/main/examples/bayes_llama3):
Uses `posteriors` and `lightning` to train a bayesian ensemble language model [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
on the [TQA](https://allenai.org/data/tqa) dataset.

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/plot_B_laplace.png">](https://github.com/normal-computing-oss/posteriors/tree/main/examples/continual_lora)

- [`continual_lora`](https://github.com/normal-computing-oss/posteriors/tree/main/examples/continual_lora):
Uses [`posteriors.laplace.diag_fisher`](../api/laplace/diag_fisher.md) to avoid catastrophic forgetting in fine-tuning
[Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
on a series of books from the [pg19](https://huggingface.co/datasets/pg19) dataset.

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/yelp_spanish_uncertainty.png">](https://github.com/normal-computing-oss/posteriors/tree/main/examples/yelp)

- [`yelp`](https://github.com/normal-computing-oss/posteriors/tree/main/examples/yelp):
Compares a host of `posteriors` methods (highlighting the easy exchangeability) on a
sentiment analysis task adapted from the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/cold_posterior_sghmc_loss.png">](https://github.com/normal-computing-oss/posteriors/tree/main/examples/imdb)

- [`imdb`](https://github.com/normal-computing-oss/posteriors/tree/main/examples/imdb): Investigates [cold posterior effect](https://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf) for a range of approximate
Bayesian methods with a CNN-LSTM model on [IMDB](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data)
data, with some interesting takeaways.


[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/variational_continual_learning.png">](https://github.com/normal-computing-oss/posteriors/blob/main/examples/continual_regression.ipynb)

- [`continual_regression`](https://github.com/normal-computing-oss/posteriors/blob/main/examples/continual_regression.ipynb):
[Variational continual learning](https://arxiv.org/abs/1710.10628) notebook for a simple
regression task that's easy to visualize.


[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/pyro_posteriors.png">](https://github.com/normal-computing-oss/posteriors/blob/main/examples/pyro_pima_indians_vi.ipynb)

- [`pyro_pima_indians_sghmc`](https://github.com/normal-computing-oss/posteriors/blob/main/examples/pyro_pima_indians_sghmc.ipynb):
An accessible notebook demonstrating the use of `posteriors` with a [`pyro`](https://pyro.ai)-defined Bayesian logistic regression model, as well as convergence diagnostics from [`pyro`](https://pyro.ai).

