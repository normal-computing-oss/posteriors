If you want to add a new algorithm, example, feature, report or fix a bug, please open 
an [issue on GitHub](https://github.com/normal-computing-oss/posteriors/issues). 
We'd love to have you involved in any capacity!

If you are interested in contributing to `posteriors`, please follow these steps:

1. [Fork the repo from GitHub](https://github.com/normal-computing-oss/posteriors/fork)
and clone it locally:
```
git clone git@github.com/YourUserName/posteriors.git
cd posteriors
```
2. Install the development dependencies and pre-commit hooks:
```
pip install -e '.[test, docs]'
pre-commit install
```
3. **Add your code. Add your tests. Update the docs if needed. Party on!**  
    New methods should list `build`, `state`, `init` and `update`
    at the top of the module in order.
4. Check any changes in the docs render nicely:
```
mkdocs serve
```
and navigate to [`http://localhost:8000/`](`http://localhost:8000/`) in your browser.[^1]
5. Make sure to run the linter, tests and check coverage:
```
pre-commit run --all-files
python -m pytest --cov=posteriors --cov-report term-missing
```
6. Commit your changes and push your new branch to your fork.
7. Open a [pull request on GitHub](https://github.com/normal-computing-oss/posteriors/pulls).

[^1]: For more docs info check out [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).


!!! note
    Feel free to open a draft PR to discuss changes or get feedback.