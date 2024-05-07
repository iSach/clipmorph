# CI / CD

## Continuous Integration

We perform continuous integration (CI) through GitHub Actions. We created 
two separate workflows to check our code both for functionality and for 
style/language.

These tests are run for every commit pushed to the `master` and `dev` branches. They
are also run for every submitted pull request, to avoid breaking the codebase when
merging.

> [!NOTE]
> We tried to adopt a "GitFlow" workflow, but as we are Ph.D. candidates, this
> was not very easy to do, and we started to use this late. Therefore, many
> commits were done directly in the `dev` branch. However, we've used some
> feature branches, such as for deploying to GCR and including UX changes.

### Core Tests

> *Workflow file: [`.github/workflows/clipmorph_tests.yml`](https://github.com/iSach/clipmorph/actions/workflows/clipmorph_tests.yml)*

The core tests are run on every push to the repository and the development 
branch. We use [PyTest](https://docs.pytest.org/) to easily create unit 
tests. In the `tests/` folder, you can find all our test files, which 
verify that the overall functionality of our codebase is correct and that 
committed changes have not broken some parts.

### Style Tests

> *Workflow file: 
> [`.github/workflows/code_style.yml`](https://github.com/iSach/clipmorph/actions/workflows/code_style.yml)*

We check for the style and language correctness of our codebase using
[Ruff](https://docs.astral.sh/ruff/), a popular and recent alternative to 
linters like [PyLint](https://github.com/pylint-dev/pylint) and
[Flake8](https://github.com/pycqa/flake8), and formatters like
[Black](https://pypi.org/project/black/). It is much faster and provides 
both linting and formatting checks. 

With these checks, we ensure a codebase that has a consistent style, and 
does not use deprecated, unrecommended, or dangerous language features.

## Continuous Deployment

We perform continuous deployment (CD) through GitHub Actions as well. Our 
workflow automatically deploys the latest version of our Flask API web app. 
We configured the workflow through the Google Cloud console interface 
directly to handle authentication more easily. The trigger is enabled for 
every push on `master`, as well as our development branch `dev` for the purpose of the project.

Thanks to that, the app is always up-to-date at 
[https://clipmorph.isach.be](https://clipmorph.isach.be).

Please find more information about our deployment choices in [DEPLOYMENT.md](DEPLOYMENT.md).
