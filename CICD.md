# CI / CD

## Continuous Integration

We perform continuous integration (CI) through GitHub Actions. We created 
two separate workflows to check our code both for functionality and for 
style/language.

> [!NOTE]
> Due to lack of time as we're Ph.D. candidates, we could not adapt a 
> very regular and clean "GitFlow" workflow. A lot of work was done in the 
> feature branch (in reality it was called cicd to follow the milestones PR guidelines) and the 
> tests were therefore also run for every push to this branch. In a more regular and 
> realistic scenarios, we would not run tests on the very active 
> development (i.e., feature) branches, but only on the main and development 
> branches, as feature branches are (possibly) expected to temporarily break 
> tests.

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
workflow automatically deploys the latest version of our Flask API, if it 
passes the other styles, to Google Cloud Run.

TODO