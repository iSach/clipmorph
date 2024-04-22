# CI / CD

## Continuous Integration

We perform continuous integration (CI) through GitHub Actions. We created 
two separate workflows to check our code both for functionality and for 
style/language.

### Core Tests

> *Workflow file: `.github/workflows/core-tests.yml`*

The core tests are run on every push to the repository and the development 
branch. We use [PyTest](https://docs.pytest.org/) to easily create unit tests.


> Note: due to lack of time as we're Ph.D., we could not adapt a very 
> regular gitflow workflow. A lot of work was done in the feature branch (in 
> reality was called cicd to respect the PR milestones) and the tests were 
> therefore also run for every push to this branch. In a more regular and 
> realistic scenarios, we would not run tests on the very active 
> development (i.e., feature) branches, but only on the main and development 
> branches, as feature branches are expected to temporarily break tests.