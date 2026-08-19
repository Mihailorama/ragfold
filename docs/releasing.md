# Releasing ragfold to PyPI

The `ragfold` name is unclaimed on PyPI (checked 2026-08-19). Publishing is
automated via `.github/workflows/publish.yml` using PyPI trusted publishing,
so no API tokens are stored in the repository.

## One-time setup

1. Log in to <https://pypi.org> with the account that will own the package.
2. Go to **Your account → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `ragfold`
   - Owner: `mihailorama`
   - Repository: `ragfold`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In the GitHub repository, create an environment named `pypi`
   (**Settings → Environments**). Optionally add required reviewers so every
   release needs a manual approval.

## Cutting a release

1. Update the version in `pyproject.toml` and move the `[Unreleased]` section
   of `CHANGELOG.md` under the new version heading.
2. Verify locally:

   ```bash
   pip install build twine
   python -m build
   twine check dist/*
   ```

3. Commit, tag (`git tag v0.1.0`), and push the tag.
4. Create a GitHub release from the tag. Publishing the release triggers the
   workflow, which builds the sdist and wheel and uploads them to PyPI.

The `workflow_dispatch` trigger runs the build job only (a dry run that
produces the distributions as a downloadable artifact without publishing).
