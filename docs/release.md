# Releasing Kodiak to PyPI

Kodiak is published as the `kodiak-ai` distribution. Its Python import package and console command
remain `kodiak`.

## Prerequisites

Use Python 3.12 or newer in a clean virtual environment. Confirm that the version in
`kodiak/__init__.py` is the intended release version and that the repository passes CI.

Install the packaging tools:

```console
python -m pip install --upgrade build twine
```

## Build and validate

Build both the source distribution and wheel, then validate their metadata:

```console
python -m build
python -m twine check dist/*
```

Before uploading, install the wheel in a clean environment and verify the public CLI:

```console
pip install dist/*.whl
kodiak --help
kodiak doctor
```

## TestPyPI

Upload the validated artifacts to TestPyPI:

```console
python -m twine upload --repository testpypi dist/*
```

Install the uploaded distribution and verify it:

```console
pip install -i https://test.pypi.org/simple/ kodiak-ai
kodiak --help
kodiak doctor
```

If TestPyPI cannot resolve a runtime dependency, use PyPI as the supplemental dependency index:

```console
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kodiak-ai
```

## PyPI

After TestPyPI verification succeeds, upload the same validated artifacts to PyPI:

```console
python -m twine upload dist/*
```

Verify the public installation in a new virtual environment:

```console
pip install kodiak-ai
kodiak --help
kodiak doctor
kodiak analyze analyze . --deep
kodiak task run "Improve the README quickstart section" --path . --dry-run
```

Do not upload from CI yet. Publishing remains an explicit maintainer action.
