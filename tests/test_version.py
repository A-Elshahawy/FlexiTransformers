"""Tests for package version and public API surface."""

import flexit


def test_version_is_string() -> None:
    assert isinstance(flexit.__version__, str)


def test_version_format() -> None:
    parts = flexit.__version__.split('.')
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_version_matches_pyproject() -> None:
    """Version in version.py should match pyproject.toml."""
    from pathlib import Path

    import tomllib

    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    if pyproject.exists():
        with open(pyproject, 'rb') as f:
            data = tomllib.load(f)
        assert flexit.__version__ == data['project']['version']


def test_author_present() -> None:
    assert flexit.__author__


def test_all_exports_importable() -> None:
    """Every name in __all__ should be importable from the top-level package."""
    for name in flexit.__all__:
        assert hasattr(flexit, name), f'flexit.{name} missing from package'
