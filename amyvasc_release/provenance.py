"""Versions recorded with locally generated analysis outputs."""

from importlib.metadata import version
from platform import python_version


def software_versions() -> dict[str, str]:
    """Record Python and the six declared analysis dependencies."""
    return {
        "python": python_version(),
        **{
            package: version(package)
            for package in ("matplotlib", "nibabel", "nilearn", "numpy", "pandas", "scipy")
        },
    }
