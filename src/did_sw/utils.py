"""
Sorry for calling this module for `utils`.
"""

from pathlib import Path

__all__ = []


def proj_folder() -> Path:  # pragma: no cover
    """Returns the project folder."""
    fp = Path(__file__).parents[2]
    if not fp.joinpath("src", "did_sw").exists():
        raise FileNotFoundError(
            "Could not find the project folder. "
            "This function only works on the development version when cloning "
            "the repository from github"
        )
    return fp
