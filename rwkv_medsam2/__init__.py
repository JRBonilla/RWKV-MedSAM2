"""RWKV-MedSAM2 package exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vcr import VCRBackbone

__all__ = ["VCRBackbone"]


def __getattr__(name):
    """
    Load the backbone lazily when it is requested from the package.

    Args:
        name (str): Package attribute to resolve; only VCRBackbone is exported lazily.

    Returns:
        type[VCRBackbone]: Backbone class exposed as a package attribute.
    """
    if name == "VCRBackbone":
        from .vcr import VCRBackbone

        return VCRBackbone

    raise AttributeError(name)
