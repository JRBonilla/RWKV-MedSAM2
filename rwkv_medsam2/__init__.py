"""RWKV-MedSAM2 package exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vcr import VCRBackbone

__all__ = ['VCRBackbone']


def __getattr__(name):
    if name == "VCRBackbone":
        from .vcr import VCRBackbone

        return VCRBackbone
    raise AttributeError(name)
