# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components for the FoundAD model.

This module provides the manifold projector and CutPaste augmentation components
used in the FoundAD architecture.
"""

from .cutpaste import CutPasteUnion
from .manifold_projector import ManifoldProjector

__all__ = ["CutPasteUnion", "ManifoldProjector"]
