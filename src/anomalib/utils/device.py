# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device related utility functions."""

import torch

# Set of xpu devices that might not support all funtions/operations out of the box
UNSUPPORTED_XPU: frozenset = frozenset({"Intel(R) Iris(R) Xe Graphics"})


def is_unsupported_xpu() -> bool:
    """Check if the current device is in the unsupported XPU set."""
    if not torch.xpu.is_available():
        return False
    return any(device in torch.xpu.get_device_name() for device in UNSUPPORTED_XPU)
