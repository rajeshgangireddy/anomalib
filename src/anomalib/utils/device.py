# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device related utility functions."""

import torch

# Set of xpu devices that might not support all funtions/operations out of the box
# Update this set as we identify more unsupported devices
# Raptor Lake can have two different iGPUs, Iris Xe or UHD 770 (desktop RPL-S)
UNSUPPORTED_XPU: frozenset = frozenset({"Intel(R) Iris(R) Xe Graphics", "Intel(R) UHD Graphics 770"})


def is_unsupported_xpu() -> bool:
    """Check if the current device is in the unsupported XPU set."""
    if not torch.xpu.is_available():
        return False
    return any(device in torch.xpu.get_device_name() for device in UNSUPPORTED_XPU)
