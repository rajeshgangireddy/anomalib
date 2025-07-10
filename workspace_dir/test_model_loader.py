#!/usr/bin/env python3
"""Simple test script for the improved model loader."""

import sys
from pathlib import Path

# Add the source directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from anomalib.models.image.dinomaly.model_loader import ModelLoader, load


def test_model_loader():
    """Test the model loader functionality."""
    
    # Test parsing different model names
    loader = ModelLoader()
    
    test_cases = [
        "dinov2_vit_base_14",
        "dinov2_vit_small_14", 
        "dinov2reg_vit_base_14",
        "dinov2_reg_vit_large_14"
    ]
    
    for model_name in test_cases:
        try:
            model_type, architecture, patch_size = loader._parse_name(model_name)
            print(f"✓ {model_name} -> {model_type}, {architecture}, {patch_size}")
        except Exception as e:
            print(f"✗ {model_name} -> Error: {e}")
    
    # Test invalid names
    invalid_cases = [
        "invalid_name",
        "dinov2_vit_unknown_14",
        "dinov2_vit_base"
    ]
    
    for model_name in invalid_cases:
        try:
            loader._parse_name(model_name)
            print(f"✗ {model_name} -> Should have failed but didn't")
        except Exception:
            print(f"✓ {model_name} -> Correctly rejected")


if __name__ == "__main__":
    test_model_loader()
    print("\nModel loader tests completed!")
