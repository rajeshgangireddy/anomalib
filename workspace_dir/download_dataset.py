#!/usr/bin/env python3
"""Simple script to download Kaputt defect dataset files."""

from pathlib import Path

import requests
from tqdm import tqdm

# Dataset files with their URLs and sizes
DATASET_FILES = [
    (
        "LICENSE",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/LICENSE?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=031d302698c0a8fd34ec7f55b9b29c60b01d336efae25e1ad2269cbdd3ab77b3",
        0.02,
    ),
    (
        "datasets.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/datasets.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e876e80d54122e04bb62d94e411d89b0e08f5342b222ff8b168770d8aeb9ac53",
        22.09,
    ),
    (
        "query-crop.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/query-crop.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=2b22d341896878439970eb5508677bdb1866facb8b618a96c61332f620a8a638",
        30101.78,
    ),
    (
        "query-image.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/query-image.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=b5d93dd6c8e7bf2cfeafe63b1a2edfacdf2407a8a02ff8242f6982dd016ddb82",
        101821.74,
    ),
    (
        "query-mask.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/query-mask.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=8eef246db42facceb42c2b056641828e2d1c30319cc1b7a167dd960f9bb3c37b",
        1092.62,
    ),
    (
        "reference-crop.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/reference-crop.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=429e386f1ee343a0990d557660de473b94e3192fe5216e9d41cf143dea2ea388",
        39664.97,
    ),
    (
        "reference-image.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/reference-image.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=03c39b8248050a22c643272cad6cd84a52d510204074fe55c1d50deb9aaa3503",
        138064.79,
    ),
    (
        "reference-mask.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/reference-mask.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=86c5abb413e4db28d998d00f9c544279c5a3ae4a9a196b45a4f286940fa56868",
        1496.97,
    ),
    (
        "sample-data.tar.gz",
        "https://kaputt-defect-dataset-all-data.s3.amazonaws.com/kaputt-release/sample-data.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASFUIRLP746Y77H4O%2F20251008%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T091714Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=0402ad416f0f5feb01e09889b01fe1c3557d41687c76a83ab77413463fac0d62",
        501.56,
    ),
]


def download_file(url: str, filename: str, download_dir: Path) -> bool:
    """Download a file with progress bar."""
    filepath = download_dir / filename

    # Skip if file already exists
    if filepath.exists():
        print(f"{filename} already exists, skipping...")
        return True

    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Downloaded {filename}")
        return True

    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        # Remove partial file if it exists
        if filepath.exists():
            filepath.unlink()
        return False


def main():
    """Main download function."""
    # Create download directory
    download_dir = Path("/home/devuser/workspace/code/Anomalib/dev/anomalib/datasets/Kaputt")
    download_dir.mkdir(exist_ok=True)

    print(f"Starting download to {download_dir.absolute()}")
    print(f"Total files: {len(DATASET_FILES)}")
    total_size = sum(size for _, _, size in DATASET_FILES)
    print(f"Total size: {total_size:.2f} MB")
    print("-" * 50)

    downloaded_files = []
    failed_files = []

    # Download all files
    for filename, url, size_mb in DATASET_FILES:
        print(f"\n[{len(downloaded_files) + len(failed_files) + 1}/{len(DATASET_FILES)}] {filename} ({size_mb} MB)")

        if download_file(url, filename, download_dir):
            downloaded_files.append(filename)
        else:
            failed_files.append(filename)

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Successfully downloaded: {len(downloaded_files)}")
    print(f"Failed downloads: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for filename in failed_files:
            print(f"  - {filename}")

    if downloaded_files:
        print(f"\nFiles saved to: {download_dir.absolute()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
