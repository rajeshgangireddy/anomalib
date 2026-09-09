#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Time image *editing* (the donor-generation step) end-to-end.

The donor step conditions on a normal image and asks the model to add a specific defect.
This script times a batch of such calls so we can fill in the "(manual)" row of
``gsoc_workspace/synthetic_generation_timing.md`` with a measured number.

Two providers:
    Gemini (default):  pip install google-genai pillow ; export GEMINI_API_KEY=...
    OpenAI:            pip install openai pillow         ; export OPENAI_API_KEY=...
                       (run with ``--provider openai``)

Gemini model note: the default is ``gemini-3.1-flash-image`` ("Nano Banana 2", the general
workhorse). Pass ``--model gemini-3.1-flash-lite-image`` for the fastest/cheapest tier, or
``--model gemini-3-pro-image`` for the highest-quality tier. OpenAI image editing uses
``gpt-image-1`` by default.

The Gemini interactions input schema follows
https://ai.google.dev/gemini-api/docs/image-generation (text-and-image-to-image). If your
installed SDK version differs, adjust the ``input`` block accordingly -- the timing wrapper
around the API call is the part that matters.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import statistics
import time

from PIL import Image


def generate_defect_gemini(client, model: str, normal_path: str, prompt: str, out_path: str) -> float:
    """Generate one edited image via Gemini and return wall-clock seconds for the API call."""
    with open(normal_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "image", "data": img_b64, "mime_type": "image/png"},
            {"type": "text", "text": prompt},
        ],
    )
    dt = time.perf_counter() - t0

    out = base64.b64decode(interaction.output_image.data)
    Image.open(io.BytesIO(out)).save(out_path)
    return dt


def generate_defect_openai(client, model: str, normal_path: str, prompt: str, out_path: str) -> float:
    """Generate one edited image via OpenAI and return wall-clock seconds for the API call."""
    t0 = time.perf_counter()
    with open(normal_path, "rb") as img_file:
        resp = client.images.edit(model=model, image=img_file, prompt=prompt, n=1)
    dt = time.perf_counter() - t0

    data = resp.data[0]
    if data.b64_json:
        Image.open(io.BytesIO(base64.b64decode(data.b64_json))).save(out_path)
    else:
        print(f"    (url result, not saved: {data.url})")
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None, help="Image-editing model (provider-dependent).")
    parser.add_argument("--normal", required=True, help="Normal image to condition on.")
    parser.add_argument("--prompt", required=True, help="Edit instruction, e.g. 'add a crack to this can'.")
    parser.add_argument("--n", type=int, default=4, help="Number of images to generate for the timing average.")
    parser.add_argument("--out-dir", default="/tmp/gemini_timing")
    args = parser.parse_args()

    if args.provider == "gemini":
        from google import genai  # local import so `--help` works without the SDK
        model = args.model or "gemini-3.1-flash-image"
        client = genai.Client()  # reads GEMINI_API_KEY / ADC
        fn = generate_defect_gemini
    else:
        from openai import OpenAI  # local import
        model = args.model or "gpt-image-1"
        client = OpenAI()  # reads OPENAI_API_KEY
        fn = generate_defect_openai

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[provider={args.provider}, model={model}, n={args.n}]")

    times = []
    for i in range(args.n):
        out_path = os.path.join(args.out_dir, f"gen_{i}.png")
        t0 = time.perf_counter()
        dt = fn(client, model, args.normal, args.prompt, out_path)
        times.append(dt)
        print(f"[{i + 1}/{args.n}] {dt:.2f}s -> {out_path}")

    print(
        f"\n{args.n} images: mean={statistics.mean(times):.2f}s "
        f"(std={statistics.stdev(times):.2f}s, min={min(times):.2f}, max={max(times):.2f})"
    )


if __name__ == "__main__":
    main()
