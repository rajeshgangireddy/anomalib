# Synthetic-Anomaly Generation Timing

Wall-clock cost of generating the synthetic anomaly images used for threshold
calibration, measured for the three methods we compare: **stock Perlin (P0)**, **AnoStyler
(P8, AAAI 2026)**, and **our semantic-defect-bank hybrid (P6, `flash-part2.ipynb`)**.

All measurements are per-image (image + mask where applicable), on a single NVIDIA RTX 3090
(24 GiB), averaged over fresh, non-cached images. **Time / image is reported in
milliseconds; category time is reported in seconds** for a representative 90-image category
(MVTec AD 2 categories hold 60–105 real test anomalies).

## Summary

| method | native resolution | time / image (ms) | category time (s, ~90 img) | emits mask? |
|---|---|---|---|---|
| Perlin (P0) | 448×448 | **14.4** | **1.3** | yes |
| Our hybrid (P6) | 1024×1024 | **586** | **52.7** | yes |
| AnoStyler (P8) | 256×256 | **14,982** | **1,348** | yes |

Category time = time/image × 90 (the representative category size). Our hybrid additionally
pays a one-time ~0.4 s/category bank extraction (negligible, amortized over the 90 images).

Our method and Perlin are copy-paste/blend methods (sub-second); AnoStyler is a per-image
optimization (style transfer) and is **~3 orders of magnitude slower** than Perlin and
~25× slower than our method. Note our method runs at a higher working resolution
(1024 px) than Perlin (448 px), so its 586 ms is not directly comparable to Perlin's 14.4 ms
on equal work — see the per-method breakdown.

---

## Why the three methods use different resolutions

The native resolution is *not* an arbitrary choice I made — it is fixed by each method's
own design, and it is not equalized here (equalizing would change each method's cost):

1. **Perlin → 448×448.** Perlin is a *live* generator embedded in the anomalib harness. It
   runs on-the-fly during `engine.fit`, operating on whatever the datamodule hands it — and
   the datamodule resizes MVTec AD 2 to the harness eval resolution
   (`RESOLUTIONS["mvtec2"] = (448, 448)`). Perlin never sees any other size; it inherits 448
   from the pipeline.
2. **AnoStyler → 256×256.** Its style-transfer U-Net runs for 75 epochs × 64 crops per image,
   so cost scales with resolution; 256×256 is its native/optimized output (the save path
   hardcodes `F.interpolate(..., size=(256, 256))`). The harness resizes to 448 afterwards.
3. **Our hybrid → 1024×1024.** `flash-part2.ipynb` hardcodes `WORK_SIZE = 1024`, with this
   comment: *"1024 as in realbank_transfer: at 768 a rice defect replays at ~22px and the
   seam is below the resolution at which it can be judged."* Fine defects (rice, scratches)
   must replay large enough to remain visible, so synthesis is done at 1024 and then resized
   to 448 by the harness.

**Implication for the timing:** our method does ~5× the pixels per image of Perlin
(1024² vs 448²) and yet is only ~40× slower (586 ms vs 14 ms); normalized to equal pixel
count it would be substantially closer to Perlin. AnoStyler, by contrast, does the *fewest*
pixels (256²) but is dominated by its optimization loop, not resolution.

---

## Perlin (P0) — `src/anomalib/data/utils/generators/perlin.py`

**Measured:** 30 images at 448×448 → **14.4 ms/img** (std 3.4 ms, range 8.4–25.5 ms). At
90 images/category this is **1.3 s** per category.

**What is included** (one `PerlinAnomalyGenerator.forward(img)` call, the harness's exact
P0 config: `anomaly_source_path=./datasets/dtd`, `probability=1.0`, `blend_factor=(0.01,0.2)`):
1. Perlin-noise anomaly mask generation (GPU).
2. Random source-patch selection from the DTD texture bank + random affine (rotation) +
   random color augmentation (`MultiRandomChoice` of 10 torchvision ops).
3. Patch blend into the host via the mask (alpha with the drawn blend factor).

**Not included:** model training, dataset I/O, threshold fitting, metric evaluation; the
DTD texture bank is loaded once at generator construction.

---

## AnoStyler (P8) — `gsoc_workspace/AnoStyler/generate_mvtec2_anostyler.py`

**Measured:** 10 images solo at 256×256 → **14,982 ms/img** (std 501 ms, range
14,237–15,818 ms). The full 2115-image run logged a mean of ~20 s/img when two processes
shared a GPU (throughput ~2×, per-image latency ~1.3× higher); the solo number is the fair
per-image cost. At 90 images/category this is **1,348 s (~22.5 min)** per category.

**What is included** (one generated image + its mask):
1. **One-time model load** (amortized over the whole run, ~1 min total, *not* counted in
   the 15 s): SAM ViT-B (foreground), VGG19, CLIP ViT-B/32, and the tiny StyleNet U-Net.
2. **Mask generation** (~1–2 s): SAM background-inversion foreground mask ∩ procedurally
   generated meta-shape mask (`msp_line`/`msp_dot`/`msp_freeform`).
3. **Style transfer** (~13 s, dominant): a fresh U-Net (`ngf=16`) optimized for `max_step=75`
   epochs × `num_crops=64` random 128×128 crops under VGG perceptual (content) + CLIP
   directional losses (`lambda_tv/pdir/gdir/c/mclip`), blending the stylized patch into the
   host only inside the mask.

**Not included:** harness training/eval; the one-time model download/load.

---

## Our hybrid (P6) — `gsoc_workspace/flash-part2.ipynb`

**Measured:** 30 hosts at 1024×1024 → **586 ms/img** (std 62 ms, range 429–756 ms), plus a
**one-time bank extraction of ~125 ms/donor** (3 donors per category ≈ 0.4 s/category). At
90 images/category this is **52.7 s** per category (+ the ~0.4 s bank extraction).

### Per-image stage breakdown (1024×1024, poisson arm)

| stage | time | what it does |
|---|---|---|
| `host_and_region` | ~154 ms | load host, resize to 1024, OBS object-foreground (`_fg_robust`) |
| `generate_mrsp` | ~12 ms | 6-octave FFT spectral-pyramid noise (`generate_mrsp`) |
| `mrsp_mask_for` | ~37 ms | noise → coverage threshold → largest-blob mask |
| `place_entry` | ~5 ms | resize/rotate/stamp the bank patch |
| `harmonise` | ~139 ms | Lab colour-harmonisation of patch substrate → host |
| `composite` (poisson) | ~212 ms | collar + `cv2.seamlessClone` (NORMAL_CLONE) blend |

(Stages overlap/cache slightly, so the sum ≈ the measured 586 ms total.)

### What is included vs. not

- **Included in the 586 ms/img:** everything *per synthetic image* — host load, foreground
  extraction, mask generation, patch placement, harmonisation, and the poisson composite
  (image + mask are both written).
- **Included in the one-time bank cost (~125 ms/donor):** `regions_of` (connected-component
  defect-mask recovery) + `extract_entry` (substrate-retained square crop + soft alpha +
  contrast/coverage QC). Done once per donor, amortized over the whole corpus.
- **NOT included (offline, manual):** generating the 3–4 *donor* defect images per category
  with Gemini (or another external generator). This is the "Additional" time, added by hand
  below.

### Additional (offline) time

The donor defect-mask is **not generated in this notebook** — it is a "diffmask" computed
*before* the notebook runs (anomaly image minus its normal counterpart, thresholded), read
back from `MASK_DIR` as `<cat>/<did>/mask.png`. The notebook only consumes it ("This
notebook assumes diffmask output; register first if that changed"), so the mask-generation
step is external and its cost is reported separately here.

| step | time | notes |
|---|---|---|
| Gemini (or external) generation of 3–4 donor defect images per category | **~10–30 s/donor (estimated)** → ~1–2 min/category (3–4 donors) | offline API call, not measured here (no key). Instrumented script: `gsoc_workspace/time_gemini_donor.py`; estimate from Gemini Flash image (~3–15 s) to OpenAI gpt-image-1 (~10–60 s) tiers |
| diff-mask recovery from those donors (|anomaly−normal| → Otsu → morphology → connected components) | **~163 ms/donor → ~0.49 s/category** (3 donors) | measured on 20 MVTec AD 2 pairs; NOT in the notebook — external pre-registration |

---

## Methodology / notes

- **Measurement:** Perlin and our method via a standalone script reusing the exact function
  bodies extracted from `flash-part2.ipynb` (`ast`-extracted, verified importable); AnoStyler
  via its own batch generator on fresh (non-cached) images.
- **Averaging:** Perlin 30 images; our method 30 hosts (single seed, `poisson` arm); AnoStyler
  10 images solo (plus the full 2115-image log distribution). Sample counts are more than
  the requested 10 to tighten the estimate; timings are stable (std ≤ ~10% of the mean).
- **Resolution is not equalized** — it is part of each method's identity: Perlin generates at
  the harness eval resolution (448 px), AnoStyler at its native 256 px, and our method at
  its 1024 px working resolution (all are resized to 448 px by the harness before training).
- **Hardware:** single RTX 3090; AnoStyler uses fp16, our method uses CPU + torch FFT (GPU
  available for `generate_mrsp`).
