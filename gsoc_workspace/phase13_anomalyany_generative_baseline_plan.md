# Phase 13 — AnomalyAny (CVPR 2025) Generative Baseline

**Status:** generation running on local 2 GPUs at 448 px. ~115 s/image is intrinsic to
the method (60 actual denoising steps × a 10-iteration attend-and-excite/CLIP gradient
loop; see §4b) — not a CPU/device issue: the SD pipeline is already fp16 on the 3090,
and bf16/torch.compile don't apply (Ampere has no bf16 tensor-core win; the repo pins
torch 1.13.1, which predates `torch.compile`). **One seed first** (705 images, ~11 h
ETA) to get a result quickly; seeds 1/2 deferred. Harness + phase spec implemented and
verified.

Goal: add a **third synthetic-anomaly arm** to our 3-arm comparison (Perlin P0 / our
hybrid P6 / **generative**) using [AnomalyAny](https://github.com/EPFL-IMOS/AnomalyAny)
(Sun, Cao, Dong, Fink — CVPR 2025, "Unseen Visual Anomaly Generation"). This gives us a
strong, genuinely-*generative* diffusion baseline against which to position our own
copy-paste/blend method, and a third data point for the WACV-2027 paper.

---

## 1. What AnomalyAny does (mechanism)

Repo cloned to `gsoc_workspace/AnomalyAny/` (git-ignored). Core files:

- `clip_pipeline_attend_and_excite.py` — `RelationalAttendAndExcitePipeline`, a custom
  subclass of `diffusers.StableDiffusionPipeline` (SD v1.4 / v1.5 / v2.1).
- `run.py` — entry point (`@pyrallis.wrap`), `run_on_prompt_and_masked_image`.
- `config.py` — `RunConfig` dataclass of all hyperparameters.
- `clip_anomaly_generation_mvtec.ipynb` — the MVTec-specific notebook (what we adapt).
- `utils/fg_extraction.py` — coarse foreground mask (global threshold 127).

Generation is **text + normal-image-conditioned img2img diffusion** with two added
mechanisms:

1. **Test-time normal-sample conditioning** (`init_image_guidance_scale`, γ):
   the normal image is VAE-encoded, noised, and denoising starts from
   `t_start = steps − int(steps·γ)`. γ controls edit strength (notebook: **0.3**). This
   is the "condition on a single normal sample at test time" of the paper.
2. **Attention-guided anomaly optimization** (Attend-and-Excite): for the first
   `max_iter_to_alter` steps, the cross-attention of the *anomaly token(s)*
   (`indices_to_alter`) is boosted so SD is forced to render the defect concept
   (scale_factor=50, thresholds={0:.05,10:.5,20:.8}).
3. **Prompt-guided anomaly refinement** (CLIP loss): `normal_prompt` vs
   `detailed_prompt` (anomaly description) steer the edit toward the defect
   (relational CLIP in the pipeline; plus optional explicit `clip_loss` in `run.py`).
4. **Optional mask inpainting** (`mask_image`): if given, the anomaly is localized to
   that region (masked-noise blending). The MVTec notebook passes a **foreground mask**
   produced by `fg_extraction` (global threshold 127), so the defect is confined to the
   object's foreground.

Inputs per image: `prompt` (anomaly text), `normal_prompt`, `detailed_prompt`,
`token_indices` (anomaly keyword), `init_image` (one normal image), optional
`mask_image`, and a torch seed. Output: one 512×512 RGB anomaly image (+ optional
latents / cross-attention maps).

Notebook defaults (MVTec): `runwayml/stable-diffusion-v1-5`,
`NUM_DIFFUSION_STEPS=200`, `GUIDANCE_SCALE=12.5`, `init_image_guidance_scale=0.3`,
`scale_factor=50`, `max_iter_to_alter=25`, `attention_res=16`.

## 2. How it differs from our method (why it's a good baseline)

| axis | our hybrid (P6) | AnomalyAny |
|---|---|---|
| mechanism | copy-paste real donor defect patches (semantic bank) + poisson/alpha blend | diffusion (SD 1.5) edit of a normal image, text/attention/CLIP-guided |
| realism of defect | real (donated) texture | synthesized but photorealistic, "unseen" |
| needs donor bank | yes (3 pairs/category) | no — text prompt only |
| needs a mask *input* | no (paste site is the GT) | optional foreground mask (localization) |
| produces GT defect mask | yes (exact paste region) | **no** — only attention maps / coarse FG mask |
| compute | near-instant per image | ~tens of seconds / image (200-step diffusion) |

Key point: AnomalyAny is a **true generative** method that creates the defect *in the
image*, not by transplanting a real defect. That is the natural foil for our method in
the paper ("transplantation vs generation for label-free threshold calibration").

## 3. Environment (separate — NOT our current uv env)

AnomalyAny pins **Python 3.10 + torch 1.13.1 (CUDA 11.6) + diffusers 0.21.1 +
transformers 4.29.2 + open-clip-torch 2.10.1 + clip 1.0 + pyrallis**. This is
incompatible with our uv env (Python 3.13, modern torch/diffusers), so we create a
**dedicated uv venv** (no conda/mamba on this machine).

Plan: `uv venv` pinned to **Python 3.10**, install only the generation-critical deps
(pinned to the repo's versions):

```
torch==1.13.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
diffusers==0.21.1
transformers==4.29.2
accelerate==0.33.0
open-clip-torch==2.10.1
clip==1.0
pyrallis==0.3.1
numpy pillow opencv-python scikit-image pandas matplotlib
```

Skipped (metric/data-download only, not needed to *generate*): `mmcv`, `opendatalab`,
`openxlab`, `datasets`, `wilds`, `ogb`, `segment-anything`, `visdom`, jupyter stack.

Risks / mitigations:
- **torch 1.13.1 + cu116** on driver 580: driver is backward-compatible with the
  bundled CUDA 11.6 runtime → expected to work. Fallback if the old wheel misbehaves:
  try `torch==2.0.1` (still compatible with diffusers 0.21.1 / transformers 4.29.2).
- **SD weights** (~4 GB, `runwayml/stable-diffusion-v1-5`) + CLIP/open-clip weights
  download from HuggingFace on first run — must pre-download before bulk generation
  (same lesson as the phase-10 remote download outage).

## 4. Generation scope (mirror SynthetciGenMVAD2 layout)

Reuse the exact on-disk contract of our hybrid dataset so the harness can consume it
with minimal new code:

```
datasets/SynthetciGenMVAD2/MVTec_AD_2_anomalyany_{gen_seed}/{category}/test_public/
    bad/{idx}_{category}-*.png                 # generated anomaly image (no masks, §5)
```

- **Categories**: the same 8 MVTec AD 2 categories.
- **Counts = real `test_public/bad` count per category** (verified on disk):
  can 90, fabric 90, fruit_jelly 60, rice 90, sheet_metal 90, vial 105, wallplugs 90,
  walnuts 90 → **705 / seed**.
- **Seeds**: 3 supported (matching our `MVTec_AD_2_hybrid_{0,1,2}` and the harness's
  seed-remap `(seed−1) % 3`), but **only `gen_seed=0` is generated first** (~11 h) to
  get a result quickly — see §4b. Seeds 1/2 are deferred until seed 0 is reviewed.
- **Guidance image**: one distinct normal `train/` image per generated anomaly (1:1
  host mapping) → maximal diversity and faithful to "condition on a single normal
  sample". Train counts (137–432/category) comfortably cover 60–105 anomalies.

## 4b. Why ~115 s/image, and the 1-seed-first decision

Measured ~115 s/image on the 3090, and it is **not** a device/CPU problem: the SD
pipeline is already fp16 (`torch_dtype=float16`; log confirms `unet dtype=float16`),
and bf16/torch.compile don't apply (Ampere has no bf16 tensor-core benefit; the repo
pins torch 1.13.1, which predates `torch.compile`).

The cost is intrinsic to the AnomalyAny mechanism: `--steps 200` is misleading because
`init_image_guidance_scale=0.3` sets `t_start = 200 − int(200·0.3)`, so only **60
actual denoising steps** run. Each of those runs a **10-iteration gradient loop**
(`for q in range(10)`) doing attend-and-excite (UNet forward + backward) plus, for late
steps, a CLIP global loss (VAE decode + CLIP forward + backward) and gradient updates
on both latents and prompt embeddings. That is ~10× the compute of plain SD img2img;
GPU util sits at 40–60% (latency-bound on many small ops). xformers isn't installed and
would only shave ~1.3× (the bottleneck is the 10× loop, not attention throughput).

**Decision: 1 seed first** (seed 0, 705 images ≈ 11 h wall-clock) keeping 200 steps for
faithfulness to the paper. Add seeds 1/2 later if seed 0 looks good.

## 4a. Generation recipe (prompts + settings)

**Seed handling — yes, multiple seeds are trivially supported.** AnomalyAny is plain
SD img2img: the only stochasticity is the diffusion noise, controlled by a
`torch.Generator(device).manual_seed(s)` passed to the pipeline. Varying `s` gives a
different anomaly image even for the *same* conditioning normal image. So our 3-seed
convention maps directly onto `gen_seed ∈ {0,1,2}` → `manual_seed(gen_seed)` (plus a
per-image host index offset so all 705 within a seed are also distinct). No repo change
needed — `run.py` already accepts a `seed` field in `RunConfig`.

**Prompts** (per category; the `detailed_prompt`/`prompt` used as the anomaly target,
`normal_prompt` as the "before" baseline for the relational CLIP loss):

| category | `normal_prompt` | `detailed_prompt` (anomaly) |
|---|---|---|
| can | "a photo of a can" | "a photo of a can with a crack on it" |
| fabric | "a photo of a fabric" | "a photo of a fabric with a hole in it" |
| fruit_jelly | "a photo of a fruit jelly" | "a photo of a fruit jelly with a crack on it" |
| rice | "a photo of rice" | "a photo of rice with a broken grain" |
| sheet_metal | "a photo of a sheet metal" | "a photo of a sheet metal with a scratch on it" |
| vial | "a photo of a vial" | "a photo of a vial with a crack on it" |
| wallplugs | "a photo of a wall plug" | "a photo of a wall plug with a crack on it" |
| walnuts | "a photo of a walnut" | "a photo of a walnut with a crack on it" |

The `indices_to_alter` token(s) are the last word(s) of `detailed_prompt`
(crack / hole / broken grain / scratch).

**Settings** (notebook defaults kept):
`runwayml/stable-diffusion-v1-5`, `NUM_DIFFUSION_STEPS=200`, `GUIDANCE_SCALE=12.5`,
`init_image_guidance_scale=0.3`, `scale_factor=50`, `max_iter_to_alter=25`.
**Resolution 448 px** (not the notebook's 512 px): the default 512 px OOMs the 24 GiB
3090 (21.7 GiB), and 448 px matches the harness's MVTec AD 2 eval resolution exactly
(`RESOLUTIONS["mvtec2"] = (448, 448)`), avoiding an extra resize. `attention_res`
scales with resolution (`resolution // 32` → 14 at 448 px), which required a small fix
in `generate_mvtec2.py` (the notebook's hardcoded `attention_res=16` only matches
512-px latents and crashes aggregation at any other size). **Mask inpainting disabled**
(no `mask_image`) since we generate image-only (§5) and want the full-image edit, not
FG-confined inpainting.

## 5. The mask question — resolved: **no masks needed** for the primary flow

The user's clarification reframes this correctly, and the harness confirms it:

- **Primary evaluation = real anomalous test set of MVTec AD 2, calibrated on the
  synthetic images.** The synthetic images feed only the **arm B** threshold
  calibration; the final headline scores come from arm B (synthetic threshold
  transferred to the *real* test set), which ships its own `ground_truth/` masks. So
  the real set supplies the pixel-level GT, and the synthetic images do **not** need
  masks at all.
- **Synthetic-on-synthetic (arm C) is a secondary, optional diagnostic** (evaluate the
  detectors on the generated anomalies themselves as a test set). *Only* that case
  needs masks on the generated images, and only to report pixel-F1 there. It is not
  required for the headline claim.

Harness support needed **two small fixes** (now implemented and verified) beyond the
one-line registry entry:

- `_pregenerated_eval_set` previously hardcoded `attrs["task"] = "segmentation"`. That
  crashes for image-only sets: `AnomalibDataset.__getitem__` calls `read_mask(None)`
  for abnormal rows in segmentation mode. It now sets `"classification"` when no
  `*_mask.png` files exist, which is correct — the synthetic rows are never pixel-
  evaluated anyway (arm B transfers only the image threshold to the real test).
- Added `_reset_post_processor(model)` before each arm-B `engine.validate`. Without it,
  a maskless calibration set never updates the pixel-threshold metric, so the buffer
  silently retains the *oracle* threshold fitted during `engine.fit` on the real test
  — inflating arm-B pixel metrics (leakage). Resetting to NaN forces a clean refit or
  the documented midpoint-normalization fallback.

**Decision:** generate **image-only** (no masks). If we later want a synthetic-on-
synthetic pixel number, we revisit one of two options and cite it as an approximation:
(a) binarized anomaly-token cross-attention as a pseudo-mask, or (b) the `fg_extraction`
foreground mask. Either way we state clearly that AnomalyAny emits no defect-level GT
— **this is a stated limitation of the generative baseline in the paper.**

## 6. Harness integration (small, additive)

- Add `PREGENERATED_PIPELINES["P7"] = "anomalyany"` (+ `PREGENERATED_CATEGORIES` stays
  the same; `PREGENERATED_SEEDS` unchanged).
- `_pregenerated_eval_set` generalizes to any `MVTec_AD_2_{arm}_{gen_seed}` directory
  name and now tolerates missing masks (classification fallback, see §5) — implemented
  and verified.
- Run the new `phase13_anomalyany` spec (superadd huge_plus / padim / patchcore /
  anomaly_dino, `pipelines=["P0","P7"]`, 1 seed, heldout calibration, arm C on) to
  get P0-vs-P7 numbers directly comparable to phase11's P0-vs-P6.

## 7. Throughput & machine allocation

- ~115 s/image at 448 px on a 24 GiB 3090, fp16 + gradient checkpointing (see §4b for
  why it's slow — the 10× inner optimization loop, not the device). **1 seed = 705
  images ≈ 11 h wall-clock** split across 2 local GPUs. Memory is ~20 GiB/image (fits,
  but no room to raise resolution). xformers is not installed; it would only give
  ~1.3× (attention is not the bottleneck).
- **Machines**: as the user suggested, keep the 2 local 3090s for generation and leave
  phase-12 (backbone ablation) running on the remote 6 GPUs. If we need more parallel
  generation, we can also put the 3 smaller-backbone models on remote and free remote
  capacity later — phase 12 is already partitioned and healthy.

## 8. Decisions (finalized, generation running)

1. **Masks → none.** Generate image-only (§5). Calibration uses the synthetic images;
   evaluation uses the real test set (which has its own GT masks). "No defect-level GT
   on generated images" is a stated limitation of the generative baseline.
2. **Prompts & settings** → the §4a table (crack/hole/broken-grain/scratch per
   category) with 200 diffusion steps (notebook default), γ=0.3, guidance 12.5,
   resolution 448 px, mask inpainting off.
3. **Seeds → 1 first (gen_seed 0).** 3 seeds are supported (each `gen_seed ∈ {0,1,2}`
   maps to a distinct `torch.Generator.manual_seed`), but only seed 0 is generated
   initially to get a result in ~11 h instead of ~34 h; seeds 1/2 are added later if
   seed 0 looks good (per §4b).
4. **Scope** → generation running on 2 local GPUs; harness `P7` registry + maskless
   fallback + post-processor leak fix + `phase13_anomalyany` spec all implemented and
   verified. The P0-vs-P7 sweep launches once generation completes.
