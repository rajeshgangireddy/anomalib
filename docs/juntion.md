# Junction 2026: Intel Physical AI Challenge with Physical AI Studio

**Challenge:** _Stop Simulating, Start Moving: See It. Detect It. Act on It._

Build a closed-loop robot system on an SO-101 arm:

1. Look at tabletop objects, such as Duplo bricks.
2. Detect the defective ones, such as a brick marked with a black X, dents, cracks, or color and texture anomalies.
3. Decide what to do with each object.
4. Sort it physically into a "good" or "defect" bowl.

Everything runs on an Intel Core Ultra AI PC.

```text
PERCEIVE ──► DETECT ──► REASON ──► EXECUTE ──► OPTIMIZE
cameras      Anomalib    VLA + task  SO-101      OpenVINO on
                         prompt      arm         CPU / iGPU / NPU
```

The exact objects, defects, and sorting rules are announced at kickoff. This guide works for any rule set.

## Contents

- [Junction 2026: Intel Physical AI Challenge with Physical AI Studio](#junction-2026-intel-physical-ai-challenge-with-physical-ai-studio)
  - [Contents](#contents)
  - [How the pieces fit together](#how-the-pieces-fit-together)
  - [Step 1: Check the station](#step-1-check-the-station)
  - [Step 2: Set up Modal for GPU training](#step-2-set-up-modal-for-gpu-training)
    - [Managing your $100 credit](#managing-your-100-credit)
  - [Step 3: Configure Studio](#step-3-configure-studio)
  - [Step 4: Train the anomaly detector (Anomalib)](#step-4-train-the-anomaly-detector-anomalib)
    - [4.1 Install](#41-install)
    - [4.2 Collect images](#42-collect-images)
    - [4.3 Choose a model](#43-choose-a-model)
    - [4.4 Train, test, and export](#44-train-test-and-export)
    - [4.5 Run inference with OpenVINO](#45-run-inference-with-openvino)
    - [4.6 Anomalib best practices](#46-anomalib-best-practices)
  - [Step 5: Record demonstrations](#step-5-record-demonstrations)
  - [Step 6: Train the VLA policy](#step-6-train-the-vla-policy)
  - [Step 7: Run the policy on the arm](#step-7-run-the-policy-on-the-arm)
  - [Step 8: Close the loop](#step-8-close-the-loop)
  - [Step 9: Optimize on Intel hardware](#step-9-optimize-on-intel-hardware)
  - [Judging rubric](#judging-rubric)

---

## How the pieces fit together

Physical AI Studio handles **data collection → policy training → export → running the policy on the robot**. It does not do anomaly detection, so you train an **Anomalib** model separately and connect the two through the policy's **task prompt**:

1. Anomalib classifies the inspected object as `good` or `defective`.
2. Your code maps that result to a fixed language task:
   - `good` → `Pick up the brick and place it in the blue bowl`
   - `defective` → `Pick up the brick and place it in the red bowl`
3. A language-conditioned VLA policy trained in Studio (SmolVLA or Pi0.5) carries out that task on the SO-101.

Because you record demonstrations for each task string, a single policy learns both sorting actions.

---

## Step 1: Check the station

Do this in the first 30–60 minutes.

- [ ] **Software works:** Physical AI Studio opens at http://localhost:3000.
  - **Recommended: run it natively.** Start the backend and the UI in two terminals:

    ```bash
    # Terminal 1: backend (http://localhost:7860)
    cd application/backend
    cp .env.example .env
    uv run --extra xpu physicalai-studio serve
    ```

    ```bash
    # Terminal 2: UI (http://localhost:3000)
    cd application/ui
    npm install
    npm run start
    ```

  - **Alternative: Docker.** From `application/docker/`, run `./setup-devices.sh --xpu`, then `docker compose up -d`.

- [ ] **Arms are connected:** the SO-101 leader and follower are powered and connected through the USB hub. Route cables away from the arm's path.
- [ ] **Each camera has a role:**
  - **Overhead** (preferably RealSense) for the scene overview.
  - **Inspection** camera aimed at the spot where objects are checked for defects. Only Anomalib uses this camera, so it is not added to Studio.
  - **Wrist** camera on the SO-101.
- [ ] **Scene is set up:** lighting is even and the bowls are within reach. **Fix the camera mounts firmly**, because a camera that moves no longer matches your training data.
- [ ] **Hugging Face token:** a read-only token is set in **Settings → General → Hugging Face**. SmolVLA and Pi0.5 download weights from the Hub.

## Step 2: Set up Modal for GPU training

Training a VLA needs an NVIDIA GPU. Studio trains on **Modal**, and **each participant gets $100 of Modal training credit**. Set this up early, because the first image build takes several minutes.

1. **Create an account.** Each participant creates an account at [modal.com](https://modal.com) and redeems their $100 credit as the organizers instruct.
2. **Give the Studio backend your Modal credentials.** The backend process makes the Modal calls, so it needs the token. Logging in through your browser is not enough.
   - **Native backend:** run this as the same OS user that runs the backend:

     ```bash
     uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup
     ```

   - **Docker:** create a token in the Modal dashboard. Pass `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to the backend container in a `docker-compose.override.yml` or an `env_file`, because the default Compose files do not forward them. Never commit the token.

3. **Create the training target.** Open **Settings → Training Targets → Modal Provider**, enter a name such as `modal-<yourname>`, and click **Deploy and connect**. Studio creates a Modal app and storage Volume and builds the trainer image. Wait until the target's health status shows it is ready.

### Managing your $100 credit

- **Choose the GPU for each job** in **Train model → NVIDIA GPU**: **A10G** (default and cheapest), **A100 80 GB**, or **H100**. Check [modal.com/pricing](https://modal.com/pricing) to see how many GPU hours $100 buys.
- **Test on A10G.** Use A100 or H100 only for final, longer SmolVLA or Pi0.5 runs, after a short run shows your data and settings work.
- **Running jobs keep billing**, even if you close Studio or lose the network. Cancel jobs you don't need, then check the Modal dashboard to confirm the container stopped.
- **Plan runs to finish in time.** Each job gets one GPU, runs for at most 24 hours, and cannot resume from a checkpoint.
- **Clean up afterwards.** Volume storage is billed separately. After the event, delete the target in Studio and check the Modal dashboard for any leftover apps or Volumes.

## Step 3: Configure Studio

1. **Projects:** create a project, for example `junction-sorter`.
2. **Robots:** add the **SO101 follower** and **SO101 leader**. Calibrate with the setup wizard, or choose _Skip verification_ if the board is already calibrated. Then move the arm and check that the 3D view matches.
3. **Cameras:** add the overhead and wrist cameras, and check each preview. Leave out the inspection camera: the policy doesn't need it, and Anomalib opens it directly, which it can't do while Studio holds the device. Use **640×480 at 30 FPS**. Models resize images internally, so a higher resolution only uses more disk space.
4. **Environments:** create one environment with the follower, the leader, and the overhead and wrist cameras. Open it and confirm everything streams.

> **Freeze camera positions after this step.** If a camera moves later, you have to re-record your data.

## Step 4: Train the anomaly detector (Anomalib)

This step runs outside Studio, so one teammate can work on it while another does Step 5.

**References:** [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib) · [documentation](https://anomalib.readthedocs.io/en/latest/) · [example notebooks](https://github.com/open-edge-platform/anomalib/tree/main/examples/notebooks). Start with the custom `Folder` dataset and OpenVINO inference notebooks.

### 4.1 Install

Install Anomalib in its own virtual environment so its PyTorch version doesn't clash with Studio's. The Anomalib README recommends `uv` with an explicit hardware extra:

```bash
uv venv anomalib-env && source anomalib-env/bin/activate
uv pip install "anomalib[openvino,xpu]"   # train on the Intel iGPU; use [openvino,cpu] for CPU only
```

> **Version:** the latest release is 2.7.0, which requires PyTorch 2.10 or later. Use the version on the event image, and keep the whole team on it.

### 4.2 Collect images

Capture frames from the **inspection camera** with OpenCV (for example `cv2.VideoCapture(<index>)`; `lerobot-find-cameras` lists the indices). Arrange them in a `Folder` layout:

```text
datasets/bricks/
├── good/     # ~100–300 images of acceptable objects (used for training)
└── defect/   # a few dozen defective examples (used only for validation and the threshold)
```

- **Training needs only good images.** The model learns what "good" looks like and flags anything different. You don't need defect labels, and it can also flag defect types first revealed at kickoff.
- **Cover all normal variation.** Capture good objects in every pose, color, and position you expect at runtime. Anything not represented in `good/` gets flagged as a defect, which causes false rejects.
- **Keep the view consistent.** Use the event lighting and a fixed camera mount, and **crop to a fixed region of interest** where the object is inspected. Anomaly models work much better on a centered, consistent view than on a cluttered scene.

### 4.3 Choose a model

| Model           | Description                                                                 | When to use it                                     |
| --------------- | --------------------------------------------------------------------------- | -------------------------------------------------- |
| **PaDiM**       | Simplest; trains in about 1 minute                                          | Quick first baseline                               |
| **PatchCore**   | Very good results; trains in 3–5 minutes                                    | **Recommended default**                            |
| **Dinomaly**    | One of the state-of-the-art models, but needs a lot of training             | When you have time and GPU budget to push accuracy |
| **EfficientAd** | Not state of the art, but fast at inference (training batch size must be 1) | When latency matters most                          |
| **FoundAD**     | New model, reportedly better for few-shot                                   | When you have only a few good images               |

Start with PaDiM or PatchCore. If time allows, compare models with `anomalib benchmark` and report the trade-off between accuracy and latency.

### 4.4 Train, test, and export

```python
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

datamodule = Folder(
    name="bricks",
    root="./datasets/bricks",
    normal_dir="good",
    abnormal_dir="defect",
)
model = Patchcore()
engine = Engine(max_epochs=1)  # PatchCore needs one pass to build its memory bank

engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)  # image AUROC / F1 on held-out data
engine.export(model=model, export_type=ExportType.OPENVINO)
```

You can do the same from the CLI. To make runs repeatable, put these options in a YAML config and use `--config`:

```bash
anomalib train --model Patchcore --data anomalib.data.Folder \
  --data.name bricks --data.root ./datasets/bricks \
  --data.normal_dir good --data.abnormal_dir defect
```

> **Note: train anomaly models on the laptop's Intel GPU (XPU).** Anomaly models are small enough to train locally, so they don't need Modal or your GPU credit. Only the VLA policy needs a cloud GPU.
>
> - **CLI:** add `--trainer.accelerator xpu --trainer.strategy xpu_single`.
> - **Python:**
>
>   ```python
>   from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
>
>   engine = Engine(accelerator=XPUAccelerator(), strategy=SingleXPUStrategy())
>   ```
>
> Anomalib supports only a single XPU. Install it with the `xpu` extra (Step 4.1).

### 4.5 Run inference with OpenVINO

```python
from anomalib.deploy import OpenVINOInferencer

inferencer = OpenVINOInferencer(
    path="results/Patchcore/bricks/latest/weights/openvino/model.xml",
    device="NPU",  # or "GPU" / "CPU"; compare all three in Step 9
)
pred = inferencer.predict(image=frame)  # pred.pred_score, pred.pred_label, pred.anomaly_map
```

- **Check the threshold.** Anomalib sets it during training and stores it in the exported model. Check it on held-out good and defect images, and adjust it if there are too many false rejects. Record precision and recall for the judges.
- **Wrap it in a function** `frame -> ("good" | "defective", score)` to use in Step 8.
- **Show the `anomaly_map` heat map** in your demo. It shows judges _why_ an object was rejected.

### 4.6 Anomalib best practices

- Use a dedicated virtual environment with `uv`, and install the hardware extra you need.
- Keep training in a **YAML config or script** in your repository so it is reproducible.
- Turn on a **logger** (TensorBoard, MLflow, or W&B) to track runs and metrics.
- Use `Engine.predict()` or `OpenVINOInferencer` for inference.
- Turn off saved visualizations in the live loop (`--model.visualizer false`) to reduce latency.

## Step 5: Record demonstrations

1. Go to **Datasets → New Dataset**. Select your environment and give the dataset a clear name, such as `sort-v1`.
2. Choose **one fixed task string per outcome**, for example:
   - `Pick up the brick and place it in the blue bowl` (good)
   - `Pick up the brick and place it in the red bowl` (defective)

   At inference time you must use exactly the same text.

3. Record in a loop:
   1. Reset the scene.
   2. Set the task.
   3. Click **Start episode** (or press →).
   4. Demonstrate the task with the leader arm.
   5. Click **Accept** (→) to keep the episode or **Discard** (←) to drop it.
4. Record **at least 50 episodes per task**, with roughly equal numbers for each task. Vary object position, rotation, and clutter, because judges score robustness to shifted and cluttered scenes.
5. **Review** every episode: replay it, check the joint graphs, and delete hesitant or failed demonstrations.
6. **Export** the dataset in LeRobot v3 format as a backup and for your submission.

**Recording tips:** move smoothly at a steady pace, start and end in the same home pose, and keep the leader arm away from its joint limits.

## Step 6: Train the VLA policy

1. Go to **Models → Train model** and pick your dataset.
2. Choose a policy:
   - **SmolVLA** (recommended): takes language input and fine-tunes quickly.
   - **Pi0.5**: stronger, but slower to train and to run.
   - **ACT**: has no language input. You would need one ACT model per bowl and code that switches between them.
3. Under **Run on**, select your Modal target and choose an **NVIDIA GPU**: A10G for test runs, A100 or H100 only for the final run.
4. Watch the loss curve and logs on the Models page. **Interrupt** stops training, saves a checkpoint, and still exports the model.
5. When training finishes, Studio automatically exports the model to **PyTorch, OpenVINO, ONNX, and ExecuTorch**.

## Step 7: Run the policy on the arm

1. On the trained model, click **Run model**. Choose the **OpenVINO** backend and the **GPU** (iGPU) or **CPU** device.
2. Studio loads the environment used to record the dataset. Choose a task string and press **Play**.
3. Test each task manually: place a known good brick, select the "good" task, and check that it goes to the right bowl. Then do the same with a defective brick.
4. If a case fails, record more episodes of it (different positions, more clutter), retrain, and test again. Expect two or three rounds.

## Step 8: Close the loop

In Studio's Run screen you choose the task yourself. To have Anomalib choose it automatically:

1. Download the exported **OpenVINO** model from Studio.
2. Write a small control script with [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai) (`InferenceModel(...)`) that repeats this cycle for each object:
   1. Capture an inspection frame and run **Anomalib** to get `good` or `defective`.
   2. Map the result to the matching **task string**.
   3. Run the **VLA policy** with that task, using the camera and joint observations, until the pick-and-place finishes.
   4. Return the arm to its home pose.
3. Add safety checks: joint limits, a timeout for each cycle, and a manual stop.

**Fallback if time runs short:** run Anomalib live, show its result on screen, and pick the matching task in Studio's Run screen yourself. The pipeline is still complete; only the hand-off between detection and action is manual.

## Step 9: Optimize on Intel hardware

Intel tech integration is worth 25 points and system efficiency another 20. Back every claim with measurements.

- **Spread the work across devices:** for example, Anomalib on the **NPU**, the VLA policy on the **iGPU**, and the control loop on the **CPU**. The two models can then run in parallel.
- **Compare backends and devices:** PyTorch vs OpenVINO, and CPU vs iGPU vs NPU. Record latency per inference step and per full sorting cycle.
- **Try lower precision** (FP16, or INT8 with NNCF) wherever accuracy holds up.
- **Measure power and temperature** during a run, for example with `intel_gpu_top`, `powertop`, and NPU utilization. Say clearly which metrics you could not measure.

## Judging rubric

| Criterion (points)              | Where you earn it                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Task Success Rate (25)          | Steps 5, 7, 8: varied demonstrations, fixing failures, working closed loop                                                            |
| Intel Tech Integration (25)     | Steps 4, 8, 9: OpenVINO exports, NPU/iGPU offload, Studio workflow                                                                    |
| Dataset & Pipeline Quality (20) | Steps 4, 5, 6: clean teleop data, reproducible training                                                                               |
| System Efficiency (20)          | Step 9: measured latency, power, and thermals on Core Ultra                                                                           |
| Creativity (10)                 | Your own ideas beyond the baseline, such as an original sorting rule, a recovery behavior, or a clear live visualization of decisions |
| **Total (100)**                 |                                                                                                                                       |
