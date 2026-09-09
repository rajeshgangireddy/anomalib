<!-- Supplementary material (Overleaf/LaTeX) -->
<!-- Tables S1–S2: per-category F1. Table S3: software/hardware environment. -->
<!-- Single-column (table) floats; copy each block into your supplementary.tex. -->
<!-- Requires: booktabs. Bold marks the best synthetic source in the Mean row only -->
<!-- (Real oracle excluded); per-category cells are not bolded. -->

## Table S1 — Image-level F1 (%)

\begin{table}[t]
\centering
\scriptsize
\begin{tabular}{lcccc}
\toprule
\textbf{Category} & \textbf{Real} & \textbf{Perlin} & \textbf{AnoStyler} & \textbf{FLASH (Ours)} \\
\midrule
\multicolumn{5}{l}{\textbf{PaDiM}} \\
\midrule
Can & 71.49 & 44.60 & 64.67 & 68.93 \\
Fabric & 73.89 & 73.17 & 58.37 & 73.17 \\
Fruit Jelly & 88.27 & 69.37 & 87.24 & 85.71 \\
Rice & 81.31 & 81.08 & 6.27 & 81.08 \\
Sheet Metal & 89.64 & 88.24 & 65.13 & 88.24 \\
Vial & 86.10 & 48.79 & 78.46 & 85.71 \\
Wallplugs & 75.24 & 75.00 & 63.80 & 75.00 \\
Walnuts & 77.06 & 75.00 & 75.00 & 75.00 \\
\midrule
\textbf{Mean} & 80.37 & 69.41 & 62.37 & \textbf{79.11} \\
\midrule
\multicolumn{5}{l}{\textbf{PatchCore}} \\
\midrule
Can & 70.92 & 37.86 & 42.56 & 45.33 \\
Fabric & 79.47 & 73.17 & 16.33 & 73.17 \\
Fruit Jelly & 92.16 & 46.64 & 54.16 & 73.56 \\
Rice & 80.73 & 81.08 & 11.42 & 81.08 \\
Sheet Metal & 89.00 & 88.24 & 26.75 & 88.24 \\
Vial & 91.02 & 64.28 & 84.67 & 85.71 \\
Wallplugs & 74.53 & 75.00 & 32.53 & 75.00 \\
Walnuts & 81.88 & 73.76 & 75.71 & 75.00 \\
\midrule
\textbf{Mean} & 82.46 & 67.50 & 43.02 & \textbf{74.64} \\
\midrule
\multicolumn{5}{l}{\textbf{AnomalyDINO}} \\
\midrule
Can & 71.34 & 46.75 & 44.28 & 52.45 \\
Fabric & 74.44 & 73.17 & 24.40 & 73.17 \\
Fruit Jelly & 86.83 & 24.76 & 48.14 & 85.71 \\
Rice & 84.26 & 81.08 & 17.98 & 81.08 \\
Sheet Metal & 88.69 & 88.24 & 58.08 & 88.24 \\
Vial & 91.61 & 54.81 & 74.65 & 85.71 \\
Wallplugs & 75.07 & 54.98 & 55.17 & 75.00 \\
Walnuts & 79.49 & 73.70 & 76.11 & 75.00 \\
\midrule
\textbf{Mean} & 81.47 & 62.19 & 49.85 & \textbf{77.05} \\
\midrule
\multicolumn{5}{l}{\textbf{Dinomaly}} \\
\midrule
Can & 71.43 & 43.21 & 51.48 & 50.81 \\
Fabric & 77.42 & 73.17 & 29.18 & 73.17 \\
Fruit Jelly & 87.03 & 6.32 & 45.02 & 85.28 \\
Rice & 81.16 & 81.08 & 40.71 & 81.08 \\
Sheet Metal & 88.66 & 88.24 & 50.27 & 88.24 \\
Vial & 89.05 & 56.19 & 87.57 & 85.71 \\
Wallplugs & 76.27 & 56.15 & 55.82 & 75.63 \\
Walnuts & 83.43 & 79.02 & 82.94 & 80.15 \\
\midrule
\textbf{Mean} & 81.81 & 60.42 & 55.37 & \textbf{77.51} \\
\midrule
\multicolumn{5}{l}{\textbf{SuperADD}} \\
\midrule
Can & 71.19 & 61.14 & 61.46 & 61.10 \\
Fabric & 75.60 & 73.17 & 41.03 & 73.17 \\
Fruit Jelly & 85.86 & 80.74 & 81.54 & 85.71 \\
Rice & 86.60 & 81.08 & 25.17 & 81.08 \\
Sheet Metal & 89.75 & 88.24 & 70.47 & 88.24 \\
Vial & 99.52 & 98.61 & 99.84 & 85.71 \\
Wallplugs & 76.47 & 75.00 & 52.87 & 75.00 \\
Walnuts & 84.09 & 75.00 & 82.51 & 75.00 \\
\midrule
\textbf{Mean} & 83.64 & \textbf{79.12} & 64.36 & 78.13 \\
\bottomrule
\end{tabular}
\caption{Per-category image-level F1 (\%) across all five detectors on MVTec AD 2. The Mean row averages over the eight categories; bold marks the best synthetic source per detector (the Real oracle is excluded).}
\label{tab:supp-image-f1}
\end{table}

## Table S2 — Pixel-level F1 (%)

\begin{table}[t]
\centering
\scriptsize
\begin{tabular}{lcccc}
\toprule
\textbf{Category} & \textbf{Real} & \textbf{Perlin} & \textbf{AnoStyler} & \textbf{FLASH (Ours)} \\
\midrule
\multicolumn{5}{l}{\textbf{PaDiM}} \\
\midrule
Can & 0.16 & 0.05 & 0.05 & 0.05 \\
Fabric & 3.36 & 1.44 & 0.93 & 1.01 \\
Fruit Jelly & 12.41 & 4.46 & 5.13 & 2.96 \\
Rice & 6.26 & 2.03 & 6.21 & 1.74 \\
Sheet Metal & 11.95 & 3.70 & 11.27 & 9.08 \\
Vial & 10.01 & 0.18 & 3.22 & 0.37 \\
Wallplugs & 1.09 & 0.88 & 0.94 & 0.72 \\
Walnuts & 15.83 & 11.63 & 15.23 & 15.21 \\
\midrule
\textbf{Mean} & 7.63 & 3.05 & \textbf{5.37} & 3.89 \\
\midrule
\multicolumn{5}{l}{\textbf{PatchCore}} \\
\midrule
Can & 0.06 & 0.04 & 0.00 & 0.00 \\
Fabric & 15.27 & 1.88 & 2.05 & 15.19 \\
Fruit Jelly & 40.02 & 38.92 & 36.06 & 14.28 \\
Rice & 22.81 & 2.64 & 0.00 & 3.26 \\
Sheet Metal & 30.69 & 5.12 & 0.01 & 26.32 \\
Vial & 32.23 & 17.66 & 27.54 & 28.14 \\
Wallplugs & 16.22 & 2.76 & 15.94 & 8.53 \\
Walnuts & 51.41 & 50.15 & 47.04 & 49.38 \\
\midrule
\textbf{Mean} & 26.09 & 14.90 & 16.08 & \textbf{18.14} \\
\midrule
\multicolumn{5}{l}{\textbf{AnomalyDINO}} \\
\midrule
Can & 0.06 & 0.04 & 0.00 & 0.00 \\
Fabric & 46.11 & 16.61 & 20.11 & 37.66 \\
Fruit Jelly & 40.22 & 15.92 & 13.42 & 21.26 \\
Rice & 58.47 & 31.29 & 1.84 & 56.34 \\
Sheet Metal & 32.09 & 9.37 & 3.12 & 8.16 \\
Vial & 32.45 & 9.00 & 25.64 & 27.30 \\
Wallplugs & 2.39 & 1.57 & 0.00 & 1.83 \\
Walnuts & 56.59 & 55.95 & 47.82 & 56.30 \\
\midrule
\textbf{Mean} & 33.55 & 17.47 & 13.99 & \textbf{26.11} \\
\midrule
\multicolumn{5}{l}{\textbf{Dinomaly}} \\
\midrule
Can & 0.03 & 0.03 & 0.00 & 0.00 \\
Fabric & 27.45 & 15.68 & 3.77 & 19.46 \\
Fruit Jelly & 52.89 & 28.04 & 9.76 & 37.18 \\
Rice & 46.46 & 22.35 & 0.00 & 21.21 \\
Sheet Metal & 44.24 & 15.00 & 0.00 & 30.90 \\
Vial & 35.46 & 0.58 & 9.36 & 23.97 \\
Wallplugs & 2.36 & 1.27 & 0.00 & 2.18 \\
Walnuts & 46.27 & 40.70 & 3.90 & 45.35 \\
\midrule
\textbf{Mean} & 31.90 & 15.45 & 3.35 & \textbf{22.53} \\
\midrule
\multicolumn{5}{l}{\textbf{SuperADD}} \\
\midrule
Can & 0.02 & 0.01 & 0.00 & 0.00 \\
Fabric & 78.36 & 18.57 & 44.30 & 69.06 \\
Fruit Jelly & 56.07 & 40.02 & 55.93 & 55.57 \\
Rice & 58.75 & 6.21 & 25.36 & 51.90 \\
Sheet Metal & 35.87 & 2.53 & 7.79 & 7.39 \\
Vial & 57.05 & 56.46 & 46.63 & 39.41 \\
Wallplugs & 54.30 & 1.98 & 50.79 & 17.41 \\
Walnuts & 71.86 & 29.78 & 65.12 & 66.28 \\
\midrule
\textbf{Mean} & 51.53 & 19.44 & 36.99 & \textbf{38.38} \\
\bottomrule
\end{tabular}
\caption{Per-category pixel-level F1 (\%) across all five detectors on MVTec AD 2. The Mean row averages over the eight categories; bold marks the best synthetic source per detector (the Real oracle is excluded).}
\label{tab:supp-pixel-f1}
\end{table}

## Table S3 — Software and Hardware Environment

\begin{table}[t]
\centering
\scriptsize
\begin{tabular}{ll}
\toprule
\textbf{Component} & \textbf{Configuration} \\
\midrule
Anomalib & 2.5.2 \\
Python & 3.13 \\
PyTorch & 2.13.0 (CUDA 13.0) \\
PyTorch Lightning & 2.6.5 \\
TorchMetrics & 1.9.0 \\
Torchvision & 0.28.0 \\
timm & 1.0.28 \\
GPU & NVIDIA RTX 3090 (24 GB) \\
CPU & Intel Core i9-10920X @ 3.50 GHz \\
OS & Ubuntu 24.04 LTS \\
\bottomrule
\end{tabular}
\caption{Software and hardware environment used for all experiments.}
\label{tab:supp-env}
\end{table}

## Table S4 — Per-Stage Synthesis Timing (FLASH)

\begin{table}[t]
\centering
\scriptsize
\begin{tabular}{lcc}
\toprule
\textbf{Stage} & \textbf{Time (ms)} & \textbf{Share (\%)} \\
\midrule
OBS foreground extraction & 154 & 26.3 \\
MRSP noise generation & 12 & 2.0 \\
Placement-mask computation & 37 & 6.3 \\
Defect placement & 5 & 0.9 \\
CIELAB colour harmonisation & 139 & 23.7 \\
Poisson blending & 212 & 36.2 \\
\midrule
Total (measured) & 586 & 100.0 \\
\bottomrule
\end{tabular}
\caption{Per-image timing breakdown of FLASH synthesis at 1024$\times$1024 resolution on a single NVIDIA RTX 3090. Stage times are means over fresh, non-cached images; the sub-stage sum (559~ms) is slightly below the measured total (586~ms) because stages overlap and cache.}
\label{tab:supp-timing}
\end{table}

\paragraph{Observations on Per-Stage Timing}
Blending dominates the per-image cost: Poisson blending (36\%) and CIELAB
colour harmonisation (24\%) together account for roughly 60\% of the 586~ms,
with OBS foreground extraction adding a further 26\%. In contrast, the
components that distinguish FLASH from ordinary copy-paste synthesis are
cheap, with MRSP noise generation, mask computation, and defect placement
together costing under 10\% of the total. The only generative step, defect
generation, is paid once per category (about 0.4~s of bank extraction plus
offline donor generation) and never repeated per image, which is what makes
the ``generate once, synthesize many'' formulation practical. For context,
FLASH is about 25$\times$ faster than the per-image generative baseline
AnoStyler (14{,}982~ms) and about 40$\times$ slower than the procedural Perlin
baseline (14.4~ms), though it operates at a higher working resolution
(1024$^2$ vs.\ 448$^2$).
