# Results

All numbers are MMEB Precision@1, teacher is `raghavlite/B3_Qwen2_2B` (d = 1536). "Student" is the same
backbone fine-tuned contrastively without any distillation loss.

## In-distribution, FastVLM-0.5B student

### Classification and VQA

| Method | IN-1K | N24News | Hateful | VOC07 | SUN397 | **Avg** | OK | A-OK | Doc | I-VQA | Chart | Vis7W | **Avg** |
| ------ | ----- | ------- | ------- | ----- | ------ | ------- | -- | ---- | --- | ----- | ----- | ----- | ------- |
| Teacher | 82.9 | 79.2 | 56.2 | 88.0 | 80.9 | 77.4 | 63.0 | 53.5 | 92.3 | 58.9 | 53.1 | 53.1 | 62.3 |
| Student | 52.9 | 68.4 | 60.0 | 77.7 | 64.7 | 64.7 | 50.7 | 50.2 | 74.0 | 34.1 | 49.4 | 46.1 | 50.8 |
| MSE | 53.0 | 70.1 | 58.4 | 78.2 | 65.2 | 65.0 | 52.4 | 49.8 | 74.3 | 34.3 | 50.8 | 46.4 | 51.3 |
| RKD | 53.6 | 70.3 | 60.8 | 79.9 | 66.2 | 66.2 | 51.1 | 51.2 | 74.1 | 34.9 | 50.2 | 46.5 | 51.3 |
| CKD | 53.2 | 70.6 | 61.2 | 78.6 | 66.7 | 66.1 | 51.8 | 51.0 | 74.7 | 35.0 | 49.6 | 46.2 | 51.4 |
| EMO | 52.4 | 68.4 | 59.1 | 80.4 | 59.3 | 63.9 | 50.9 | 50.1 | 73.1 | 33.8 | 49.0 | 47.8 | 50.8 |
| EM-KD | 53.4 | 67.3 | 59.4 | 78.0 | 63.2 | 64.3 | 51.1 | 49.3 | 77.2 | 36.7 | 47.6 | 47.3 | 51.5 |
| **PGA-KD** | **62.6** | 70.4 | 56.9 | **81.9** | **75.1** | **69.4** | **59.5** | **54.6** | 74.6 | **38.4** | 44.6 | **52.3** | **54.0** |

### Retrieval and Grounding

| Method | VisDial | CIRR | VN-t2i | VN-i2t | COCO-t2i | COCO-i2t | NIGHTS | WebQA | **Avg** | COCO (GRD) |
| ------ | ------- | ---- | ------ | ------ | -------- | -------- | ------ | ----- | ------- | ---------- |
| Teacher | 83.4 | 60.3 | 76.0 | 79.9 | 76.6 | 49.7 | 68.7 | 88.6 | 72.9 | 70.5 |
| Student | 61.2 | 32.9 | 46.8 | 49.6 | 65.2 | 58.6 | 62.6 | 69.4 | 55.8 | 65.0 |
| MSE | 64.2 | 27.7 | 46.8 | 51.6 | 69.6 | 59.0 | 64.0 | 72.5 | 56.9 | 68.3 |
| RKD | 62.1 | 32.1 | 45.7 | 50.9 | 64.8 | 58.3 | 63.0 | 70.6 | 55.9 | 68.1 |
| CKD | 61.5 | 28.9 | 45.0 | 47.7 | 66.0 | 59.2 | 63.0 | 69.8 | 56.2 | 68.7 |
| EMO | 41.7 | 26.7 | 0.5 | 22.6 | 3.9 | 33.3 | 62.4 | 3.0 | 24.3 | 66.8 |
| EM-KD | 61.5 | **36.5** | 47.3 | 49.8 | 68.4 | 58.5 | 63.5 | **73.0** | 57.3 | 69.2 |
| **PGA-KD** | **73.3** | 35.1 | **50.8** | **55.6** | 66.1 | **63.5** | **64.3** | 72.5 | **60.2** | **69.5** |

EMO collapses on retrieval because its loss only supervises text tokens; the image branch drifts and the
joint contrastive space breaks apart. The failure does not show up on classification, VQA or grounding,
which behave as autoregressive text tasks.

## In-distribution, LLaVA-OneVision-0.5B student

| Method | IN-1K | N24News | Hateful | VOC07 | SUN397 | **Avg** | OK | A-OK | Doc | I-VQA | Chart | Vis7W | **Avg** |
| ------ | ----- | ------- | ------- | ----- | ------ | ------- | -- | ---- | --- | ----- | ----- | ----- | ------- |
| Teacher | 82.9 | 79.2 | 56.2 | 88.0 | 80.9 | 77.4 | 63.0 | 53.5 | 92.3 | 58.9 | 53.1 | 53.1 | 62.3 |
| Student | 55.6 | 67.1 | 57.0 | 83.9 | 66.0 | 65.9 | 48.2 | 42.8 | 38.6 | 21.3 | 25.1 | 43.8 | 36.6 |
| MSE | 55.3 | 66.8 | 56.2 | **84.1** | 66.8 | 65.8 | 47.8 | 42.5 | 37.9 | 21.0 | 24.7 | 42.9 | 36.1 |
| RKD | 55.1 | 66.8 | 57.6 | 83.7 | 67.0 | 66.0 | 47.8 | 43.7 | 38.6 | 21.6 | 25.5 | 43.5 | 36.8 |
| CKD | 57.0 | 66.9 | 58.6 | **84.1** | 68.2 | 66.9 | 51.2 | 43.9 | **47.2** | 22.9 | **30.9** | **45.6** | 40.3 |
| EMO | 56.1 | 66.4 | 56.4 | 83.6 | 66.2 | 65.7 | 46.4 | 43.6 | 41.1 | 20.7 | 23.1 | 43.0 | 36.3 |
| EM-KD | 53.2 | 63.3 | 53.6 | 82.6 | 65.2 | 63.6 | 44.7 | 37.9 | 36.4 | 20.5 | 22.6 | 39.8 | 33.7 |
| **PGA-KD** | **66.5** | **68.9** | **59.5** | 83.9 | **73.3** | **70.4** | **53.6** | **47.2** | 46.0 | **23.9** | 29.7 | 45.4 | **41.0** |

## Out-of-distribution, FastVLM-0.5B student

| Method | Place365 | IN-A | IN-R | ObjectNet | Country211 | **Avg** | ScienceQA | VizWiz | GQA | TextVQA | **Avg** |
| ------ | -------- | ---- | ---- | --------- | ---------- | ------- | --------- | ------ | --- | ------- | ------- |
| Teacher | 44.7 | 49.7 | 90.4 | 71.7 | 25.7 | 56.4 | 40.7 | 48.4 | 68.1 | 80.0 | 59.3 |
| Student | 30.0 | 25.3 | 48.2 | 28.4 | 7.8 | 27.9 | 35.8 | 35.5 | 63.4 | 56.7 | 47.9 |
| MSE | 30.4 | 25.9 | 52.7 | 25.5 | 8.0 | 28.5 | **37.9** | **37.7** | 63.5 | 61.5 | 50.2 |
| RKD | 31.1 | 25.9 | 47.5 | 27.8 | **8.4** | 28.1 | 36.0 | 36.2 | 64.6 | 57.0 | 48.5 |
| CKD | 30.3 | 25.4 | 47.9 | 31.6 | 7.5 | 28.5 | 36.3 | 35.8 | 64.1 | 56.6 | 48.2 |
| EMO | 34.8 | 21.8 | 52.5 | 37.4 | 5.2 | 30.3 | 36.6 | 36.1 | 65.5 | 57.7 | 49.0 |
| EM-KD | 32.5 | **27.4** | 50.4 | 32.2 | 7.6 | 30.0 | 36.5 | 36.1 | 65.4 | 57.6 | 48.9 |
| **PGA-KD** | **38.1** | 24.1 | **62.6** | **40.7** | 7.5 | **33.7** | 36.9 | 36.8 | **67.0** | **66.4** | **51.8** |

## Ablations (FastVLM-0.5B)

`Avg. VQA` is the average over the six in-distribution VQA subsets from the table above.

### Loss components

| Configuration | OK | A-OK | Doc | I-VQA | Avg. VQA |
| ------------- | -- | ---- | --- | ----- | -------- |
| Student | 50.7 | 50.2 | 74.0 | 34.1 | 50.8 |
| w/o L_PGA | 57.1 | 54.2 | 67.7 | 37.6 | 51.6 |
| w/o L_intra | 58.0 | 54.3 | 73.9 | 35.7 | 53.1 |
| w/o L_inter | 58.1 | 54.7 | 73.3 | 39.3 | 52.5 |
| w/o L_PGA, L_intra | 56.6 | 53.1 | 68.4 | 38.1 | 51.5 |
| w/o L_PGA, L_inter | 57.8 | 54.6 | 67.9 | 37.0 | 51.6 |
| w/o L_intra, L_inter | 58.5 | 53.6 | 74.8 | 37.1 | 53.2 |
| **Full** | 59.5 | 54.6 | 74.6 | 38.4 | **54.0** |

Dropping L_PGA costs 2.4 points, the largest single drop: matching final representations alone does not
transfer the teacher's batch geometry. The two SCL pathways contribute 0.9 (intra) and 1.5 (inter).

### Spectral threshold η

| η | OK | A-OK | Doc | I-VQA | Avg. VQA |
| - | -- | ---- | --- | ----- | -------- |
| 1.00 (no filtering) | 56.9 | 54.2 | 73.8 | 37.0 | 52.7 |
| 0.95 | 58.3 | **55.7** | **75.3** | 38.2 | 53.9 |
| **0.85 (ours)** | **59.5** | 54.6 | 74.6 | **38.4** | **54.0** |
| 0.75 | 57.9 | 53.5 | 73.6 | 36.8 | 52.5 |
| 0.50 | 56.2 | 51.6 | 66.5 | 36.2 | 50.0 |

An inverted-U curve. Keeping the whole spectrum drags in non-transferable noise; cutting below ~0.75 removes
real structure. Note that η = 0.95 is the better choice for document-heavy workloads (best DocVQA).

### Loss weights

| λ_PGA | λ_SCL | λ_MSE | OK | A-OK | Doc | I-VQA | Avg. VQA |
| ----- | ----- | ----- | -- | ---- | --- | ----- | -------- |
| 0.1 | 0.01 | 1.0 | 57.3 | 54.7 | 69.4 | 37.5 | 52.3 |
| 0.5 | 0.01 | 1.0 | 58.8 | 54.6 | 70.9 | 37.1 | 52.8 |
| 2.0 | 0.05 | 1.0 | 58.9 | 55.4 | 75.1 | 36.8 | 54.0 |
| 2.0 | 0.10 | 1.0 | 58.2 | 55.5 | 74.2 | 36.9 | 53.6 |
| 2.0 | 0.01 | 0.2 | 57.4 | 54.3 | 75.8 | 37.9 | 53.4 |
| 2.0 | 0.01 | 0.5 | 58.9 | 54.1 | 75.8 | 38.4 | 53.9 |
| **2.0** | **0.01** | **1.0** | 59.5 | 54.6 | 74.6 | 38.4 | **54.0** |

λ_SCL is deliberately small: the objective sums four InfoNCE terms, so a larger weight saturates the
gradient budget without improving the average.

## Seed variance

Three seeds ({42, 1234, 2026}) on the OOD VQA track. The average here covers only these three tasks and is
not comparable to the four-task OOD average above.

| Method | ScienceQA | GQA | TextVQA | Avg |
| ------ | --------- | --- | ------- | --- |
| Teacher | 40.7 | 68.1 | 80.0 | 62.9 |
| Student | 35.8 | 63.4 | 56.7 | 52.0 |
| MSE | 37.0 ±1.1 | 64.0 ±1.6 | 60.1 ±1.5 | 53.7 ±0.8 |
| EM-KD | 33.8 ±2.4 | 64.3 ±1.1 | 56.2 ±1.2 | 51.4 ±1.5 |
| **PGA-KD** | 37.0 ±0.5 | **66.9 ±0.8** | **67.0 ±1.4** | **57.0 ±0.4** |

PGA-KD's worst single run (56.8) still beats every baseline's mean, and it has the smallest spread of the
three distillation methods.

## Training overhead

Single training step on the grounding track, batch size 16.

| Method | Step (ms) | Overhead |
| ------ | --------- | -------- |
| Student (baseline) | 1976 | 1.00× |
| + MSE | 2475 | 1.25× |
| + RKD | 2332 | 1.18× |
| + CKD | 2768 | 1.40× |
| + EM-KD | 2473 | 1.25× |
| **+ PGA-KD** | 2424 | 1.23× |

The eigendecomposition is O(B³) on a B × B matrix, which is under 1 ms per step at B = 64; truncating the
spectrum makes the downstream alignment cheaper than full-rank MSE or CKD.

## Figures

| File | Content |
| ---- | ------- |
| `asset/overview.png` | framework diagram |
| `asset/spectrum_eta_85.png` | eigenvalue decay and cumulative energy at B = 64 |
| `asset/tsne_pro_layout.png` | t-SNE of the batch geometry before and after filtering |
| `asset/spectral_filter_effect.png` | reconstructed Gram matrix and removed noise for several η |
