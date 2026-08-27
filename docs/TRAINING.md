# Training reference

## Layout of the scripts

`scripts/<meta-task>/` contains one script per method, plus the evaluation runner:

```
scripts/cls/         ImageNet-1K, N24News, HatefulMemes, VOC2007, SUN397
scripts/vqa/         OK-VQA, A-OKVQA, DocVQA, InfographicsVQA, ChartQA, Visual7W
scripts/retrieval/   VisDial, CIRR, VisualNews, MSCOCO t2i/i2t, NIGHTS, WebQA, OVEN, FashionIQ, EDIS, Wiki-SS-NQ
scripts/grounding/   MSCOCO, RefCOCO, RefCOCO-Matching, Visual7W-Pointing
```

Every script exports its hyperparameters at the top and then calls `main.py` with `torchrun
--nproc_per_node=8`. Change `NUM_GPUS_PER_NODE` and `BATCH_SIZE` together if you run on fewer GPUs — the
contrastive and geometric losses both operate on the gathered global batch, so the effective batch size
matters more than the per-device one.

## Shared settings

| Setting | Value |
| ------- | ----- |
| Precision | bf16, DDP |
| Optimizer | AdamW, weight decay 0.01 |
| Epochs | 1 |
| LR (student / LoRA) | 1e-4, constant schedule |
| LR (projectors) | 5e-4 |
| Teacher | `raghavlite/B3_Qwen2_2B`, frozen, `--teacher_lora_r 8`, eos pooling |
| Seed | 42 |

## Per-track hyperparameters

| Track | Student | Batch/GPU | Global batch | LoRA r | LoRA α | λ_MSE | λ_PGA | λ_SCL | η |
| ----- | ------- | --------- | ------------ | ------ | ------ | ----- | ----- | ----- | - |
| CLS | FastVLM | 32 | 256 | 64 | 128 | 1.0 | 2.0 | 0.01 | 0.85 |
| VQA | FastVLM | 32 | 256 | 128 | 512 | 1.0 | 2.0 | 0.01 | 0.85 |
| RET | FastVLM | 16 | 128 | 128 | 128 | 1.0 | 2.0 | 0.01 | 0.85 |
| GRD | FastVLM | 16 | 128 | 128 | 256 | 1.0 | 2.0 | 0.01 | 0.85 |
| CLS | OneVision | 8 | 64 | 16 | 32 | 1.0 | 2.0 | 0.01 | 0.85 |
| VQA | OneVision | 16 | 128 | 32 | 128 | 1.0 | 2.0 | 0.01 | 0.85 |
| RET | OneVision | 16 | 128 | 32 | 128 | 1.0 | 2.0 | 0.01 | 0.85 |
| GRD | OneVision | 16 | 128 | 32 | 128 | 1.0 | 2.0 | 0.01 | 0.85 |

Larger LoRA ranks help on VQA, retrieval and grounding, where the student has to keep more of the teacher's
fine-grained structure; classification is happy with r = 64.

## PGA-KD flags

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--kd_loss_type pga` | – | picks `PGAKDLoss` from the registry in `src/criterions/__init__.py` |
| `--pga_loss_weight` | 1.0 | λ_PGA; 2.0 in all released scripts |
| `--pga_scl_loss_weight` | 1.0 | λ_SCL; 0.01 in all released scripts |
| `--pga_mse_loss_weight` | 1.0 | λ_MSE on the projected teacher embedding |
| `--pga_spectral_variance_threshold` | 0.95 | η; 0.85 in all released scripts |
| `--projector_config_path` | – | `config/projector_config.json`, enables the `t2s` / `t2s_img` / `t2s_txt` heads |
| `--projector_lr` | 5e-4 | separate param group added by the criterion |

The SCL projection heads are created lazily on the first forward pass (teacher and student hidden sizes are
only known then) and registered into the optimizer by `_add_optimizer_param_group`. They are training-only:
inference uses the student alone.

## Baselines

Same folders, same interface, only `--kd_loss_type` and the method-specific weights change:

| Script | `kd_loss_type` | Implementation |
| ------ | -------------- | -------------- |
| `train_student.sh` | `contrastive` | `src/criterions/contrastive.py` |
| `train_MSE.sh` | `mse` | `src/criterions/mse.py` |
| `train_RKD.sh` | `contrastive_rkd` | `src/criterions/contrastive_loss_with_RKD.py` |
| `train_CKD.sh` | `ckd` | `src/criterions/ckd.py` |
| `train_EMO.sh` | `emo_loss` | `src/criterions/emo_loss.py` |
| `train_EMKD.sh` | `em_kd` | `src/criterions/em_kd.py` |
| `train_HOLO.sh` | `holo` | `src/criterions/holo.py` |

To add another one, implement a `nn.Module` with the same `forward(distiller, input_data) -> dict` contract
as `PGAKDLoss` and register it in `criterion_list`.

## Evaluation

`scripts/<task>/run_eval.sh` reads `EXP_NAME`, resolves the checkpoint at
`training/$EXP_NAME/checkpoint-final`, and runs `eval_mmeb.py` over the subsets of that task with
`accelerate launch --multi_gpu`. Results land in `eval_outputs/$EXP_NAME`.

Set `SUBSET="all"` to score the full 36-dataset MMEB suite, including the OOD splits (Place365, ImageNet-A,
ImageNet-R, ObjectNet, Country211, ScienceQA, VizWiz, GQA, TextVQA).

Aggregation helpers:

```bash
python scripts/parse_eval_results.py     # eval_outputs/* -> one table
python scripts/log_eval_results.py       # push a run to the tracking sheet
python scripts/compare_performance.py    # side-by-side comparison across runs
```

## Notes

- `config/ds_config*.json` hold DeepSpeed ZeRO-3 configs; they are optional and unused by the released
  scripts, which rely on plain DDP.
- Teacher and student use different tokenizers, so SCL rebuilds the image/text masks after token expansion
  instead of reusing the input mask — see `_get_image_text_representation` in `src/criterions/pga.py`.
- Multi-image samples are not handled: the VLM2Vec setting assumes at most one image per sample.
