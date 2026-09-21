# Configuration names

Paths are relative to `config/experiment/`. Executable leaves compose the shared dataset, method, backbone and schedule groups listed below. See [REPRODUCE.md](REPRODUCE.md) for commands and checkpoint availability.

| Configuration | Paper row / role |
|---|---|
| `pavesplat_vits_re10k.yaml` | Supplement Table 4; rebuttal Table B |
| `pavesplat_vitb_re10k.yaml` | Supplement Table 4; rebuttal Table B |
| `pavesplat_vitl_re10k.yaml` | Main Tables 1, 3, 4; supplement Tables 3–4 / Figures 2–6; rebuttal Table B |
| `surfsplat_vits_re10k.yaml` | Supplement Table 4; rebuttal Table B |
| `surfsplat_vitl_re10k.yaml` | Main Tables 1, 3, 4; supplement Tables 3–4 / Figures 2–6; rebuttal Table B |
| `ablation_tangent_frame_vitl_re10k.yaml` | Main Table 4: + tangent frame; rebuttal black-hole analysis |
| `zeroshot_acid_pavesplat_vitl.yaml` | Supplement Table 1: ACID zero-shot |
| `zeroshot_acid_surfsplat_vitl.yaml` | Supplement Table 1: ACID zero-shot |
| `zeroshot_dl3dv_pavesplat_vitl.yaml` | Main Table 2: DL3DV zero-shot |
| `zeroshot_dl3dv_surfsplat_vitl.yaml` | Main Table 2: DL3DV zero-shot |
| `zeroshot_scannet_pavesplat_vitl.yaml` | Supplement Tables 1–2 / Figure 5; rebuttal Table A |
| `zeroshot_scannet_surfsplat_vitl.yaml` | Supplement Tables 1–2 / Figure 5; rebuttal Table A |
| `_base/re10k.yaml` | Shared RE10K recipe: main Tables 1, 3, 4; supplement Tables 3–4; rebuttal Table B |
| `_base/acid.yaml` | Shared ACID protocol: supplement Table 1 |
| `_base/dl3dv.yaml` | Shared DL3DV protocol: main Table 2 |
| `_base/scannet.yaml` | Shared ScanNet protocol: supplement Tables 1–2; rebuttal Table A |
| `_method/pavesplat.yaml` | PaveSplat rows in main Tables 1–4, supplement Tables 1–4 and rebuttal Tables A–B |
| `_method/surfsplat.yaml` | SurfSplat‡ rows in main Tables 1–4, supplement Tables 1–4 and rebuttal Tables A–B |
| `_method/tangent_frame_only.yaml` | Main Table 4: + tangent frame; rebuttal black-hole analysis |
| `_backbone/vits.yaml` | ViT-S rows: supplement Table 4; rebuttal Table B |
| `_backbone/vitb.yaml` | ViT-B rows: supplement Table 4; rebuttal Table B |
| `_backbone/vitl.yaml` | ViT-L rows: main Tables 1–4, supplement Tables 1–4 and rebuttal Tables A–B |
| `_schedule/re10k_150k.yaml` | Shared RE10K training recipe for the retained paper rows |
