# Table 5 — Temporal faithfulness on the expert set (%)

Expert set sha b7bc34e99217. BAcc and AUC are computed on all decided pairs; the per-type columns follow Sect. 4.4: V = recall on valid rewrites, I = recall on invalid rewrites (rejected). Threshold P(faithful) ≥ 0.5.

| | Model | BAcc | AUC | Gen BAcc | Gen V | Gen I | Prec BAcc | Prec V | Prec I | Reform V | Incorr I |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| English — Baselines | Zero-shot mDeBERTa | 69.0 | 75.6 | 54.4 | 58.1 | 50.7 | 72.9 | 50.0 | 95.7 | 66.2 | 91.2 |
| English — Baselines | Zero-shot MiniCheck | 65.6 | 65.0 | 49.1 | 43.0 | 55.2 | 65.6 | 33.3 | 97.9 | 64.8 | 92.6 |
| English — Baselines | Qwen3-4B | 72.9 | – | 61.3 | 73.3 | 49.3 | 63.8 | 100.0 | 27.7 | 97.2 | 95.6 |
| English — Baselines | Qwen3-30B-A3B | 69.7 | – | 58.1 | 74.4 | 41.8 | 55.3 | 100.0 | 10.6 | 95.8 | 97.1 |
| English — FT: criterion + silver labels (ours) | mDeBERTa | 80.0 | 90.5 | 76.0 | 64.0 | 88.1 | 70.7 | 50.0 | 91.5 | 78.9 | 94.1 |
| English — FT: criterion + silver labels (ours) | MiniCheck | 81.4 | 89.9 | 75.5 | 67.4 | 83.6 | 78.1 | 58.3 | 97.9 | 80.3 | 94.1 |
| English — Ablation: criterion labels only | mDeBERTa | 79.4 | 86.7 | 69.0 | 66.3 | 71.6 | 74.9 | 58.3 | 91.5 | 87.3 | 92.6 |
| English — Ablation: criterion labels only | MiniCheck | 78.3 | 87.5 | 69.5 | 62.8 | 76.1 | 78.1 | 58.3 | 97.9 | 77.5 | 94.1 |
| English — Ablation: generator labels | mDeBERTa | 75.8 | 86.2 | 58.5 | 76.7 | 40.3 | 86.3 | 83.3 | 89.4 | 90.1 | 83.8 |
| English — Ablation: generator labels | MiniCheck | 72.1 | 84.7 | 49.2 | 68.6 | 29.9 | 88.5 | 83.3 | 93.6 | 88.7 | 83.8 |
| English (Sect. 6.2) — FT with 3,000 MNLI pairs added | mDeBERTa | 81.4 | 88.8 | 76.8 | 64.0 | 89.6 | 71.8 | 50.0 | 93.6 | 83.1 | 94.1 |
| English (Sect. 6.2) — FT with 3,000 MNLI pairs added | MiniCheck | 80.6 | 89.2 | 74.5 | 64.0 | 85.1 | 72.9 | 50.0 | 95.7 | 84.5 | 91.2 |
| French — Baseline | mDeBERTa | 68.6 | 73.6 | 53.4 | 70.9 | 35.8 | 78.0 | 66.7 | 89.4 | 73.2 | 79.4 |
| French — FT (EN) | mDeBERTa | 73.2 | 84.1 | 62.2 | 55.8 | 68.7 | 69.7 | 50.0 | 89.4 | 74.6 | 94.1 |
| French — FT (FR) | mDeBERTa | 71.1 | 75.1 | 47.0 | 68.6 | 25.4 | 81.1 | 75.0 | 87.2 | 94.4 | 82.4 |
| French — FT (EN → FR) | mDeBERTa | 78.4 | 87.7 | 70.2 | 62.8 | 77.6 | 69.7 | 50.0 | 89.4 | 81.7 | 95.6 |
