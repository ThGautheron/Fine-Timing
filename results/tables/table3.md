# Table 3 — Verdicts on rewrites of «The treaty was signed in 1294.»

| Rewrite of 1294 | Type | Gran. | Criterion | mDeBERTa | MiniCheck | Qwen3-30B-A3B |
|---|---|---|:-:|:-:|:-:|:-:|
| the 1290s | generalization | decade | ✓ | ✓ (0.98) | ✓ (0.90) | ✓ |
| the 13th century | generalization | century | ✓ | ✓ (0.99) | ✓ (0.91) | ✓ |
| the High Middle Ages | wrong_date | period | ✗ | ✓ (0.86) | ✓ (0.79) | ✓ |
| the Middle Ages | generalization | period | ✗ | ✓ (0.90) | ✓ (0.96) | ✓ |
| the summer of 1294 | precision | month | ✗ | ✗ (0.00) | ✗ (0.02) | ✓ |
| anno Domini 1294 | reformulation | year | ✓ | ✓ (0.77) | ✗ (0.10) | ✓ |
| 1295 | wrong_date | year | ✗ | ✗ (0.03) | ✗ (0.11) | ✗ |
