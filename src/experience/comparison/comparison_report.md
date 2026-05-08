# ICDv3 vs ICDv4 – Comparison Report

**ICDv3 checkpoint**: `/mnt/c/Users/Admin/HUIT - Học Tập/Năm 3/Semester_2/Class/Capstone Project/result/ICD/checkpoints/best_model_v3.pth`
**ICDv4 checkpoint**: `src/experience/icdv4/checkpoints/ICDv4_9FL/icdv4_best.pt`

## Global Metrics

|           |   ICDv3 |   ICDv4 |   Δ (v4-v3) |
|:----------|--------:|--------:|------------:|
| Accuracy  |  0.8363 |  0.8647 |      0.0284 |
| Precision |  0.7065 |  0.7871 |      0.0806 |
| Recall    |  0.8125 |  0.7722 |     -0.0403 |
| F1        |  0.7558 |  0.7796 |      0.0237 |
| Brier     |  0.1188 |  0.1065 |      0.0123 |

## McNemar's Test

| b | c | chi2 | p-value | Significant |
|---|---|------|---------|-------------|
| 12 | 27 | 5.0256 | 0.0250 | Yes |


## Conclusion

✅ ICDv4 cải thiện F1 từ **0.7558** lên **0.7796** (+0.0237)
