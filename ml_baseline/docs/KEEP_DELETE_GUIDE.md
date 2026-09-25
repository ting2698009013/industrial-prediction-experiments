# 当前项目建议保留 / 删除文件

## 必须保留

- 官方训练集原始 npy：`"中控杯"工业大模型初赛训练集.npy` 或 `data/raw/train.npy`
- 清洗后数据：`output/03_outliers/data_after_outlier_handling.npy`
- 学长给的预测特征：`selected_feature_indices.npy`
- `PREPROCESSING.md`
- `optimization/extreme_ml.py`
- 模式识别结果：`mode_labels.npy`、`mode_segments.json`、`mode_labels_classifier_summary.json`
- 新工程：`zk_mode_models_v1/`

## 可以删或归档

- `zhongkong_ml_baseline/`
- `zhongkong_ml_baseline_v2/`
- `zhongkong_ml_baseline_v3/`
- `zk_competition_v4/`
- `verify_zhongkong_progress.py`
- 之前无模式递推 baseline 的 `runs/baseline_*`、`runs/debug_*`、`runs/fold9_*`
- 旧的低分 `submit.npy`

## 暂时不要删

- `output/02_modes/`：如果存在，可能有学长原模式标签。
- `output/04_features/`：特征选择结果。
- `output/05_baseline/importance_feature_*.csv`：后续筛 f64 交叉有用。
