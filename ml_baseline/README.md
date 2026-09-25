# zk_mode_models_v1

最新版分模式机器学习实验工程。它替换之前效果差的无模式递推 baseline。

## 核心设定

- 预测特征：`configs/selected_feature_indices.npy` 的 34 个特征。
- 模式表示：持续工况段展开成 `mode_0/mode_1/mode_2`，不是每个点独立乱判。
- f62：预测 `f62 - f01` 残差，最后加回未来可见 `f01`，并做残差分位数约束。
- f63/f64：Direct horizon block forecast，避免 12000 步纯递推误差累积。
- f63/f64：支持全局 mode-onehot 与分模式模型；分模式预测可与全局模型小比例融合。
- f64 交叉：`none/core/full` 三档消融。

## 推荐运行

### 0. 生成模式标签
```powershell
python scripts/00_build_mode_segments.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --out runs/modes_known
```

### 1. 全局模型 + mode one-hot
```powershell
python scripts/01_eval_mode_onehot_direct.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --labels runs/modes_known/mode_labels.npy `
  --folds 9 `
  --model lgbm `
  --fast `
  --row-stride 10 `
  --cut-stride 3000 `
  --cross core `
  --out runs/exp01_onehot_fold9
```

### 2. 分模式模型
```powershell
python scripts/02_eval_mode_specific_direct.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --labels runs/modes_known/mode_labels.npy `
  --folds 9 `
  --model lgbm `
  --fast `
  --row-stride 10 `
  --cut-stride 3000 `
  --cross core `
  --blend-global-weight 0.15 `
  --out runs/exp02_modespec_fold9
```

### 3. f64 交叉消融
```powershell
python scripts/03_ablate_f64_cross.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --labels runs/modes_known/mode_labels.npy `
  --folds 9 `
  --fast `
  --out runs/ablate_f64_cross_fold9
```

### 4. 全量训练
```powershell
python scripts/04_fit_full_mode_specific.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --labels runs/modes_known/mode_labels.npy `
  --model-dir runs/full_mode_specific_model `
  --model lgbm `
  --row-stride 5 `
  --cut-stride 3000 `
  --cross core `
  --blend-global-weight 0.15
```

### 5. 生成提交
需要提供官方 question 的模式标签，shape 为 `(10,60000)`。

```powershell
python scripts/05_predict_submit.py `
  --question "e:/中控杯/validation_question.npy" `
  --question-labels runs/question_modes/question_mode_labels.npy `
  --model-dir runs/full_mode_specific_model `
  --out runs/submit.npy
```


## v1.1 修正

预测窗口第一个点对应 `horizon=0`，最后一个点对应 `horizon=11999`。
上一版把 horizon 写成 `1..12000`，在 batch 长度 60000 时会访问 index 60000 越界。


## v1.2 新增

### f63 一次性消融实验

```powershell
python scripts/03b_ablate_f63_state.py `
  --data "e:/中控杯/output/03_outliers/data_after_outlier_handling.npy" `
  --labels runs/modes_known/mode_labels.npy `
  --folds 2,5,9 `
  --fast `
  --out runs/f63_ablation_259
```

它会一次运行：

- `onehot_base`
- `modespec_base_blend0`
- `modespec_base_blend015`
- `modespec_strong_blend0`
- `modespec_strong_blend015`

并生成：

```text
runs/f63_ablation_259/comparison.csv
```

### f63_state

`--f63-state strong` 会给 f63 额外加入多尺度历史状态特征，只使用 cut 前历史，不使用验证窗口答案。


## v1.3 修正

修复 v1.2 中 `01_eval_mode_onehot_direct.py` / `02_eval_mode_specific_direct.py`
未注册 `--f63-state` 参数的问题。


## v1.4 修正

修复 `--f63-state strong` 没有真正传入训练脚本的问题。

如果 base 和 strong 的结果完全一致，请先运行：

```powershell
python scripts/00_check_f63_state_features.py `
  --data data\processed\clean_train_wcj.npy `
  --labels runs/modes_known/mode_labels.npy
```

预期 `strong shape` 的列数应该大于 `base shape`。
