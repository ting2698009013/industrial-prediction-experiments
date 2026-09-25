# Standalone Mode Detector

这个小工具只负责模式识别：给它一个 `.npy` 数据集，它会输出 mode labels、模式切换时间点、模式区间和诊断文件。

支持两种输入：

- 训练集格式：`(N, 65)`，例如 `clean_train_wcj.npy`
- 官方 question 格式：`(65, batch, batch_len)`，例如 `(65, 10, 60000)`

默认不会使用目标列 `62/63/64` 做模式识别。

## PowerShell / VSCode 用法

```powershell
python detect_modes_standalone.py `
--input "D:\data\clean_train_wcj.npy" `
--out "runs\modes_train" `
--features auto `
--sample-stride 10 `
--smooth-win auto
```

换成验证集或测试集 question 也一样：

```powershell
python detect_modes_standalone.py `
--input "D:\data\question.npy" `
--out "runs\modes_question" `
--features auto `
--sample-stride 10 `
--smooth-win auto
```

PowerShell 的反引号 `` ` `` 后面不要有空格。

## 输出文件

- `mode_labels.npy`：一维展开后的模式标签，长度为总时间长度。
- `mode_labels_batched.npy`：只有输入为 `(features,batch,batch_len)` 时生成，shape 为 `(batch,batch_len)`。
- `mode_raw_labels.npy`：GMM 原始标签，未平滑。
- `mode_segments.csv`：连续 mode 区间。
- `mode_change_points.csv`：mode 切换点。
- `mode_feature_separation.csv`：各特征对 mode 的区分度。
- `mode_model.joblib`：本次识别拟合出来的 scaler + GMM。
- `summary.json`：整体摘要。

## 依赖

```powershell
pip install numpy pandas scikit-learn joblib
```
