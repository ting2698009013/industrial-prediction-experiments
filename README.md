# Industrial Time-Series Prediction Experiments

[![synthetic-demo](https://github.com/ting2698009013/industrial-prediction-experiments/actions/workflows/demo.yml/badge.svg)](https://github.com/ting2698009013/industrial-prediction-experiments/actions/workflows/demo.yml)

这是我参加工业时序预测任务时留下的一组实验代码。项目目标是根据长序列过程变量，预测 `feature_62`、`feature_63` 和 `feature_64` 的未来窗口。

这不是一个成功的竞赛解法，也不代表最终效果达到了预期。我保留它，是因为其中记录了从数据清洗、模式识别、伪验证集构造，到递推预测、分块预测和分模式建模的完整探索过程。它更适合作为一次工程实验档案，而不是可直接复用的生产模型。

## 主要内容

- `src/`：数据处理、模型与预测逻辑。
- `scripts/`：训练、评估、伪验证和答案生成脚本。
- `ml_baseline/`：分模式机器学习基线与消融实验。
- `mode_detector_standalone/`：独立的运行模式识别脚本。
- `configs/`：特征选择等小型配置文件。
- `PREPROCESSING.md`：早期数据处理与实验总结。
- `工业时序预测项目工作流程备忘录.md`：完整工作流程备忘。

## 数据说明

仓库不包含竞赛原始数据、验证集、训练输出、模型权重或提交文件。运行脚本前，需要自行准备与代码期望格式一致的 NumPy 数据，并根据本地路径调整参数。

## 可复现的合成数据示例

由于原始竞赛数据不能公开，仓库提供了一个独立的小型基准：它生成带运行模式、周期项和相关传感器的合成工业过程，并在“未来外生变量已知、末段目标缺失”的设定下比较持久性、Ridge 和梯度提升模型。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-demo.txt
python demo/synthetic_benchmark.py --quick --output demo-results.json
```

该示例不声称复现原比赛数据分布；它的作用是让评估流程、时间切分和基线对比能够在没有私有数据时自动运行。GitHub Actions 会在每次推送和 Pull Request 时执行快速版本，并保存 JSON 指标。

## 环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

不同实验脚本依赖的库不完全相同；部分深度学习实验需要 PyTorch，机器学习基线可能需要 LightGBM 或 XGBoost。

## 已知限制

- 原始实验针对固定的竞赛数据形状，完整复现仍需要自行准备相同格式的数据。
- 预测结果未达到预期，仓库中的方案不应被视为竞赛最佳实践。
- 部分实验之间存在迭代和重复，这是有意保留的研究轨迹。

## 后续计划

- 统一配置与命令行入口。
- 整理实验依赖并补充可复现实验说明。
- 将可复用模块与一次性实验脚本分离。
- 在合成基准中加入缺失值、噪声漂移和运行模式突变等压力测试。
