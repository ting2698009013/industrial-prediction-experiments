1. 文件放置位置
把脚本放到：
D:\industrial_prediction\scripts\preprocess_train_v1.py
训练集放到：
D:\industrial_prediction\data\raw\train.npy
2. 安装依赖
先进入项目根目录：
cd D:\industrial_prediction
激活虚拟环境：
.\.venv\Scripts\Activate.ps1
安装依赖：
python -m pip install numpy pandas scikit-learn joblib
3. 全量运行：生成清洗数据 + 完整特征
python .\scripts\preprocess_train_v1.py `
  --input .\data\raw\train.npy `
  --out-dir .\data\processed `
  --feature-set all `
  --save-cleaned `
  --prefix v1
运行后会输出：
data/processed/
├── cleaned_train_v1.npy
├── y_targets_v1.npy
├── X_all_v1.npy
├── feature_names_all_v1.json
└── preprocess_summary_v1.json
4. 输出文件说明
cleaned_train_v1.npy
清洗后的训练集。
形状：
(65, 10, 60000)
含义：
65 个特征 × 10 个 batch × 每个 batch 60000 个时间点
y_targets_v1.npy
目标列，也就是需要预测的三列：
feature_62, feature_63, feature_64
形状：
(3, 10, 60000)
X_all_v1.npy
特征工程后的输入特征。
包含：
原始输入特征
diff1
diff5
lag1
lag5
lag10
lag30
rolling_mean10
rolling_mean30
rolling_mean60
不包含当前时刻的 feature_62, feature_63, feature_64。
feature_names_all_v1.json
记录 X_all_v1.npy 每一列对应的特征名。
preprocess_summary_v1.json
记录本次预处理配置，包括：
删除了哪些常量列
使用了哪些输入列
做了哪些特征工程
输出文件路径
5. 只生成某一种特征工程版本
只生成原始输入特征
python .\scripts\preprocess_train_v1.py `
  --input .\data\raw\train.npy `
  --out-dir .\data\processed `
  --feature-set base `
  --prefix v1
输出：
X_base_v1.npy
feature_names_base_v1.json
生成差分特征版本
python .\scripts\preprocess_train_v1.py `
  --input .\data\raw\train.npy `
  --out-dir .\data\processed `
  --feature-set diff `
  --prefix v1
输出：
X_diff_v1.npy
feature_names_diff_v1.json
生成滞后特征版本
python .\scripts\preprocess_train_v1.py `
  --input .\data\raw\train.npy `
  --out-dir .\data\processed `
  --feature-set lag `
  --prefix v1
输出：
X_lag_v1.npy
feature_names_lag_v1.json
生成滑动均值特征版本
python .\scripts\preprocess_train_v1.py `
  --input .\data\raw\train.npy `
  --out-dir .\data\processed `
  --feature-set rolling `
  --prefix v1
输出：
X_rolling_v1.npy
feature_names_rolling_v1.json