# 决策树桩 (Decision Stump) 使用指南

## 📖 简介

决策树桩是一种只有一个分裂节点的简单决策树，通常用作集成学习算法（如AdaBoost）的弱学习器。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
python main.py
```

## 📁 输出文件夹

运行程序后，所有生成的图片和报告将保存在：
```
./output/decision_stump_example/
```

### 生成的文件包括：

#### 📊 可视化图片 (13张)
- `01_simple_dataset.png` - 简单数据集散点图
- `01_simple_dataset_boundary.png` - 简单数据集决策边界
- `02_sklearn_train.png` - sklearn数据集训练结果
- `02_sklearn_test.png` - sklearn数据集测试结果  
- `02_sklearn_boundary.png` - sklearn数据集决策边界
- `03_breast_cancer_train.png` - 乳腺癌数据集训练结果
- `03_breast_cancer_test.png` - 乳腺癌数据集测试结果
- `03_breast_cancer_boundary.png` - 乳腺癌数据集决策边界
- `04_uniform_weights.png` - 均匀权重训练结果
- `04_weighted_training.png` - 加权训练结果
- `04_uniform_weights_boundary.png` - 均匀权重决策边界
- `04_weighted_training_boundary.png` - 加权训练决策边界

#### 📄 报告文件
- `summary_report.txt` - 详细的实验报告和算法说明

## 📊 数据集要求

1. **特征矩阵 X**: 形状为 `(n_samples, n_features)` 的数组
2. **标签向量 y**: 值必须为 `-1` 或 `1` 的二分类标签
3. **权重向量 w**: 长度为 `n_samples`，通常归一化到和为1
4. **数据类型**: 支持连续值特征
5. **问题类型**: 仅适用于二分类问题

## 🔧 基本用法

```python
from decision_stump import DecisionStump
import numpy as np

# 创建数据
X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
y = np.array([1, 1, -1, -1])
w = np.ones(len(y)) / len(y)  # 均匀权重

# 训练模型
stump = DecisionStump()
stump.fit(X, y, w)

# 预测
predictions = stump.predict(X)
print(predictions)
```

## 🎯 推荐数据集

### 1. Sklearn内置数据集
```python
from sklearn.datasets import make_classification, load_breast_cancer, load_wine

# 生成分类数据
X, y = make_classification(n_samples=200, n_features=2, n_classes=2)

# 乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 葡萄酒数据集（选择两个类别）
data = load_wine()
X, y = data.data, data.target
```

### 2. 自定义数据集
```python
import numpy as np

# 线性可分数据
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
```

## ⚙️ 算法特点

- ✅ **简单有效**: 只有一个分裂节点，计算快速
- ✅ **支持权重**: 适用于AdaBoost等集成算法
- ✅ **自动特征选择**: 自动选择最佳分裂特征和阈值
- ✅ **可视化友好**: 生成清晰的决策边界图
- ⚠️ **仅二分类**: 只适用于二分类问题
- ⚠️ **弱学习器**: 单独使用效果有限，适合集成使用

## 🔍 输出解释

运行示例后，控制台会显示：
- **准确率**: 模型在数据上的分类准确率
- **选择的特征**: 用于分裂的特征索引
- **阈值**: 分裂的阈值
- **方向**: 分裂方向（greater/less）

## 📈 性能分析

从示例结果可以看出：
1. **简单数据集**: 准确率约82%，表现良好
2. **生成数据集**: 训练和测试准确率都在85%以上
3. **真实数据集**: 在复杂数据上表现有限，适合作为弱学习器使用
4. **权重影响**: 不同权重设置会产生不同的分裂策略

## 📞 注意事项

1. 确保标签值为 -1 和 1
2. 权重向量应该归一化
3. 特征值建议标准化处理
4. 适合作为集成学习的基学习器使用
5. 所有输出文件保存在 `./output/decision_stump_example/` 文件夹中 