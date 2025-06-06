# C4.5决策树 (Decision Tree C4.5) 使用指南

## 📖 简介

C4.5决策树是ID3算法的改进版本，使用信息增益比作为分裂准则，支持连续特征处理，并包含剪枝功能来防止过拟合。

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
./output/
```

### 生成的文件包括：

#### 📊 可视化图片 (15张)
- `01_simple_dataset.png` - 简单数据集散点图
- `01_simple_dataset_boundary.png` - 简单数据集决策边界
- `02_sklearn_train.png` - sklearn数据集训练结果
- `02_sklearn_test.png` - sklearn数据集测试结果  
- `02_sklearn_boundary.png` - sklearn数据集决策边界
- `03_iris_train.png` - 鸢尾花数据集训练结果
- `03_iris_test.png` - 鸢尾花数据集测试结果
- `03_iris_boundary.png` - 鸢尾花数据集决策边界
- `04_wine_train.png` - 葡萄酒数据集训练结果
- `04_wine_test.png` - 葡萄酒数据集测试结果
- `04_wine_boundary.png` - 葡萄酒数据集决策边界
- `05_unpruned.png` - 剪枝前结果
- `05_pruned.png` - 剪枝后结果
- `05_unpruned_boundary.png` - 剪枝前决策边界
- `05_pruned_boundary.png` - 剪枝后决策边界

#### 📄 报告文件
- `summary_report.txt` - 详细的实验报告和算法说明

## 📊 数据集要求

1. **特征矩阵 X**: 形状为 `(n_samples, n_features)` 的数组
2. **标签向量 y**: 支持多分类
3. **特征类型**: 支持连续和离散特征的混合
4. **自动处理**: 算法自动识别特征类型
5. **问题类型**: 适用于多分类问题

## 🔧 基本用法

```python
from decision_tree_c45 import C45
import numpy as np

# 创建数据
X = np.array([[1.2, 2.3], [2.1, 3.4], [3.5, 1.8], [4.2, 2.9]])
y = np.array([0, 1, 1, 0])

# 训练模型
c45 = C45()
c45.fit(X, y)

# 预测
predictions = c45.predict(X)
print(predictions)

# 剪枝
c45.prune_pep()
pruned_predictions = c45.predict(X)
```

## 🎯 推荐数据集

### 1. Sklearn内置数据集
```python
from sklearn.datasets import make_classification, load_iris, load_wine

# 生成分类数据
X, y = make_classification(n_samples=300, n_features=2, n_classes=3)

# 鸢尾花数据集
data = load_iris()
X, y = data.data, data.target

# 葡萄酒数据集
data = load_wine()
X, y = data.data, data.target
```

### 2. 混合特征数据
```python
import numpy as np

# 连续特征 + 离散特征
continuous_features = np.random.randn(100, 2)
discrete_features = np.random.choice(['A', 'B', 'C'], (100, 1))
X = np.column_stack([continuous_features, discrete_features])
```

## ⚙️ 算法特点

- ✅ **信息增益比**: 使用信息增益比避免偏向多值特征
- ✅ **连续特征**: 自动处理连续和离散特征
- ✅ **多叉树**: 生成多分支决策树
- ✅ **剪枝功能**: 内置PEP剪枝防止过拟合
- ✅ **多分类**: 支持多类别分类问题
- ✅ **自动选择**: 自动选择最佳分裂特征和阈值

## 🔍 输出解释

运行示例后，控制台会显示：
- **准确率**: 模型在数据上的分类准确率
- **树的节点数**: 决策树的总节点数
- **树的深度**: 决策树的最大深度
- **剪枝效果**: 剪枝前后的性能对比

## 📈 性能分析

从示例结果可以看出：
1. **简单数据集**: 准确率通常在80-95%之间
2. **真实数据集**: 在鸢尾花、葡萄酒等数据上表现优秀
3. **剪枝效果**: 剪枝可以提高泛化能力，减少过拟合
4. **连续特征**: 相比ID3能更好处理数值型特征

## 📞 注意事项

1. 算法会自动识别连续和离散特征
2. 支持字符串类型的离散特征
3. 数值型特征会自动寻找最佳分割点
4. 剪枝功能可以有效防止过拟合
5. 所有输出文件保存在 `./output/` 文件夹中

## 🔄 与其他算法对比

| 特性 | ID3 | C4.5 | CART |
|------|-----|------|------|
| 分裂准则 | 信息增益 | 信息增益比 | 基尼系数/方差 |
| 特征类型 | 仅离散 | 连续+离散 | 连续+离散 |
| 树结构 | 多叉 | 多叉 | 二叉 |
| 剪枝 | 无 | PEP剪枝 | CCP剪枝 |
| 应用 | 离散特征 | 通用分类 | 分类+回归 | 