# CART决策树 (Decision Tree CART) 使用指南

## 📖 简介

CART（Classification and Regression Trees）是一种既可以用于分类也可以用于回归的决策树算法。它生成二叉树，使用基尼系数（分类）或方差（回归）作为分裂准则。

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

#### 📊 可视化图片 (18张)

**分类模式:**
- `01_classification_simple.png` - 简单分类数据集结果
- `01_classification_simple_boundary.png` - 简单分类数据集决策边界
- `02_classification_train.png` - sklearn分类训练结果
- `02_classification_test.png` - sklearn分类测试结果
- `02_classification_boundary.png` - sklearn分类决策边界
- `03_iris_train.png` - 鸢尾花训练结果
- `03_iris_test.png` - 鸢尾花测试结果
- `03_iris_boundary.png` - 鸢尾花决策边界

**回归模式:**
- `04_regression_simple.png` - 简单回归数据集结果
- `05_regression_train.png` - sklearn回归训练结果
- `05_regression_test.png` - sklearn回归测试结果

**剪枝对比:**
- `06_cart_unpruned.png` - 剪枝前结果
- `06_cart_pruned.png` - 剪枝后结果
- `06_cart_unpruned_boundary.png` - 剪枝前决策边界
- `06_cart_pruned_boundary.png` - 剪枝后决策边界

#### 📄 报告文件
- `summary_report.txt` - 详细的实验报告和算法说明

## 📊 数据集要求

1. **特征矩阵 X**: 形状为 `(n_samples, n_features)` 的数组
2. **标签向量 y**: 
   - 分类：离散标签值
   - 回归：连续目标值
3. **特征类型**: 支持连续和离散特征
4. **树结构**: 生成二叉决策树
5. **双模式**: 支持分类和回归两种模式

## 🔧 基本用法

### 分类模式
```python
from decision_tree_cart import CART
import numpy as np

# 创建分类数据
X = np.array([[1.2, 2.3], [2.1, 3.4], [3.5, 1.8], [4.2, 2.9]])
y = np.array([0, 1, 1, 0])

# 训练分类模型
cart_clf = CART(mode='classification')
cart_clf.fit(X, y)

# 预测
predictions = cart_clf.predict(X)
print(predictions)

# 剪枝
cart_clf.prune_ccp(X, y)
```

### 回归模式
```python
# 创建回归数据
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
y = np.array([1.5, 2.8, 2.1, 3.2])

# 训练回归模型
cart_reg = CART(mode='regression')
cart_reg.fit(X, y)

# 预测
predictions = cart_reg.predict(X)
print(predictions)
```

## 🎯 推荐数据集

### 1. 分类数据集
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

### 2. 回归数据集
```python
from sklearn.datasets import make_regression, load_diabetes, fetch_california_housing

# 生成回归数据
X, y = make_regression(n_samples=200, n_features=2, noise=0.1)

# 加州房价数据集（推荐替代Boston数据集）
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 糖尿病数据集
data = load_diabetes()
X, y = data.data, data.target
```

## ⚙️ 算法特点

- ✅ **双模式**: 同时支持分类和回归
- ✅ **二叉树**: 生成二进制分裂的决策树
- ✅ **分裂准则**: 基尼系数（分类）或方差（回归）
- ✅ **连续特征**: 自动处理连续和离散特征
- ✅ **CCP剪枝**: 内置成本复杂度剪枝
- ✅ **高效**: 二叉树结构更简洁高效

## 🔍 输出解释

### 分类模式
- **准确率**: 分类的正确率
- **树的节点数**: 决策树的总节点数
- **树的深度**: 决策树的最大深度

### 回归模式
- **MSE**: 均方误差
- **R²分数**: 决定系数，衡量模型拟合程度
- **树的结构**: 节点数和深度信息

## 📈 性能分析

### 分类性能
1. **简单数据集**: 准确率通常在80-95%之间
2. **真实数据集**: 在标准数据集上表现稳定
3. **剪枝效果**: CCP剪枝有效提高泛化能力

### 回归性能
1. **MSE**: 均方误差越小越好
2. **R²分数**: 接近1表示拟合效果好
3. **过拟合**: 剪枝可以减少过拟合现象

## 📞 注意事项

1. 选择正确的模式（'classification' 或 'regression'）
2. 分类标签建议从0开始的连续整数
3. 回归目标值应为连续数值
4. CCP剪枝需要提供训练数据
5. 所有输出文件保存在 `./output/` 文件夹中

## 🔄 与其他算法对比

| 特性 | ID3 | C4.5 | CART |
|------|-----|------|------|
| 分裂准则 | 信息增益 | 信息增益比 | 基尼系数/方差 |
| 特征类型 | 仅离散 | 连续+离散 | 连续+离散 |
| 树结构 | 多叉 | 多叉 | 二叉 |
| 剪枝 | 无 | PEP剪枝 | CCP剪枝 |
| 应用 | 分类 | 分类 | 分类+回归 |
| 优势 | 简单快速 | 处理连续特征 | 双模式，二叉树 |

## 📚 理论背景

### 基尼系数
对于分类问题，CART使用基尼系数衡量不纯度：
```
Gini(D) = 1 - Σ(pi²)
```
其中pi是类别i在数据集D中的比例。

### 方差
对于回归问题，CART使用方差作为分裂准则：
```
Variance(D) = (1/n) * Σ(yi - ȳ)²
```
其中yi是目标值，ȳ是目标值的均值。 