# ID3决策树 (Decision Tree ID3) 使用指南

## 📖 简介

ID3（Iterative Dichotomiser 3）是最早的决策树算法之一，使用信息增益作为分裂准则。**注意：ID3算法只支持离散特征，连续特征需要先进行离散化处理。**

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

#### 📊 可视化图片 (2张)
- `01_iris_train.png` - 鸢尾花训练集结果（离散化后）
- `01_iris_test.png` - 鸢尾花测试集结果（离散化后）

#### 📄 报告文件
- `summary_report.txt` - 详细的实验报告和算法说明

## 📊 数据集要求

1. **特征矩阵 X**: 形状为 `(n_samples, n_features)` 的数组
2. **标签向量 y**: 支持多分类
3. **⚠️ 重要限制**: **只支持离散特征！**
4. **连续特征处理**: 需要先离散化处理
5. **问题类型**: 适用于多分类问题

## 🔧 基本用法

### 离散特征数据
```python
from decision_tree_id3 import ID3
import numpy as np

# 创建离散特征数据
X = np.array([
    ['sunny', 'hot', 'high', 'false'],
    ['sunny', 'hot', 'high', 'true'],
    ['overcast', 'hot', 'high', 'false'],
    ['rainy', 'mild', 'high', 'false']
])
y = np.array(['no', 'no', 'yes', 'yes'])

# 训练模型
id3 = ID3()
id3.fit(X, y)

# 预测
predictions = id3.predict(X)
print(predictions)
```

### 连续特征离散化
```python
from sklearn.preprocessing import KBinsDiscretizer

# 连续特征数据
X_continuous = np.array([[1.2, 2.3], [2.1, 3.4], [3.5, 1.8], [4.2, 2.9]])

# 离散化处理
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X_continuous).astype(int)

# 现在可以使用ID3
id3 = ID3()
id3.fit(X_discrete, y)
```

## 🎯 推荐数据集

### 1. 天然离散数据集
```python
# 经典天气决策数据集
weather_data = {
    'outlook': ['sunny', 'overcast', 'rainy'],
    'temperature': ['hot', 'mild', 'cool'],
    'humidity': ['high', 'normal'],
    'windy': ['true', 'false'],
    'play': ['yes', 'no']
}
```

### 2. 离散化后的连续数据
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

# 加载鸢尾花数据集
data = load_iris()
X, y = data.data, data.target

# 离散化连续特征
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X).astype(int)
```

### 3. 其他推荐数据集
- **蘑菇分类数据集**: 天然的离散特征
- **动物分类数据集**: 布尔型特征
- **投票记录数据集**: 离散投票选项
- **汽车评估数据集**: 等级特征

## ⚙️ 算法特点

- ✅ **信息增益**: 使用信息增益选择最佳分裂特征
- ✅ **多叉树**: 生成多分支决策树
- ✅ **简单快速**: 算法实现简单，计算快速
- ✅ **多分类**: 支持多类别分类问题
- ⚠️ **仅离散特征**: 只能处理离散特征
- ⚠️ **无剪枝**: 没有剪枝功能，容易过拟合
- ⚠️ **偏向多值**: 偏向选择取值较多的特征

## 🔍 数据预处理

### 连续特征离散化方法

#### 1. 等宽分箱
```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X)
```

#### 2. 等频分箱
```python
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X)
```

#### 3. 自定义阈值
```python
import numpy as np

def custom_discretize(x, thresholds):
    """自定义阈值离散化"""
    return np.digitize(x, thresholds)

# 使用自定义阈值
thresholds = [2.0, 4.0, 6.0]
X_discrete = custom_discretize(X_continuous, thresholds)
```

## 🔍 输出解释

运行示例后，控制台会显示：
- **准确率**: 模型在数据上的分类准确率
- **树的节点数**: 决策树的总节点数
- **树的深度**: 决策树的最大深度
- **决策树结构**: 会打印出完整的决策树

## 📈 性能分析

1. **天然离散数据**: 在离散特征数据上表现优秀
2. **离散化影响**: 离散化策略对性能影响很大
3. **过拟合倾向**: 容易在训练数据上过拟合
4. **分箱数量**: 合适的分箱数量是关键

## 📞 注意事项

1. **⚠️ 只支持离散特征**: 这是最重要的限制
2. **离散化策略**: 选择合适的离散化方法很重要
3. **分箱数量**: 太少会损失信息，太多会过拟合
4. **字符串特征**: 算法直接支持字符串类型特征
5. **无剪枝**: 没有剪枝功能，需要注意过拟合
6. **所有输出文件保存在 `./output/` 文件夹中**

## 🔄 与其他算法对比

| 特性 | ID3 | C4.5 | CART |
|------|-----|------|------|
| 分裂准则 | 信息增益 | 信息增益比 | 基尼系数/方差 |
| 特征类型 | **仅离散** | 连续+离散 | 连续+离散 |
| 树结构 | 多叉 | 多叉 | 二叉 |
| 剪枝 | **无** | PEP剪枝 | CCP剪枝 |
| 应用 | 离散特征分类 | 通用分类 | 分类+回归 |
| 优势 | 简单快速 | 改进ID3缺陷 | 双模式支持 |
| 缺陷 | 只能离散特征 | 相对复杂 | 仅二叉分裂 |

## 📚 理论背景

### 信息增益
ID3使用信息增益作为特征选择准则：

```
Information_Gain(D, A) = Entropy(D) - Σ(|Dv|/|D|) * Entropy(Dv)
```

其中：
- `D` 是数据集
- `A` 是特征
- `Dv` 是特征A取值为v的子集

### 信息熵
```
Entropy(D) = -Σ(pi * log2(pi))
```

其中pi是类别i在数据集D中的比例。

## 🎯 使用建议

1. **优先考虑其他算法**: 除非特征天然离散，否则建议使用C4.5或CART
2. **教学用途**: ID3算法适合作为决策树算法的入门学习
3. **快速原型**: 对于简单的离散特征分类问题，ID3可以快速建模
4. **预处理重要**: 如果必须使用ID3处理连续特征，离散化预处理是关键 