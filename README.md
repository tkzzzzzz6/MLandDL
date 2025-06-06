# MLandDL - 机器学习与深度学习项目

这是一个专注于机器学习和深度学习算法实现的项目，目前主要包含决策树算法的多种实现。

## 📁 项目结构

```
MLandDL/
├── decision_tree/           # 决策树模块
│   ├── src/                # 源代码
│   │   ├── decision_tree_id3.py    # ID3算法实现
│   │   ├── decision_tree_c45.py    # C4.5算法实现
│   │   └── decision_tree_cart.py   # CART算法实现
│   └── demo/               # 示例代码
│       ├── decision_tree_id3_example/
│       ├── decision_tree_c45_example/
│       ├── decision_tree_cart_example/
│       └── decision_stump_example/
├── License                 # MIT协议
└── README.md              # 项目说明
```

## 🚀 功能特性

### 决策树算法

- **ID3算法**: 基于信息增益的决策树构建算法
- **C4.5算法**: 改进的ID3算法，使用信息增益率并支持连续属性
- **CART算法**: 分类与回归树，支持二元分割和剪枝
- **决策桩**: 最简单的决策树形式，只有一个分割节点

## 🛠️ 安装和使用

### 环境要求

- Python 3.6+
- NumPy
- Pandas（用于数据处理）
- Matplotlib（用于可视化）

### 安装依赖

```bash
pip install numpy pandas matplotlib
```

### 快速开始

1. 克隆项目到本地：
```bash
git clone <repository-url>
cd MLandDL
```

2. 运行示例代码：
```bash
cd decision_tree/demo/decision_tree_id3_example
python example.py
```

## 📚 算法说明

### ID3算法
ID3（Iterative Dichotomiser 3）是最经典的决策树算法之一，使用信息增益作为特征选择的标准。

### C4.5算法
C4.5是ID3的改进版本，解决了ID3算法的一些不足：
- 使用信息增益率代替信息增益
- 支持连续属性的处理
- 支持缺失值的处理

### CART算法
CART（Classification and Regression Trees）算法特点：
- 支持分类和回归问题
- 使用基尼系数或均方误差作为分割标准
- 支持后剪枝操作

## 🤝 贡献指南

欢迎提交问题和拉取请求。对于重大更改，请先开启一个问题来讨论您想要改变的内容。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](License) 文件了解详情。

## 📧 联系方式

如果您有任何问题或建议，请通过以下方式联系我们：
- 提交 Issue
- 发送邮件

## 🔮 未来计划

- [ ] 添加更多机器学习算法实现
- [ ] 增加深度学习模块
- [ ] 提供更多数据集示例
- [ ] 添加算法性能对比
- [ ] 提供Web界面展示

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！ 