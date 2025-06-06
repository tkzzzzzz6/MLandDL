import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from decision_tree_c45 import C45
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_folder():
    """创建输出文件夹"""
    folder_name = "./output"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name

def create_simple_dataset():
    """创建一个简单的分类数据集"""
    np.random.seed(42)
    n_samples = 100
    
    # 生成两个特征
    X = np.random.randn(n_samples, 2)
    
    # 创建标签：基于简单规则
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > 0) & (X[:, 1] > 0)] = 0  # 第一象限
    y[(X[:, 0] <= 0) & (X[:, 1] > 0)] = 1  # 第二象限
    y[(X[:, 0] <= 0) & (X[:, 1] <= 0)] = 2  # 第三象限
    y[(X[:, 0] > 0) & (X[:, 1] <= 0)] = 1  # 第四象限
    
    return X, y

def create_mixed_dataset():
    """创建混合数据集（连续和离散特征）"""
    np.random.seed(42)
    n_samples = 150
    
    # 连续特征
    continuous_features = np.random.randn(n_samples, 2)
    
    # 离散特征
    discrete_feature1 = np.random.choice(['A', 'B', 'C'], n_samples)
    discrete_feature2 = np.random.choice([0, 1, 2], n_samples)
    
    # 合并特征
    X = np.column_stack([
        continuous_features,
        discrete_feature1,
        discrete_feature2
    ])
    
    # 创建标签
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if continuous_features[i, 0] > 0 and discrete_feature1[i] == 'A':
            y[i] = 0
        elif continuous_features[i, 1] > 0 and discrete_feature2[i] == 1:
            y[i] = 1
        else:
            y[i] = 2
    
    return X, y

def use_sklearn_dataset():
    """使用sklearn内置数据集"""
    print("=== 使用鸢尾花数据集 ===")
    
    # 加载鸢尾花数据集
    data = load_iris()
    X, y = data.data, data.target
    
    # 只使用前两个特征便于可视化
    X = X[:, :2]
    
    return X, y

def use_wine_dataset():
    """使用葡萄酒数据集"""
    print("=== 使用葡萄酒数据集 ===")
    
    # 加载葡萄酒数据集
    data = load_wine()
    X, y = data.data, data.target
    
    # 只使用前两个特征便于可视化
    X = X[:, :2]
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def visualize_results(X, y, y_pred, title, output_folder, filename):
    """可视化结果并保存到文件"""
    plt.figure(figsize=(15, 5))
    
    # 创建颜色映射
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(y))))
    
    # 原始数据
    plt.subplot(1, 3, 1)
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.7, s=50, 
                   label=f'类别 {class_label}')
    plt.title(f'{title} - 真实标签', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 预测结果
    plt.subplot(1, 3, 2)
    for i, class_label in enumerate(np.unique(y_pred)):
        mask = y_pred == class_label
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.7, s=50,
                   label=f'类别 {class_label}')
    plt.title(f'{title} - 预测结果', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率图
    plt.subplot(1, 3, 3)
    correct = (y == y_pred)
    plt.scatter(X[correct, 0], X[correct, 1], c='green', alpha=0.7, s=50, label='正确预测')
    plt.scatter(X[~correct, 0], X[~correct, 1], c='red', alpha=0.7, s=50, label='错误预测')
    plt.title(f'{title} - 预测准确性', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")

def visualize_decision_boundary(X, y, model, title, output_folder, filename):
    """可视化决策边界"""
    plt.figure(figsize=(12, 10))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set1)
    except:
        print("决策边界绘制失败，可能是由于模型预测错误")
    
    # 绘制数据点
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(y))))
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.8, s=50, 
                   edgecolors='black', label=f'类别 {class_label}')
    
    plt.title(f'{title} - 决策边界', fontsize=16)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_folder, f"{filename}_boundary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"决策边界图已保存: {save_path}")

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def print_tree_info(model, title):
    """打印决策树信息"""
    try:
        tree_size = model._C45__tree.size()
        tree_depth = model._C45__tree.depth()
        print(f"{title}:")
        print(f"  树的节点数: {tree_size}")
        print(f"  树的深度: {tree_depth}")
    except:
        print(f"{title}: 无法获取树的结构信息")

def main():
    print("C4.5决策树 (Decision Tree C4.5) 使用示例")
    print("=" * 60)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 示例1：简单数据集
    print("\n1. 简单自定义数据集（多分类）")
    X1, y1 = create_simple_dataset()
    
    # 训练C4.5决策树
    c45_1 = C45()
    c45_1.fit(X1, y1)
    
    # 预测
    y_pred1 = c45_1.predict(X1)
    accuracy1 = calculate_accuracy(y1, y_pred1)
    
    print(f"准确率: {accuracy1:.3f}")
    print_tree_info(c45_1, "决策树信息")
    
    # 可视化并保存
    visualize_results(X1, y1, y_pred1, "简单数据集", output_folder, "01_simple_dataset")
    visualize_decision_boundary(X1, y1, c45_1, "简单数据集", output_folder, "01_simple_dataset")
    
    # 示例2：sklearn生成的数据集
    print("\n2. Sklearn生成的数据集")
    X2, y2 = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.3, random_state=42
    )
    
    # 训练
    c45_2 = C45()
    c45_2.fit(X_train, y_train)
    
    # 预测
    y_train_pred = c45_2.predict(X_train)
    y_test_pred = c45_2.predict(X_test)
    
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"训练准确率: {train_accuracy:.3f}")
    print(f"测试准确率: {test_accuracy:.3f}")
    print_tree_info(c45_2, "决策树信息")
    
    # 可视化并保存
    visualize_results(X_train, y_train, y_train_pred, "训练集", output_folder, "02_sklearn_train")
    visualize_results(X_test, y_test, y_test_pred, "测试集", output_folder, "02_sklearn_test")
    visualize_decision_boundary(X_train, y_train, c45_2, "Sklearn数据集", output_folder, "02_sklearn")
    
    # 示例3：鸢尾花数据集
    print("\n3. 鸢尾花数据集")
    X3, y3 = use_sklearn_dataset()
    
    # 分割数据集
    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X3, y3, test_size=0.3, random_state=42
    )
    
    # 训练
    c45_3 = C45()
    c45_3.fit(X_train3, y_train3)
    
    # 预测
    y_train_pred3 = c45_3.predict(X_train3)
    y_test_pred3 = c45_3.predict(X_test3)
    
    train_accuracy3 = calculate_accuracy(y_train3, y_train_pred3)
    test_accuracy3 = calculate_accuracy(y_test3, y_test_pred3)
    
    print(f"训练准确率: {train_accuracy3:.3f}")
    print(f"测试准确率: {test_accuracy3:.3f}")
    print_tree_info(c45_3, "决策树信息")
    
    # 可视化并保存
    visualize_results(X_train3, y_train3, y_train_pred3, "鸢尾花数据集-训练", output_folder, "03_iris_train")
    visualize_results(X_test3, y_test3, y_test_pred3, "鸢尾花数据集-测试", output_folder, "03_iris_test")
    visualize_decision_boundary(X_train3, y_train3, c45_3, "鸢尾花数据集", output_folder, "03_iris")
    
    # 示例4：葡萄酒数据集
    print("\n4. 葡萄酒数据集")
    X4, y4 = use_wine_dataset()
    
    # 分割数据集
    X_train4, X_test4, y_train4, y_test4 = train_test_split(
        X4, y4, test_size=0.3, random_state=42
    )
    
    # 训练
    c45_4 = C45()
    c45_4.fit(X_train4, y_train4)
    
    # 预测
    y_train_pred4 = c45_4.predict(X_train4)
    y_test_pred4 = c45_4.predict(X_test4)
    
    train_accuracy4 = calculate_accuracy(y_train4, y_train_pred4)
    test_accuracy4 = calculate_accuracy(y_test4, y_test_pred4)
    
    print(f"训练准确率: {train_accuracy4:.3f}")
    print(f"测试准确率: {test_accuracy4:.3f}")
    print_tree_info(c45_4, "决策树信息")
    
    # 可视化并保存
    visualize_results(X_train4, y_train4, y_train_pred4, "葡萄酒数据集-训练", output_folder, "04_wine_train")
    visualize_results(X_test4, y_test4, y_test_pred4, "葡萄酒数据集-测试", output_folder, "04_wine_test")
    visualize_decision_boundary(X_train4, y_train4, c45_4, "葡萄酒数据集", output_folder, "04_wine")

def demonstrate_pruning():
    """演示剪枝效果"""
    print("\n5. 剪枝效果演示")
    
    output_folder = "./output"
    
    # 创建数据集
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练未剪枝的树
    c45_unpruned = C45()
    c45_unpruned.fit(X_train, y_train)
    y_pred_unpruned = c45_unpruned.predict(X_test)
    accuracy_unpruned = calculate_accuracy(y_test, y_pred_unpruned)
    
    print("剪枝前:")
    print(f"  测试准确率: {accuracy_unpruned:.3f}")
    print_tree_info(c45_unpruned, "  树信息")
    
    # 剪枝
    c45_pruned = C45()
    c45_pruned.fit(X_train, y_train)
    c45_pruned.prune_pep()
    y_pred_pruned = c45_pruned.predict(X_test)
    accuracy_pruned = calculate_accuracy(y_test, y_pred_pruned)
    
    print("剪枝后:")
    print(f"  测试准确率: {accuracy_pruned:.3f}")
    print_tree_info(c45_pruned, "  树信息")
    
    # 可视化对比
    visualize_results(X_test, y_test, y_pred_unpruned, "剪枝前", output_folder, "05_unpruned")
    visualize_results(X_test, y_test, y_pred_pruned, "剪枝后", output_folder, "05_pruned")
    visualize_decision_boundary(X_test, y_test, c45_unpruned, "剪枝前", output_folder, "05_unpruned")
    visualize_decision_boundary(X_test, y_test, c45_pruned, "剪枝后", output_folder, "05_pruned")

def save_summary_report(output_folder):
    """保存总结报告"""
    report_path = os.path.join(output_folder, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("C4.5决策树 (Decision Tree C4.5) 实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## 数据集要求:\n")
        f.write("1. X: 特征矩阵，形状为 (n_samples, n_features)\n")
        f.write("2. y: 标签向量，支持多分类\n")
        f.write("3. 支持连续和离散特征的混合\n")
        f.write("4. 自动处理特征类型\n")
        f.write("5. 适用于多分类问题\n\n")
        
        f.write("## 算法特点:\n")
        f.write("1. 使用信息增益比作为分裂准则\n")
        f.write("2. 支持连续和离散特征\n")
        f.write("3. 内置剪枝功能（PEP剪枝）\n")
        f.write("4. 处理过拟合问题\n")
        f.write("5. 相比ID3算法更稳定\n\n")
        
        f.write("## 推荐数据集:\n")
        f.write("1. sklearn.datasets.make_classification() - 生成分类数据\n")
        f.write("2. sklearn.datasets.load_iris() - 鸢尾花数据集\n")
        f.write("3. sklearn.datasets.load_wine() - 葡萄酒数据集\n")
        f.write("4. sklearn.datasets.load_breast_cancer() - 乳腺癌数据集\n")
        f.write("5. 任何多分类数据集\n\n")
        
        f.write("## 生成的图片说明:\n")
        f.write("- 散点图: 显示数据分布和分类结果\n")
        f.write("- 决策边界图: 显示C4.5决策树的分类边界\n")
        f.write("- 准确性图: 显示预测正确和错误的点\n")
        f.write("- 剪枝对比图: 比较剪枝前后的效果\n\n")
        
        f.write("## 与其他决策树算法的比较:\n")
        f.write("- ID3: 只支持离散特征，使用信息增益\n")
        f.write("- C4.5: 支持连续特征，使用信息增益比，有剪枝\n")
        f.write("- CART: 二叉树，支持回归，使用基尼系数\n")
    
    print(f"总结报告已保存: {report_path}")

if __name__ == "__main__":
    main()
    demonstrate_pruning()
    
    output_folder = "./output"
    save_summary_report(output_folder)
    
    print("\n" + "=" * 60)
    print("所有图片和报告已保存到文件夹:", output_folder)
    print("=" * 60) 