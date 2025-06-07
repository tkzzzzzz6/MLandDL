import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decision_tree_cart import CART
import sys
import os
import graphviz

# 添加metrics模块路径
sys.path.append('../../src')

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

def create_simple_classification_dataset():
    """创建一个简单的分类数据集"""
    np.random.seed(42)
    n_samples = 200
    
    # 生成两个特征
    X = np.random.randn(n_samples, 2)
    
    # 创建标签：基于复杂规则
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > 0) & (X[:, 1] > 0)] = 0  # 第一象限
    y[(X[:, 0] <= 0) & (X[:, 1] > 0)] = 1  # 第二象限
    y[(X[:, 0] <= 0) & (X[:, 1] <= 0)] = 2  # 第三象限
    y[(X[:, 0] > 0) & (X[:, 1] <= 0)] = 1  # 第四象限
    
    return X, y

def create_simple_regression_dataset():
    """创建一个简单的回归数据集"""
    np.random.seed(42)
    n_samples = 100
    
    # 生成特征
    X = np.random.uniform(-3, 3, (n_samples, 2))
    
    # 创建目标变量：非线性关系
    y = X[:, 0]**2 + X[:, 1]**2 + 0.5 * np.random.randn(n_samples)
    
    return X, y

def use_sklearn_classification_dataset():
    """使用sklearn分类数据集"""
    print("=== 使用鸢尾花数据集 ===")
    
    # 加载鸢尾花数据集
    data = load_iris()
    X, y = data.data, data.target
    
    # 只使用前两个特征便于可视化
    X = X[:, :2]
    
    return X, y

def use_sklearn_regression_dataset():
    """使用sklearn回归数据集"""
    print("=== 使用生成的回归数据集 ===")
    
    # 生成回归数据集
    X, y = make_regression(
        n_samples=200,
        n_features=2,
        noise=0.1,
        random_state=42
    )
    
    return X, y

def visualize_classification_results(X, y, y_pred, title, output_folder, filename):
    """可视化分类结果并保存到文件"""
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

def visualize_regression_results(X, y, y_pred, title, output_folder, filename):
    """可视化回归结果并保存到文件"""
    plt.figure(figsize=(15, 5))
    
    # 真实值 vs 特征1
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], y, c=y, alpha=0.7, s=50, cmap='viridis')
    plt.title(f'{title} - 真实值 vs 特征1', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('目标变量', fontsize=12)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    # 预测值 vs 特征1
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], y_pred, c=y_pred, alpha=0.7, s=50, cmap='viridis')
    plt.title(f'{title} - 预测值 vs 特征1', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    # 真实值 vs 预测值
    plt.subplot(1, 3, 3)
    plt.scatter(y, y_pred, alpha=0.7, s=50)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title(f'{title} - 真实值 vs 预测值', fontsize=14)
    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
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
    except Exception as e:
        print(f"决策边界绘制失败: {e}")
    
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

def calculate_mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def calculate_r2(y_true, y_pred):
    """计算R²分数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def visualize_decision_tree(model, feature_names, class_names, title, output_folder, filename, is_regression=False):
    """使用graphviz可视化决策树"""
    try:
        # 创建graphviz图形对象
        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
        dot.attr('edge', fontname='SimHei')
        
        # 获取决策树
        tree = model._CART__tree
        
        def add_nodes_edges(node_id):
            """递归添加节点和边"""
            node = tree.get_node(node_id)
            
            if node.is_leaf():
                # 叶子节点 - 显示类别或回归值
                if is_regression:
                    label = f"预测值: {node.data:.2f}"
                else:
                    if isinstance(node.data, str):
                        label = f"类别: {node.data}"
                    else:
                        label = f"类别: {class_names[node.data] if isinstance(node.data, int) and node.data < len(class_names) else node.data}"
                dot.node(str(node_id), label, fillcolor='lightgreen')
            else:
                # 内部节点 - 显示特征和阈值
                if isinstance(node.data, dict):
                    feature_idx = node.data.get('feature', 0)
                    threshold = node.data.get('threshold', 0)
                    if feature_idx < len(feature_names):
                        label = f"特征: {feature_names[feature_idx]}\\n<= {threshold:.2f}"
                    else:
                        label = f"特征 {feature_idx}\\n<= {threshold:.2f}"
                elif isinstance(node.data, int) and node.data < len(feature_names):
                    label = f"特征: {feature_names[node.data]}"
                else:
                    label = f"特征 {node.data}"
                dot.node(str(node_id), label, fillcolor='lightblue')
                
                # 添加子节点
                for child in tree.children(node_id):
                    add_nodes_edges(child.identifier)
                    # 添加边，标注分支条件
                    edge_label = str(child.tag)
                    dot.edge(str(node_id), str(child.identifier), label=edge_label)
        
        # 从根节点开始构建图形
        if tree.root:
            add_nodes_edges(tree.root)
        
        # 保存图形
        output_path = os.path.join(output_folder, filename)
        dot.render(output_path, format='png', cleanup=True)
        print(f"决策树可视化已保存: {output_path}.png")
        
        return dot
        
    except Exception as e:
        print(f"决策树可视化失败: {str(e)}")
        return None

def create_feature_class_names(X_sample, y_sample, dataset_name, is_regression=False):
    """为不同数据集创建特征名和类别名"""
    if dataset_name == "simple_classification":
        feature_names = ['特征1', '特征2']
        class_names = ['类别0', '类别1', '类别2']
    elif dataset_name == "simple_regression":
        feature_names = ['特征1', '特征2']
        class_names = []  # 回归任务不需要类别名
    elif dataset_name == "iris":
        feature_names = ['花萼长度', '花萼宽度']
        class_names = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    elif dataset_name == "sklearn_regression":
        feature_names = ['特征1', '特征2']
        class_names = []  # 回归任务不需要类别名
    else:
        n_features = X_sample.shape[1] if len(X_sample.shape) > 1 else 1
        feature_names = [f'特征{i}' for i in range(n_features)]
        if is_regression:
            class_names = []
        else:
            class_names = [f'类别{i}' for i in np.unique(y_sample)]
    
    return feature_names, class_names

def print_tree_info(model, title):
    """打印决策树信息"""
    try:
        tree_size = model._CART__tree.size()
        tree_depth = model._CART__tree.depth()
        print(f"{title}:")
        print(f"  树的节点数: {tree_size}")
        print(f"  树的深度: {tree_depth}")
    except:
        print(f"{title}: 无法获取树的结构信息")

def main_classification():
    print("CART决策树 - 分类模式 (Decision Tree CART - Classification)")
    print("=" * 70)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 示例1：简单分类数据集
    print("\n1. 简单自定义分类数据集")
    X1, y1 = create_simple_classification_dataset()
    
    # 训练CART分类树
    cart_clf_1 = CART(mode='classification')
    cart_clf_1.fit(X1, y1)
    
    # 预测
    y_pred1 = cart_clf_1.predict(X1)
    accuracy1 = calculate_accuracy(y1, y_pred1)
    
    print(f"准确率: {accuracy1:.3f}")
    print_tree_info(cart_clf_1, "决策树信息")
    
    # 决策树可视化
    feature_names1, class_names1 = create_feature_class_names(X1, y1, "simple_classification")
    visualize_decision_tree(cart_clf_1, feature_names1, class_names1, "简单分类数据集", output_folder, "01_classification_simple_tree")
    
    # 可视化并保存
    visualize_classification_results(X1, y1, y_pred1, "简单分类数据集", output_folder, "01_classification_simple")
    visualize_decision_boundary(X1, y1, cart_clf_1, "简单分类数据集", output_folder, "01_classification_simple")
    
    # 示例2：sklearn生成的分类数据集
    print("\n2. Sklearn生成的分类数据集")
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
    cart_clf_2 = CART(mode='classification')
    cart_clf_2.fit(X_train, y_train)
    
    # 预测
    y_train_pred = cart_clf_2.predict(X_train)
    y_test_pred = cart_clf_2.predict(X_test)
    
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"训练准确率: {train_accuracy:.3f}")
    print(f"测试准确率: {test_accuracy:.3f}")
    print_tree_info(cart_clf_2, "决策树信息")
    
    # 决策树可视化
    feature_names2, class_names2 = create_feature_class_names(X_train, y_train, "sklearn_classification")
    visualize_decision_tree(cart_clf_2, feature_names2, class_names2, "Sklearn分类数据集", output_folder, "02_classification_tree")
    
    # 可视化并保存
    visualize_classification_results(X_train, y_train, y_train_pred, "分类训练集", output_folder, "02_classification_train")
    visualize_classification_results(X_test, y_test, y_test_pred, "分类测试集", output_folder, "02_classification_test")
    visualize_decision_boundary(X_train, y_train, cart_clf_2, "Sklearn分类数据集", output_folder, "02_classification")
    
    # 示例3：鸢尾花数据集
    print("\n3. 鸢尾花数据集")
    X3, y3 = use_sklearn_classification_dataset()
    
    # 分割数据集
    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X3, y3, test_size=0.3, random_state=42
    )
    
    # 训练
    cart_clf_3 = CART(mode='classification')
    cart_clf_3.fit(X_train3, y_train3)
    
    # 预测
    y_train_pred3 = cart_clf_3.predict(X_train3)
    y_test_pred3 = cart_clf_3.predict(X_test3)
    
    train_accuracy3 = calculate_accuracy(y_train3, y_train_pred3)
    test_accuracy3 = calculate_accuracy(y_test3, y_test_pred3)
    
    print(f"训练准确率: {train_accuracy3:.3f}")
    print(f"测试准确率: {test_accuracy3:.3f}")
    print_tree_info(cart_clf_3, "决策树信息")
    
    # 决策树可视化
    feature_names3, class_names3 = create_feature_class_names(X_train3, y_train3, "iris")
    visualize_decision_tree(cart_clf_3, feature_names3, class_names3, "鸢尾花数据集", output_folder, "03_iris_tree")
    
    # 可视化并保存
    visualize_classification_results(X_train3, y_train3, y_train_pred3, "鸢尾花-训练", output_folder, "03_iris_train")
    visualize_classification_results(X_test3, y_test3, y_test_pred3, "鸢尾花-测试", output_folder, "03_iris_test")
    visualize_decision_boundary(X_train3, y_train3, cart_clf_3, "鸢尾花数据集", output_folder, "03_iris")

def main_regression():
    print("\nCART决策树 - 回归模式 (Decision Tree CART - Regression)")
    print("=" * 70)
    
    output_folder = "./output"
    
    # 示例1：简单回归数据集
    print("\n4. 简单自定义回归数据集")
    X1, y1 = create_simple_regression_dataset()
    
    # 训练CART回归树
    cart_reg_1 = CART(mode='regression')
    cart_reg_1.fit(X1, y1)
    
    # 预测
    y_pred1 = cart_reg_1.predict(X1)
    mse1 = calculate_mse(y1, y_pred1)
    r2_1 = calculate_r2(y1, y_pred1)
    
    print(f"均方误差 (MSE): {mse1:.3f}")
    print(f"R² 分数: {r2_1:.3f}")
    print_tree_info(cart_reg_1, "决策树信息")
    
    # 决策树可视化
    feature_names4, class_names4 = create_feature_class_names(X1, y1, "simple_regression", is_regression=True)
    visualize_decision_tree(cart_reg_1, feature_names4, class_names4, "简单回归数据集", output_folder, "04_regression_simple_tree", is_regression=True)
    
    # 可视化并保存
    visualize_regression_results(X1, y1, y_pred1, "简单回归数据集", output_folder, "04_regression_simple")
    
    # 示例2：sklearn生成的回归数据集
    print("\n5. Sklearn生成的回归数据集")
    X2, y2 = use_sklearn_regression_dataset()
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.3, random_state=42
    )
    
    # 训练
    cart_reg_2 = CART(mode='regression')
    cart_reg_2.fit(X_train, y_train)
    
    # 预测
    y_train_pred = cart_reg_2.predict(X_train)
    y_test_pred = cart_reg_2.predict(X_test)
    
    train_mse = calculate_mse(y_train, y_train_pred)
    test_mse = calculate_mse(y_test, y_test_pred)
    train_r2 = calculate_r2(y_train, y_train_pred)
    test_r2 = calculate_r2(y_test, y_test_pred)
    
    print(f"训练 MSE: {train_mse:.3f}, R²: {train_r2:.3f}")
    print(f"测试 MSE: {test_mse:.3f}, R²: {test_r2:.3f}")
    print_tree_info(cart_reg_2, "决策树信息")
    
    # 决策树可视化
    feature_names5, class_names5 = create_feature_class_names(X_train, y_train, "sklearn_regression", is_regression=True)
    visualize_decision_tree(cart_reg_2, feature_names5, class_names5, "Sklearn回归数据集", output_folder, "05_regression_tree", is_regression=True)
    
    # 可视化并保存
    visualize_regression_results(X_train, y_train, y_train_pred, "回归训练集", output_folder, "05_regression_train")
    visualize_regression_results(X_test, y_test, y_test_pred, "回归测试集", output_folder, "05_regression_test")

def demonstrate_pruning():
    """演示剪枝效果"""
    print("\n6. CART剪枝效果演示")
    
    output_folder = "./output"
    
    # 创建分类数据集
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
    cart_unpruned = CART(mode='classification')
    cart_unpruned.fit(X_train, y_train)
    y_pred_unpruned = cart_unpruned.predict(X_test)
    accuracy_unpruned = calculate_accuracy(y_test, y_pred_unpruned)
    
    print("剪枝前:")
    print(f"  测试准确率: {accuracy_unpruned:.3f}")
    print_tree_info(cart_unpruned, "  树信息")
    
    # 剪枝
    cart_pruned = CART(mode='classification')
    cart_pruned.fit(X_train, y_train)
    cart_pruned.prune_ccp(X_train, y_train)
    y_pred_pruned = cart_pruned.predict(X_test)
    accuracy_pruned = calculate_accuracy(y_test, y_pred_pruned)
    
    print("剪枝后:")
    print(f"  测试准确率: {accuracy_pruned:.3f}")
    print_tree_info(cart_pruned, "  树信息")
    
    # 可视化对比
    visualize_classification_results(X_test, y_test, y_pred_unpruned, "CART剪枝前", output_folder, "06_cart_unpruned")
    visualize_classification_results(X_test, y_test, y_pred_pruned, "CART剪枝后", output_folder, "06_cart_pruned")
    visualize_decision_boundary(X_test, y_test, cart_unpruned, "CART剪枝前", output_folder, "06_cart_unpruned")
    visualize_decision_boundary(X_test, y_test, cart_pruned, "CART剪枝后", output_folder, "06_cart_pruned")

def save_summary_report(output_folder):
    """保存总结报告"""
    report_path = os.path.join(output_folder, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CART决策树 (Decision Tree CART) 实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## 数据集要求:\n")
        f.write("1. X: 特征矩阵，形状为 (n_samples, n_features)\n")
        f.write("2. y: 标签向量（分类）或目标变量（回归）\n")
        f.write("3. 支持连续和离散特征\n")
        f.write("4. 支持分类和回归两种模式\n")
        f.write("5. 二叉树结构\n\n")
        
        f.write("## 算法特点:\n")
        f.write("1. 使用基尼系数（分类）或方差（回归）作为分裂准则\n")
        f.write("2. 生成二叉决策树\n")
        f.write("3. 支持分类和回归两种模式\n")
        f.write("4. 内置CCP剪枝功能\n")
        f.write("5. 处理连续和离散特征\n\n")
        
        f.write("## 推荐数据集:\n")
        f.write("### 分类:\n")
        f.write("1. sklearn.datasets.make_classification() - 生成分类数据\n")
        f.write("2. sklearn.datasets.load_iris() - 鸢尾花数据集\n")
        f.write("3. sklearn.datasets.load_wine() - 葡萄酒数据集\n")
        f.write("4. sklearn.datasets.load_breast_cancer() - 乳腺癌数据集\n\n")
        f.write("### 回归:\n")
        f.write("1. sklearn.datasets.make_regression() - 生成回归数据\n")
        f.write("2. sklearn.datasets.fetch_california_housing() - 加州房价数据集\n")
        f.write("3. sklearn.datasets.load_diabetes() - 糖尿病数据集\n\n")
        
        f.write("## 生成的图片说明:\n")
        f.write("### 分类模式:\n")
        f.write("- 散点图: 显示数据分布和分类结果\n")
        f.write("- 决策边界图: 显示CART决策树的分类边界\n")
        f.write("- 准确性图: 显示预测正确和错误的点\n\n")
        f.write("### 回归模式:\n")
        f.write("- 特征-目标图: 显示特征与目标变量的关系\n")
        f.write("- 真实值vs预测值图: 评估回归效果\n")
        f.write("- 剪枝对比图: 比较剪枝前后的效果\n\n")
        
        f.write("## 与其他决策树算法的比较:\n")
        f.write("- ID3: 只支持离散特征，使用信息增益，多叉树\n")
        f.write("- C4.5: 支持连续特征，使用信息增益比，多叉树\n")
        f.write("- CART: 二叉树，支持回归，使用基尼系数/方差\n")
    
    print(f"总结报告已保存: {report_path}")

if __name__ == "__main__":
    main_classification()
    main_regression()
    demonstrate_pruning()
    
    output_folder = "./output"
    save_summary_report(output_folder)
    
    print("\n" + "=" * 70)
    print("所有图片和报告已保存到文件夹:", output_folder)
    print("=" * 70) 