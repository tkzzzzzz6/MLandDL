import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decision_stump import DecisionStump
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
    """创建一个简单的二分类数据集"""
    np.random.seed(42)
    n_samples = 100
    
    # 生成两个特征
    X = np.random.randn(n_samples, 2)
    
    # 创建标签：基于简单规则
    y = np.ones(n_samples)
    y[X[:, 0] + X[:, 1] < 0] = -1
    
    return X, y

def use_sklearn_dataset():
    """使用sklearn内置数据集"""
    print("=== 使用乳腺癌数据集 ===")
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 将标签转换为 -1 和 1
    y = np.where(y == 0, -1, 1)
    
    # 只使用前两个特征便于可视化
    X = X[:, :2]
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def use_make_classification():
    """使用sklearn生成分类数据集"""
    print("=== 使用生成的分类数据集 ===")
    
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 将标签转换为 -1 和 1
    y = np.where(y == 0, -1, 1)
    
    return X, y

def visualize_results(X, y, y_pred, title, output_folder, filename):
    """可视化结果并保存到文件"""
    plt.figure(figsize=(12, 5))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=50)
    plt.title(f'{title} - 真实标签', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 预测结果
    plt.subplot(1, 2, 2)
    colors_pred = ['red' if label == -1 else 'blue' for label in y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_pred, alpha=0.7, s=50)
    plt.title(f'{title} - 预测结果', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='类别 -1')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='类别 1')
    plt.legend(handles=[red_patch, blue_patch], loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"图片已保存: {save_path}")

def visualize_decision_boundary(X, y, stump, title, output_folder, filename):
    """可视化决策边界"""
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = stump.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.8, s=50, edgecolors='black')
    
    # 绘制决策阈值线
    feature_idx = stump._DecisionStump__feature_index
    threshold = stump._DecisionStump__threshold
    
    if feature_idx == 0:  # 在第一个特征上分割
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'阈值线: x1={threshold:.3f}')
    else:  # 在第二个特征上分割
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'阈值线: x2={threshold:.3f}')
    
    plt.title(f'{title} - 决策边界', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='类别 -1')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='类别 1')
    plt.legend(handles=[red_patch, blue_patch], loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_folder, f"{filename}_boundary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"决策边界图已保存: {save_path}")

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def main():
    print("决策树桩 (Decision Stump) 使用示例")
    print("=" * 50)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 示例1：简单数据集
    print("\n1. 简单自定义数据集")
    X1, y1 = create_simple_dataset()
    
    # 创建权重（通常在AdaBoost中使用，这里设为均匀权重）
    w1 = np.ones(len(y1)) / len(y1)
    
    # 训练决策树桩
    stump1 = DecisionStump()
    stump1.fit(X1, y1, w1)
    
    # 预测
    y_pred1 = stump1.predict(X1)
    accuracy1 = calculate_accuracy(y1, y_pred1)
    
    print(f"准确率: {accuracy1:.3f}")
    print(f"选择的特征: {stump1._DecisionStump__feature_index}")
    print(f"阈值: {stump1._DecisionStump__threshold:.3f}")
    print(f"方向: {stump1._DecisionStump__direction}")
    
    # 可视化并保存
    visualize_results(X1, y1, y_pred1, "简单数据集", output_folder, "01_simple_dataset")
    visualize_decision_boundary(X1, y1, stump1, "简单数据集", output_folder, "01_simple_dataset")
    
    # 示例2：sklearn生成的数据集
    print("\n2. Sklearn生成的数据集")
    X2, y2 = use_make_classification()
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.3, random_state=42
    )
    
    # 创建权重
    w_train = np.ones(len(y_train)) / len(y_train)
    
    # 训练
    stump2 = DecisionStump()
    stump2.fit(X_train, y_train, w_train)
    
    # 预测
    y_train_pred = stump2.predict(X_train)
    y_test_pred = stump2.predict(X_test)
    
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"训练准确率: {train_accuracy:.3f}")
    print(f"测试准确率: {test_accuracy:.3f}")
    print(f"选择的特征: {stump2._DecisionStump__feature_index}")
    print(f"阈值: {stump2._DecisionStump__threshold:.3f}")
    print(f"方向: {stump2._DecisionStump__direction}")
    
    # 可视化并保存
    visualize_results(X_train, y_train, y_train_pred, "训练集", output_folder, "02_sklearn_train")
    visualize_results(X_test, y_test, y_test_pred, "测试集", output_folder, "02_sklearn_test")
    visualize_decision_boundary(X_train, y_train, stump2, "Sklearn数据集", output_folder, "02_sklearn")
    
    # 示例3：真实数据集
    print("\n3. 乳腺癌数据集")
    X3, y3 = use_sklearn_dataset()
    
    # 分割数据集
    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X3, y3, test_size=0.3, random_state=42
    )
    
    # 创建权重
    w_train3 = np.ones(len(y_train3)) / len(y_train3)
    
    # 训练
    stump3 = DecisionStump()
    stump3.fit(X_train3, y_train3, w_train3)
    
    # 预测
    y_train_pred3 = stump3.predict(X_train3)
    y_test_pred3 = stump3.predict(X_test3)
    
    train_accuracy3 = calculate_accuracy(y_train3, y_train_pred3)
    test_accuracy3 = calculate_accuracy(y_test3, y_test_pred3)
    
    print(f"训练准确率: {train_accuracy3:.3f}")
    print(f"测试准确率: {test_accuracy3:.3f}")
    print(f"选择的特征: {stump3._DecisionStump__feature_index}")
    print(f"阈值: {stump3._DecisionStump__threshold:.3f}")
    print(f"方向: {stump3._DecisionStump__direction}")
    
    # 可视化并保存
    visualize_results(X_train3, y_train3, y_train_pred3, "乳腺癌数据集-训练", output_folder, "03_breast_cancer_train")
    visualize_results(X_test3, y_test3, y_test_pred3, "乳腺癌数据集-测试", output_folder, "03_breast_cancer_test")
    visualize_decision_boundary(X_train3, y_train3, stump3, "乳腺癌数据集", output_folder, "03_breast_cancer")

def demonstrate_weighted_training():
    """演示带权重的训练"""
    print("\n4. 带权重的训练示例")
    
    output_folder = "./output/decision_stump_example"
    
    # 创建数据集
    X, y = create_simple_dataset()
    
    # 创建不均匀权重（模拟AdaBoost中的情况）
    w_uniform = np.ones(len(y)) / len(y)  # 均匀权重
    w_weighted = np.random.exponential(1, len(y))  # 指数分布权重
    w_weighted = w_weighted / np.sum(w_weighted)  # 归一化
    
    # 使用均匀权重训练
    stump_uniform = DecisionStump()
    stump_uniform.fit(X, y, w_uniform)
    y_pred_uniform = stump_uniform.predict(X)
    
    # 使用不均匀权重训练
    stump_weighted = DecisionStump()
    stump_weighted.fit(X, y, w_weighted)
    y_pred_weighted = stump_weighted.predict(X)
    
    print("均匀权重:")
    print(f"  准确率: {calculate_accuracy(y, y_pred_uniform):.3f}")
    print(f"  特征: {stump_uniform._DecisionStump__feature_index}")
    print(f"  阈值: {stump_uniform._DecisionStump__threshold:.3f}")
    
    print("加权训练:")
    print(f"  准确率: {calculate_accuracy(y, y_pred_weighted):.3f}")
    print(f"  特征: {stump_weighted._DecisionStump__feature_index}")
    print(f"  阈值: {stump_weighted._DecisionStump__threshold:.3f}")
    
    # 可视化权重效果
    visualize_results(X, y, y_pred_uniform, "均匀权重", output_folder, "04_uniform_weights")
    visualize_results(X, y, y_pred_weighted, "加权训练", output_folder, "04_weighted_training")
    visualize_decision_boundary(X, y, stump_uniform, "均匀权重", output_folder, "04_uniform_weights")
    visualize_decision_boundary(X, y, stump_weighted, "加权训练", output_folder, "04_weighted_training")

def save_summary_report(output_folder):
    """保存总结报告"""
    report_path = os.path.join(output_folder, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("决策树桩 (Decision Stump) 实验报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("## 数据集要求:\n")
        f.write("1. X: 特征矩阵，形状为 (n_samples, n_features)\n")
        f.write("2. y: 标签向量，值必须为 -1 或 1 (二分类)\n")
        f.write("3. w: 权重向量，长度为 n_samples，通常归一化\n")
        f.write("4. 适用于二分类问题\n")
        f.write("5. 特征可以是连续值\n\n")
        
        f.write("## 算法特点:\n")
        f.write("1. 决策树桩是只有一个分裂节点的决策树\n")
        f.write("2. 常用作AdaBoost的弱学习器\n")
        f.write("3. 对每个特征尝试所有可能的分割点\n")
        f.write("4. 选择加权误差最小的分割方案\n")
        f.write("5. 支持带权重的训练\n\n")
        
        f.write("## 推荐数据集:\n")
        f.write("1. sklearn.datasets.make_classification() - 生成分类数据\n")
        f.write("2. sklearn.datasets.load_breast_cancer() - 乳腺癌数据集\n")
        f.write("3. sklearn.datasets.load_wine() - 葡萄酒数据集\n")
        f.write("4. sklearn.datasets.load_digits() - 手写数字数据集\n")
        f.write("5. 任何二分类数据集\n\n")
        
        f.write("## 生成的图片说明:\n")
        f.write("- 散点图: 显示数据分布和分类结果\n")
        f.write("- 决策边界图: 显示决策树桩的分割线\n")
        f.write("- 对比图: 比较不同权重设置的效果\n")
    
    print(f"总结报告已保存: {report_path}")

if __name__ == "__main__":
    main()
    demonstrate_weighted_training()
    
    output_folder = "./output"
    save_summary_report(output_folder)
    
    print("\n" + "=" * 50)
    print("所有图片和报告已保存到文件夹:", output_folder)
    print("=" * 50) 