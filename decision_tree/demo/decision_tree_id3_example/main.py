import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import confusion_matrix
from decision_tree_id3 import ID3
import pandas as pd
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

def create_animal_dataset():
    """创建动物分类数据集（离散特征）"""
    data = {
        'size': ['small', 'medium', 'large'] * 30,
        'habitat': ['land', 'water', 'air'] * 30,
        'diet': ['meat', 'plants', 'both'] * 30,
        'social': ['pack', 'alone', 'flock'] * 30,
    }
    
    # 创建目标标签
    n_samples = len(data['size'])
    y = []
    for i in range(n_samples):
        if data['habitat'][i] == 'water':
            y.append('fish')
        elif data['habitat'][i] == 'air':
            y.append('bird')
        elif data['size'][i] == 'large' and data['diet'][i] == 'meat':
            y.append('predator')
        else:
            y.append('mammal')
    
    df = pd.DataFrame(data)
    X = df.values
    y = np.array(y)
    
    return X, y

def create_weather_dataset():
    """创建经典的天气决策数据集"""
    data = {
        'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast',
                   'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
        'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
                       'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
        'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
                    'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
        'windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true',
                 'false', 'false', 'false', 'true', 'true', 'false', 'true'],
        'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes',
                'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    
    df = pd.DataFrame(data)
    X = df[['outlook', 'temperature', 'humidity', 'windy']].values
    y = df['play'].values
    
    return X, y

def discretize_continuous_data(X, y, n_bins=3):
    """将连续数据离散化"""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_discrete = discretizer.fit_transform(X).astype(int)
    return X_discrete, y

def use_sklearn_dataset_discrete(dataset_name='iris'):
    """使用sklearn数据集并离散化"""
    print(f"=== 使用{dataset_name}数据集（离散化） ===")
    
    if dataset_name == 'iris':
        data = load_iris()
        X, y = data.data[:, :2], data.target  # 只使用前2个特征便于可视化
    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data[:, :2], data.target  # 只使用前2个特征
    
    # 离散化连续特征
    X_discrete, y = discretize_continuous_data(X, y, n_bins=3)
    
    return X, X_discrete, y

def encode_categorical_features(X):
    """将分类特征编码为数值"""
    if X.dtype == object or isinstance(X[0, 0], str):
        encoders = []
        X_encoded = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            encoder = LabelEncoder()
            X_encoded[:, i] = encoder.fit_transform(X[:, i])
            encoders.append(encoder)
        
        return X_encoded, encoders
    return X, None

def visualize_discrete_features(X, y, y_pred, title, output_folder, filename):
    """可视化离散特征的结果"""
    plt.figure(figsize=(16, 8))
    
    # 编码分类特征
    X_encoded, encoders = encode_categorical_features(X)
    y_encoded, y_encoder = encode_categorical_features(y.reshape(-1, 1))
    y_pred_encoded, _ = encode_categorical_features(y_pred.reshape(-1, 1))
    
    # 子图1：特征分布对比
    plt.subplot(2, 4, 1)
    unique_classes = np.unique(y_encoded)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_val in enumerate(unique_classes):
        mask = y_encoded.flatten() == class_val
        plt.scatter(X_encoded[mask, 0], X_encoded[mask, 1], 
                   c=[colors[i]], alpha=0.7, label=f'类别 {class_val}', s=50)
    
    plt.title(f'{title} - 真实标签')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：预测结果
    plt.subplot(2, 4, 2)
    for i, class_val in enumerate(np.unique(y_pred_encoded)):
        mask = y_pred_encoded.flatten() == class_val
        plt.scatter(X_encoded[mask, 0], X_encoded[mask, 1], 
                   c=[colors[i]], alpha=0.7, label=f'类别 {class_val}', s=50)
    
    plt.title(f'{title} - 预测结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：准确率分析
    plt.subplot(2, 4, 3)
    correct = (y_encoded.flatten() == y_pred_encoded.flatten())
    plt.scatter(X_encoded[correct, 0], X_encoded[correct, 1], 
               c='green', alpha=0.7, s=50, label='正确预测')
    plt.scatter(X_encoded[~correct, 0], X_encoded[~correct, 1], 
               c='red', alpha=0.7, s=50, label='错误预测')
    plt.title(f'{title} - 预测准确性')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4：特征值分布柱状图
    plt.subplot(2, 4, 4)
    feature_0_counts = {}
    for val in np.unique(X_encoded[:, 0]):
        for class_val in unique_classes:
            mask = (X_encoded[:, 0] == val) & (y_encoded.flatten() == class_val)
            count = np.sum(mask)
            if val not in feature_0_counts:
                feature_0_counts[val] = []
            feature_0_counts[val].append(count)
    
    x_pos = list(feature_0_counts.keys())
    width = 0.2
    for i, class_val in enumerate(unique_classes):
        counts = [feature_0_counts[val][i] for val in x_pos]
        plt.bar([x + i * width for x in x_pos], counts, width, 
               label=f'类别 {class_val}', color=colors[i], alpha=0.7)
    
    plt.title('特征1分布')
    plt.xlabel('特征值')
    plt.ylabel('计数')
    plt.legend()
    plt.xticks([x + width for x in x_pos], x_pos)
    
    # 计算并显示准确率
    accuracy = np.mean(correct)
    plt.subplot(2, 4, 5)
    plt.bar(['准确率'], [accuracy], color='skyblue', alpha=0.7)
    plt.ylim(0, 1)
    plt.title(f'总体准确率: {accuracy:.3f}')
    plt.ylabel('准确率')
    
    # 显示每个类别的精确率
    plt.subplot(2, 4, 6)
    class_accuracies = []
    class_labels = []
    for class_val in unique_classes:
        mask = y_encoded.flatten() == class_val
        if np.sum(mask) > 0:
            class_acc = np.mean((y_encoded.flatten() == y_pred_encoded.flatten())[mask])
            class_accuracies.append(class_acc)
            class_labels.append(f'类别{class_val}')
    
    plt.bar(range(len(class_accuracies)), class_accuracies, 
           color=colors[:len(class_accuracies)], alpha=0.7)
    plt.title('各类别准确率')
    plt.ylabel('准确率')
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"离散特征分析图已保存: {save_path}")

def visualize_confusion_matrix(y_true, y_pred, title, output_folder, filename):
    """绘制混淆矩阵"""
    try:
        import seaborn as sns
        
        # 获取所有类别
        all_classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=all_classes, yticklabels=all_classes)
        plt.title(f'{title} - 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 保存图片
        save_path = os.path.join(output_folder, f"{filename}_confusion.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")
    except ImportError:
        print("seaborn未安装，跳过混淆矩阵可视化")

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def print_tree_info(model, title):
    """打印决策树信息"""
    try:
        tree_size = model._ID3__tree.size()
        tree_depth = model._ID3__tree.depth()
        print(f"{title}:")
        print(f"  树的节点数: {tree_size}")
        print(f"  树的深度: {tree_depth}")
    except:
        print(f"{title}: 无法获取树的结构信息")

def main():
    print("ID3决策树 (Decision Tree ID3) 使用示例")
    print("=" * 60)
    print("注意：ID3算法只支持离散特征")
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 示例1：经典天气数据集
    print("\n1. 经典天气决策数据集")
    X1, y1 = create_weather_dataset()
    
    id3_1 = ID3()
    id3_1.fit(X1, y1)
    y_pred1 = id3_1.predict(X1)
    accuracy1 = calculate_accuracy(y1, y_pred1)
    
    print(f"准确率: {accuracy1:.3f}")
    print_tree_info(id3_1, "决策树信息")
    
    # 可视化（天气数据集只有4个样本特征，需要特殊处理）
    try:
        visualize_confusion_matrix(y1, y_pred1, "天气决策数据集", output_folder, "01_weather")
    except:
        print("天气数据集可视化跳过")
    
    # 示例2：动物分类数据集
    print("\n2. 动物分类数据集")
    X2, y2 = create_animal_dataset()
    
    # 分割数据集
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.3, random_state=42
    )
    
    id3_2 = ID3()
    id3_2.fit(X_train2, y_train2)
    y_train_pred2 = id3_2.predict(X_train2)
    y_test_pred2 = id3_2.predict(X_test2)
    
    train_accuracy2 = calculate_accuracy(y_train2, y_train_pred2)
    test_accuracy2 = calculate_accuracy(y_test2, y_test_pred2)
    
    print(f"训练准确率: {train_accuracy2:.3f}")
    print(f"测试准确率: {test_accuracy2:.3f}")
    print_tree_info(id3_2, "决策树信息")
    
    # 可视化
    visualize_discrete_features(X_test2, y_test2, y_test_pred2, "动物分类数据集", output_folder, "02_animal")
    visualize_confusion_matrix(y_test2, y_test_pred2, "动物分类数据集", output_folder, "02_animal")
    
    # 示例3：鸢尾花数据集（离散化处理）
    print("\n3. 鸢尾花数据集（离散化处理）")
    X3_continuous, X3_discrete, y3 = use_sklearn_dataset_discrete('iris')
    
    X_cont_train, X_cont_test, X_disc_train, X_disc_test, y_train3, y_test3 = train_test_split(
        X3_continuous, X3_discrete, y3, test_size=0.3, random_state=42
    )
    
    id3_3 = ID3()
    id3_3.fit(X_disc_train, y_train3)
    y_train_pred3 = id3_3.predict(X_disc_train)
    y_test_pred3 = id3_3.predict(X_disc_test)
    
    train_accuracy3 = calculate_accuracy(y_train3, y_train_pred3)
    test_accuracy3 = calculate_accuracy(y_test3, y_test_pred3)
    
    print(f"训练准确率: {train_accuracy3:.3f}")
    print(f"测试准确率: {test_accuracy3:.3f}")
    print_tree_info(id3_3, "决策树信息")
    
    # 可视化
    visualize_discrete_features(X_disc_test, y_test3, y_test_pred3, "鸢尾花数据集", output_folder, "03_iris")
    visualize_confusion_matrix(y_test3, y_test_pred3, "鸢尾花数据集", output_folder, "03_iris")
    
    # 示例4：葡萄酒数据集（离散化处理）
    print("\n4. 葡萄酒数据集（离散化处理）")
    X4_continuous, X4_discrete, y4 = use_sklearn_dataset_discrete('wine')
    
    X_cont_train4, X_cont_test4, X_disc_train4, X_disc_test4, y_train4, y_test4 = train_test_split(
        X4_continuous, X4_discrete, y4, test_size=0.3, random_state=42
    )
    
    id3_4 = ID3()
    id3_4.fit(X_disc_train4, y_train4)
    y_train_pred4 = id3_4.predict(X_disc_train4)
    y_test_pred4 = id3_4.predict(X_disc_test4)
    
    train_accuracy4 = calculate_accuracy(y_train4, y_train_pred4)
    test_accuracy4 = calculate_accuracy(y_test4, y_test_pred4)
    
    print(f"训练准确率: {train_accuracy4:.3f}")
    print(f"测试准确率: {test_accuracy4:.3f}")
    print_tree_info(id3_4, "决策树信息")
    
    # 可视化
    visualize_discrete_features(X_disc_test4, y_test4, y_test_pred4, "葡萄酒数据集", output_folder, "04_wine")
    visualize_confusion_matrix(y_test4, y_test_pred4, "葡萄酒数据集", output_folder, "04_wine")

def save_summary_report(output_folder):
    """保存总结报告"""
    report_path = os.path.join(output_folder, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ID3决策树 (Decision Tree ID3) 实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## 数据集要求:\n")
        f.write("1. X: 特征矩阵，形状为 (n_samples, n_features)\n")
        f.write("2. y: 标签向量，支持多分类\n")
        f.write("3. 只支持离散特征！\n")
        f.write("4. 连续特征需要先离散化处理\n")
        f.write("5. 适用于多分类问题\n\n")
        
        f.write("## 算法特点:\n")
        f.write("1. 使用信息增益作为分裂准则\n")
        f.write("2. 只支持离散特征\n")
        f.write("3. 生成多叉决策树\n")
        f.write("4. 没有剪枝功能\n")
        f.write("5. 容易过拟合\n\n")
        
        f.write("## 实验数据集:\n")
        f.write("1. 天气决策数据集 - 经典的决策树示例\n")
        f.write("2. 动物分类数据集 - 多特征分类任务\n")
        f.write("3. 鸢尾花数据集（离散化） - sklearn经典数据集\n")
        f.write("4. 葡萄酒数据集（离散化） - 多类别分类\n\n")
        
        f.write("## 生成的图片说明:\n")
        f.write("- 离散特征分析图: 显示特征分布、预测结果和准确率分析\n")
        f.write("- 混淆矩阵: 显示分类效果的详细情况\n")
        f.write("- 特征值分布柱状图: 展示离散特征的分布情况\n")
        f.write("- 类别准确率分析: 各个类别的分类表现\n\n")
        
        f.write("## 可视化改进:\n")
        f.write("- 针对离散特征优化的可视化方法\n")
        f.write("- 增加了特征值分布分析\n")
        f.write("- 增加了类别级别的准确率统计\n")
        f.write("- 更丰富的图表展示\n\n")
        
        f.write("## 与其他决策树算法的比较:\n")
        f.write("- ID3: 只支持离散特征，使用信息增益，容易过拟合\n")
        f.write("- C4.5: 支持连续特征，使用信息增益比，有剪枝功能\n")
        f.write("- CART: 二叉树，支持回归，使用基尼系数，有剪枝功能\n")
    
    print(f"总结报告已保存: {report_path}")

if __name__ == "__main__":
    main()
    
    output_folder = "./output"
    save_summary_report(output_folder)
    
    print("\n" + "=" * 60)
    print("所有图片和报告已保存到文件夹:", output_folder)
    print("=" * 60) 