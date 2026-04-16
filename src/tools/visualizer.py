"""
================================================================================
【自动化可视化模块】
用于数据处理和模型训练的可视化生成，便于数据分析和论文撰写。
可在此处添加/删除具体的图表生成代码。
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # 无GUI模式

# 尝试启用常见中文字体，避免中文标题显示为方块。
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================================
# 【图表1：数据集统计分布】
# ================================================================================
def plot_dataset_distribution(
    y_labels: List[str],
    labels_order: List[str],
    save_path: Optional[Path] = None,
    lang: str = "en",
) -> None:
    """绘制各类别样本数分布直方图。
    
    Args:
        y_labels: 样本标签列表
        labels_order: 标签顺序（来自 models/labels.json）
        save_path: 保存路径，如None则显示
    """
    # 统计各类别样本数
    class_counts = {label: 0 for label in labels_order}
    for label in y_labels:
        if label in class_counts:
            class_counts[label] += 1
    
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    if lang == "zh":
        ax.set_xlabel('手势类别', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax.set_title('手势类别数据分布', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Distribution by Gesture Class', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存数据集分布图: {save_path}")
    plt.close()

# ================================================================================
# 【图表2：数据有效性对比】
# ================================================================================
def plot_data_validity(
    total_files: int,
    valid_samples: int,
    save_path: Optional[Path] = None,
    lang: str = "en",
) -> None:
    """绘制数据有效性饼图。
    
    Args:
        total_files: 总文件数
        valid_samples: 有效样本数
        save_path: 保存路径
    """
    invalid_samples = total_files - valid_samples
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [valid_samples, invalid_samples]
    if lang == "zh":
        labels = [f'有效\n({valid_samples})', f'无效\n({invalid_samples})']
    else:
        labels = [f'Valid\n({valid_samples})', f'Invalid\n({invalid_samples})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, explode=explode, textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('数据有效性检查' if lang == "zh" else 'Data Validity Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存数据有效性图: {save_path}")
    plt.close()

# ================================================================================
# 【图表3：训练集/验证集/测试集分布对比】
# ================================================================================
def plot_train_val_test_split(dataset_split: Dict[str, Tuple[int, List[str]]], 
                              labels_order: List[str],
                              save_path: Optional[Path] = None,
                              lang: str = "en") -> None:
    """绘制训练/验证/测试集的样本数分布对比。
    
    Args:
        dataset_split: 字典，格式 {'train': (总数, [标签列表]), ...}
        labels_order: 标签顺序
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels_order))
    width = 0.25
    
    datasets_info = []
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    for (dataset_name, (_, y_labels)), color in zip(dataset_split.items(), colors):
        counts = [y_labels.count(label) for label in labels_order]
        datasets_info.append((dataset_name, counts, color))
    
    for i, (dataset_name, counts, color) in enumerate(datasets_info):
        offset = (i - 1) * width
        ax.bar(x + offset, counts, width, label=dataset_name, color=color, 
               alpha=0.8, edgecolor='black')
    
    if lang == "zh":
        ax.set_xlabel('手势类别', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax.set_title('训练/验证/测试集分布', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax.set_title('Train/Val/Test Set Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存数据集划分图: {save_path}")
    plt.close()

# ================================================================================
# 【图表4：训练历史曲线 - Loss & 准确率】
# ================================================================================
def plot_training_history(history, save_path: Optional[Path] = None, lang: str = "en") -> None:
    """绘制训练过程中Loss和准确率的变化曲线。
    
    Args:
        history: 若干Keras training history对象
        save_dir: 保存目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    loss_label = '训练损失' if lang == "zh" else 'Train Loss'
    val_loss_label = '验证损失' if lang == "zh" else 'Validation Loss'
    epoch_label = '轮次' if lang == "zh" else 'Epoch'
    axes[0].plot(history.history['loss'], label=loss_label, linewidth=2, marker='o', markersize=4)
    axes[0].plot(history.history['val_loss'], label=val_loss_label, linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel(epoch_label, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('损失' if lang == "zh" else 'Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('训练损失曲线' if lang == "zh" else 'Training Loss Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # 准确率曲线
    acc_label = '训练准确率' if lang == "zh" else 'Train Accuracy'
    val_acc_label = '验证准确率' if lang == "zh" else 'Validation Accuracy'
    axes[1].plot(history.history['accuracy'], label=acc_label, linewidth=2, marker='o', markersize=4)
    axes[1].plot(history.history['val_accuracy'], label=val_acc_label, linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel(epoch_label, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('准确率' if lang == "zh" else 'Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('训练准确率曲线' if lang == "zh" else 'Training Accuracy Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存训练历史图: {save_path}")
    plt.close()

# ================================================================================
# 【图表5：混淆矩阵】
# ================================================================================
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels_order: List[str],
                         save_path: Optional[Path] = None,
                         lang: str = "en") -> None:
    """绘制混淆矩阵热力图。
    
    Args:
        y_true: 真实标签（one-hot编码，shape: (N, num_classes)）
        y_pred: 预测输出（模型原始输出，shape: (N, num_classes)）
        labels_order: 标签顺序
        save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix
    
    # 转换为类别索引
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    
    # 添加数值标注
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{int(z)}', ha='center', va='center',
                color='white' if z > cm.max() / 2 else 'black', fontsize=10)
    
    if lang == "zh":
        ax.set_xlabel('预测类别', fontsize=12, fontweight='bold')
        ax.set_ylabel('真实类别', fontsize=12, fontweight='bold')
        ax.set_title('混淆矩阵（测试集）', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha='right')
    ax.set_yticklabels(labels_order)
    
    plt.colorbar(im, ax=ax, label='数量' if lang == "zh" else 'Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存混淆矩阵图: {save_path}")
    plt.close()

# ================================================================================
# 【数据输出汇总】
# ================================================================================
def plot_training_summary(test_loss: float, test_acc: float, 
                         epochs: int, batch_size: int,
                         save_path: Optional[Path] = None,
                         lang: str = "en") -> None:
    """生成训练总结文本图。
    
    Args:
        test_loss: 测试集Loss
        test_acc: 测试集准确率
        epochs: 训练轮数
        batch_size: 批次大小
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    if lang == "zh":
        summary_text = f"""
    训练结果汇总
    ========================================
    
    测试集损失:        {test_loss:.4f}
    测试集准确率:      {test_acc:.4f} ({test_acc*100:.2f}%)
    
    训练轮数:          {epochs}
    批大小:            {batch_size}
    
    ========================================
    """
    else:
        summary_text = f"""
    TRAINING SUMMARY
    ========================================
    
    Test Loss:          {test_loss:.4f}
    Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)
    
    Training Epochs:    {epochs}
    Batch Size:         {batch_size}
    
    ========================================
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存训练总结: {save_path}")
    plt.close()

# ================================================================================
