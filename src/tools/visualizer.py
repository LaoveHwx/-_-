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
    total_samples: Optional[int] = None,
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
    total = total_samples if total_samples is not None else len(y_labels)
    
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
        ax.set_ylabel(f'样本数量（总计={total}）', fontsize=12, fontweight='bold')
        ax.set_title(f'手势类别数据分布（总样本={total}）', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Sample Count (Total={total})', fontsize=12, fontweight='bold')
        ax.set_title(f'Dataset Distribution by Gesture Class (Total={total})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 保存数据集分布图: {save_path}")
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
        print(f"[OK] 保存数据有效性图: {save_path}")
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

    dataset_names = list(dataset_split.keys())
    x = np.arange(len(dataset_names))
    width = 0.6
    split_total = sum(total for total, _ in dataset_split.values())

    category_colors = plt.cm.tab20(np.linspace(0, 1, len(labels_order)))
    bottoms = np.zeros(len(dataset_names), dtype=np.float64)

    for class_idx, label in enumerate(labels_order):
        counts = np.array(
            [dataset_split[ds_name][1].count(label) for ds_name in dataset_names],
            dtype=np.float64,
        )
        ax.bar(
            x,
            counts,
            width,
            bottom=bottoms,
            label=label,
            color=category_colors[class_idx],
            edgecolor='white',
            linewidth=0.4,
        )
        bottoms += counts

    dataset_totals = [dataset_split[ds_name][0] for ds_name in dataset_names]
    for i, total in enumerate(dataset_totals):
        ax.text(
            x[i],
            total + max(dataset_totals) * 0.01,
            f'n={total}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    if lang == "zh":
        ax.set_xlabel('数据集划分', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'样本数量（总计={split_total}）', fontsize=12, fontweight='bold')
        ax.set_title(f'训练/验证/测试集样本构成（堆叠，总样本={split_total}）', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Sample Count (Total={split_total})', fontsize=12, fontweight='bold')
        ax.set_title(f'Train/Val/Test Sample Composition (Stacked, Total={split_total})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0, max(dataset_totals) * 1.08)
    ax.legend(fontsize=9, ncols=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 保存数据集划分图: {save_path}")
    plt.close()

# ================================================================================
# 【图表4：训练历史曲线 - Loss & 准确率】
# ================================================================================
def plot_training_history(
    history,
    save_path: Optional[Path] = None,
    lang: str = "en",
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
) -> None:
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
    epoch_count = len(history.history['loss'])
    epochs_idx = np.arange(1, epoch_count + 1)

    axes[0].plot(epochs_idx, history.history['loss'], label=loss_label, linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs_idx, history.history['val_loss'], label=val_loss_label, linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel(epoch_label, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('损失' if lang == "zh" else 'Loss', fontsize=12, fontweight='bold')
    if lang == "zh":
        title_loss = f'训练损失曲线（轮次=1~{epoch_count}）'
        if train_samples is not None and val_samples is not None:
            title_loss += f'\n训练集={train_samples}, 验证集={val_samples}'
    else:
        title_loss = f'Training Loss Curve (Epoch 1-{epoch_count})'
        if train_samples is not None and val_samples is not None:
            title_loss += f'\nTrain={train_samples}, Validation={val_samples}'
    axes[0].set_title(title_loss, fontsize=13, fontweight='bold')
    axes[0].set_xticks(epochs_idx)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # 准确率曲线
    acc_label = '训练准确率' if lang == "zh" else 'Train Accuracy'
    val_acc_label = '验证准确率' if lang == "zh" else 'Validation Accuracy'
    axes[1].plot(epochs_idx, history.history['accuracy'], label=acc_label, linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs_idx, history.history['val_accuracy'], label=val_acc_label, linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel(epoch_label, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('准确率' if lang == "zh" else 'Accuracy', fontsize=12, fontweight='bold')
    if lang == "zh":
        title_acc = f'训练准确率曲线（轮次=1~{epoch_count}）'
        if train_samples is not None and val_samples is not None:
            title_acc += f'\n训练集={train_samples}, 验证集={val_samples}'
    else:
        title_acc = f'Training Accuracy Curve (Epoch 1-{epoch_count})'
        if train_samples is not None and val_samples is not None:
            title_acc += f'\nTrain={train_samples}, Validation={val_samples}'
    axes[1].set_title(title_acc, fontsize=13, fontweight='bold')
    axes[1].set_xticks(epochs_idx)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 保存训练历史图: {save_path}")
    plt.close()

# ================================================================================
# 【图表5：混淆矩阵】
# ================================================================================
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels_order: List[str],
                         save_path: Optional[Path] = None,
                         lang: str = "en",
                         test_samples: Optional[int] = None) -> None:
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
    sample_count = test_samples if test_samples is not None else len(y_true_classes)
    
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
        ax.set_title(f'混淆矩阵（测试集样本={sample_count}）', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix (Test Samples={sample_count})', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha='right')
    ax.set_yticklabels(labels_order)
    
    plt.colorbar(im, ax=ax, label='数量' if lang == "zh" else 'Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 保存混淆矩阵图: {save_path}")
    plt.close()

# ================================================================================
# 【数据输出汇总】
# ================================================================================
def plot_training_summary(test_loss: float, test_acc: float, 
                         epochs: int, batch_size: int,
                         save_path: Optional[Path] = None,
                         lang: str = "en",
                         train_samples: Optional[int] = None,
                         val_samples: Optional[int] = None,
                         test_samples: Optional[int] = None) -> None:
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
        total_samples = None
        if train_samples is not None and val_samples is not None and test_samples is not None:
            total_samples = train_samples + val_samples + test_samples

        summary_text = f"""
    训练结果汇总
    ========================================
    
    测试集损失:        {test_loss:.4f}
    测试集准确率:      {test_acc:.4f} ({test_acc*100:.2f}%)
    
    训练轮数:          {epochs}
    批大小:            {batch_size}
    训练集样本:        {train_samples if train_samples is not None else 'N/A'}
    验证集样本:        {val_samples if val_samples is not None else 'N/A'}
    测试集样本:        {test_samples if test_samples is not None else 'N/A'}
    全部样本:          {total_samples if total_samples is not None else 'N/A'}
    
    ========================================
    """
    else:
        total_samples = None
        if train_samples is not None and val_samples is not None and test_samples is not None:
            total_samples = train_samples + val_samples + test_samples

        summary_text = f"""
    TRAINING SUMMARY
    ========================================
    
    Test Loss:          {test_loss:.4f}
    Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)
    
    Training Epochs:    {epochs}
    Batch Size:         {batch_size}
    Train Samples:      {train_samples if train_samples is not None else 'N/A'}
    Validation Samples: {val_samples if val_samples is not None else 'N/A'}
    Test Samples:       {test_samples if test_samples is not None else 'N/A'}
    Total Samples:      {total_samples if total_samples is not None else 'N/A'}
    
    ========================================
    """

    summary_fontfamily = 'sans-serif' if lang == "zh" else 'monospace'
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily=summary_fontfamily,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 保存训练总结: {save_path}")
    plt.close()

# ================================================================================

def generate_thesis_data_analysis_figures(
    keypoints_root: Path,
    output_dir: Path,
    labels_order: List[str],
    lang: str = "zh",
) -> List[Path]:
    """基于真实关键点数据生成4张数据分析图（非流程图）。

    图1: 各类别样本数量
    图2: 各类别有效/无效样本对比
    图3: 参考尺度(腕点到9号点距离)分布
    图4: 归一化后特征L2范数分布
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Dict[str, int] = {lb: 0 for lb in labels_order}
    valid_counts: Dict[str, int] = {lb: 0 for lb in labels_order}
    invalid_counts: Dict[str, int] = {lb: 0 for lb in labels_order}

    scale_by_class: Dict[str, List[float]] = {lb: [] for lb in labels_order}
    norm_by_class: Dict[str, List[float]] = {lb: [] for lb in labels_order}

    def _to_xy42(arr: np.ndarray) -> Optional[np.ndarray]:
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == 42:
            xy = a.reshape(21, 2)
        elif a.ndim == 2 and a.shape == (21, 2):
            xy = a
        elif a.ndim == 2 and a.shape == (21, 3):
            xy = a[:, :2]
        else:
            return None

        if not np.isfinite(xy).all():
            return None

        # 参考尺度: 第0点(腕点)到第9点距离
        scale = float(np.linalg.norm(xy[9] - xy[0]))
        if scale <= 1e-8:
            return None

        centered = xy - xy[0]
        normalized = centered / scale
        return normalized.reshape(-1)

    for lb in labels_order:
        class_dir = keypoints_root / lb
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.npy"))
        class_counts[lb] = len(files)

        for fp in files:
            try:
                arr = np.load(fp)
                # 原始尺度用于图3
                if arr.ndim == 2 and arr.shape[0] == 21:
                    xy_raw = arr[:, :2]
                elif arr.ndim == 1 and arr.shape[0] == 42:
                    xy_raw = arr.reshape(21, 2)
                else:
                    xy_raw = None

                if xy_raw is not None and np.isfinite(xy_raw).all():
                    raw_scale = float(np.linalg.norm(xy_raw[9] - xy_raw[0]))
                    if raw_scale > 1e-8:
                        scale_by_class[lb].append(raw_scale)

                feat = _to_xy42(arr)
                if feat is None:
                    invalid_counts[lb] += 1
                    continue

                valid_counts[lb] += 1
                norm_by_class[lb].append(float(np.linalg.norm(feat)))
            except Exception:
                invalid_counts[lb] += 1

    # 图1: 各类别样本数量
    p1 = output_dir / "data_analysis_01_class_count.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(labels_order))
    ys = [class_counts[lb] for lb in labels_order]
    bars = ax.bar(xs, ys, color="#4C78A8", edgecolor="black", alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_order, rotation=30, ha="right")
    ax.set_ylabel("样本数量" if lang == "zh" else "Sample Count")
    ax.set_title("各手势类别样本数量分布" if lang == "zh" else "Class-wise Sample Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 图2: 有效/无效样本对比
    p2 = output_dir / "data_analysis_02_valid_invalid.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_vals = np.array([valid_counts[lb] for lb in labels_order], dtype=np.float64)
    invalid_vals = np.array([invalid_counts[lb] for lb in labels_order], dtype=np.float64)
    ax.bar(xs, valid_vals, color="#59A14F", label="有效样本" if lang == "zh" else "Valid")
    ax.bar(xs, invalid_vals, bottom=valid_vals, color="#E15759", label="无效样本" if lang == "zh" else "Invalid")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_order, rotation=30, ha="right")
    ax.set_ylabel("样本数量" if lang == "zh" else "Sample Count")
    ax.set_title("各类别样本有效性统计" if lang == "zh" else "Valid/Invalid Samples by Class")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 图3: 原始参考尺度分布(箱线图)
    p3 = output_dir / "data_analysis_03_scale_boxplot.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    scale_data = [scale_by_class[lb] if len(scale_by_class[lb]) > 0 else [0.0] for lb in labels_order]
    bp = ax.boxplot(scale_data, patch_artist=True, labels=labels_order, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#F28E2B", alpha=0.65)
    ax.set_xticklabels(labels_order, rotation=30, ha="right")
    ax.set_ylabel("腕点到9号点距离" if lang == "zh" else "Distance(landmark0, landmark9)")
    ax.set_title("原始关键点参考尺度分布" if lang == "zh" else "Raw Scale Distribution by Class")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 图4: 归一化后特征L2范数分布(小提琴图)
    p4 = output_dir / "data_analysis_04_norm_violin.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    norm_data = [norm_by_class[lb] if len(norm_by_class[lb]) > 0 else [0.0] for lb in labels_order]
    parts = ax.violinplot(norm_data, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#B07AA1")
        pc.set_edgecolor("black")
        pc.set_alpha(0.65)
    ax.set_xticks(np.arange(1, len(labels_order) + 1))
    ax.set_xticklabels(labels_order, rotation=30, ha="right")
    ax.set_ylabel("归一化特征L2范数" if lang == "zh" else "L2 Norm of Normalized Features")
    ax.set_title("归一化后特征能量分布" if lang == "zh" else "Feature Energy After Normalization")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p4, dpi=180, bbox_inches="tight")
    plt.close(fig)

    saved = [p1, p2, p3, p4]
    for p in saved:
        print(f"[OK] 生成分析图: {p}")
    return saved

