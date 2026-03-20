# 手部手势识别系统
该项目基于 **MediaPipe** 和 **TensorFlow** 实现了实时手部手势识别系统，可捕获手部关键点并完成归一化处理，通过轻量级神经网络实现手势分类。

## 功能特性
- 基于MediaPipe Hands的实时手部追踪
- 基于关键点的手势分类（42维输入特征）
- 滑动窗口投票机制保障推理结果稳定性
- 模块化的训练与推理流程设计

## 项目结构
```
src/
├─train/        # 数据加载与模型训练相关代码
└─inference/    # 实时手势预测推理相关代码
models/         # 训练好的模型文件与标签文件
data/keypoints/ # 手势关键点数据集
```

## 使用方法
1. 通过 `OpenCV/capture` 模块采集手势数据
2. 训练模型：
```bash
python src/train/train.py
```
3. 运行实时推理：
```bash
python src/inference/gesture_infer.py
python main.py
```

## 注意事项
- 手势标签文件存储在 `models/labels.json` 路径下
- 确保训练阶段与推理阶段使用一致的数据预处理逻辑

### 总结
1. 核心技术：基于MediaPipe的手部关键点提取 + TensorFlow轻量级网络实现手势分类；
2. 核心设计：模块化的训练/推理流程，滑动窗口投票提升推理稳定性；
3. 核心使用：需先采集数据→训练模型→运行推理，且需保证预处理逻辑一致。