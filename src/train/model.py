# 不用改模型数量的跟随模型数量自适应模型版本
import tensorflow as tf
from tensorflow.keras import layers, models

# 基于全连接层（Dense）的神经网络模型构建
def build_model(input_dim, num_classes):

    model = models.Sequential([

        layers.Input(shape=(input_dim,)),

        layers.Dense(128),# 学习特征组合
        layers.BatchNormalization(),# 稳定训练、加速收敛
        layers.Activation("relu"),# 提供非线性表达能力
        layers.Dropout(0.3),# 抑制过拟合
        # 再做一层特征压缩和非线性组合
        layers.Dense(64),
        layers.Activation("relu"),

        layers.Dense(num_classes, activation="softmax")# 输出每个类别概率

    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),# 稳健通用优化器
        loss="categorical_crossentropy",# 匹配 one-hot 标签的损失函数
        metrics=["accuracy"]
    )

    return model
