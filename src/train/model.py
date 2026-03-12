# import tensorflow as tf
# from tensorflow.keras import layers, models
# 基于全连接层（Dense）的神经网络模型构建

# def build_model(input_dim=42, num_classes=8):
#
#     model = models.Sequential([
#
#         layers.Input(shape=(input_dim,)),
#
#         layers.Dense(128),
#         layers.BatchNormalization(),
#         layers.Activation("relu"),
#         layers.Dropout(0.3),
#
#         layers.Dense(64),
#         layers.Activation("relu"),
#
#         layers.Dense(num_classes, activation="softmax")
#
#     ])
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#
#     return model


# # 先训练两组测试可行性
# def build_model(num_classes):
#
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
#
#     model = Sequential()
#
#     model.add(Dense(128, input_shape=(42,)))
#     model.add(BatchNormalization())
#     model.add(Activation("relu"))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(64))
#     model.add(Activation("relu"))
#
#     model.add(Dense(num_classes, activation="softmax"))
#
#     return model


# 不用改模型数量的跟随模型数量自适应版本
import tensorflow as tf
from tensorflow.keras import layers, models

# 基于全连接层（Dense）的神经网络模型构建
def build_model(input_dim, num_classes):

    model = models.Sequential([

        layers.Input(shape=(input_dim,)),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64),
        layers.Activation("relu"),

        layers.Dense(num_classes, activation="softmax")

    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
