import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_dim=42, num_classes=8):

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
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
