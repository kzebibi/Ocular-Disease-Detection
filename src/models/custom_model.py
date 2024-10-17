import tensorflow as tf


def create_custom_model(input_shape=(150, 150, 4), num_classes=6):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16,
                (7, 7),
                activation="relu",
                input_shape=input_shape,
                padding="same",
                name="L1",
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (7, 7), activation="relu", name="L2"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (7, 7), activation="relu", name="L2"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu", name="dense1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation="relu", name="dense2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_custom_model(model):
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
