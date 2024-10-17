import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D
from tensorflow.keras.optimizers import Adam


def create_xception_model(input_shape, num_classes, weights="imagenet"):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, (1, 1))(inputs)
    
    base_model = Xception(weights=weights, include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    for layer in base_model.layers[50:]:
        layer.trainable = False

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def compile_xception_model(model, lr=0.001):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
