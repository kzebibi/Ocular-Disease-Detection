import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D
from tensorflow.keras.optimizers import Adam


def create_densenet121_model(input_shape, num_classes, weights="imagenet"):
    inputs = Input(shape=input_shape)
    
    # Use a Conv2D layer to convert 4 channels to 3
    x = Conv2D(3, (1, 1))(inputs)
    
    base_model = DenseNet121(weights=weights, include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    
    # Pass the converted input through the base model
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def compile_densenet121_model(model, lr=0.001):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
