import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from features.build_features import build_features


class EnhancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_generator):
        self.base_generator = base_generator
        self.class_indices = base_generator.class_indices
        self.classes = base_generator.classes  # Add this line

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, index):
        X, y = self.base_generator[index]
        enhanced_X = []
        for img in X:
            equalized_features, _ = build_features(img)
            enhanced_X.append(equalized_features)
        enhanced_X = np.array(enhanced_X)
        return tf.convert_to_tensor(enhanced_X, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)


def create_data_generators(train_path, test_path, image_size=(224, 224), batch_size=64):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        subset="training",
        class_mode="categorical",
        color_mode="grayscale",
    )

    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        subset="validation",
        class_mode="categorical",
        color_mode="grayscale",
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        color_mode="grayscale",
    )

    return (
        EnhancedDataGenerator(train_generator),
        EnhancedDataGenerator(validation_generator),
        EnhancedDataGenerator(test_generator),
    )
