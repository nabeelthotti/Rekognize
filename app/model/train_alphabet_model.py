import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from app.model.load_emnist import load_emnist

def filter_labels(x, y, valid_range):
    mask = (y >= valid_range[0]) & (y < valid_range[1])
    return x[mask], y[mask]

def train_and_save_alphabet_model():
    # List of datasets to load
    datasets = [
        'emnist-balanced',
        'emnist-byclass',
        'emnist-bymerge',
        'emnist-digits',
        'emnist-letters',
        'emnist-mnist'
    ]
    
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for dataset in datasets:
        train_csv_path = f'app/data/emnist/{dataset}-train.csv'
        test_csv_path = f'app/data/emnist/{dataset}-test.csv'
        x_train, y_train = load_emnist(train_csv_path)
        x_test, y_test = load_emnist(test_csv_path)

        # Filter labels to ensure they are within the valid range [0, 26)
        x_train, y_train = filter_labels(x_train, y_train, (0, 26))
        x_test, y_test = filter_labels(x_test, y_test, (0, 26))

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    # Concatenate all datasets
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    def lr_schedule(epoch):
        initial_lr = 1e-3
        if epoch > 20:
            return initial_lr * 1e-2
        elif epoch > 10:
            return initial_lr * 1e-1
        return initial_lr

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(26, activation='softmax')  # 26 classes for the alphabet
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('app/models/best_alphabet_model.keras', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, LearningRateScheduler(lr_schedule)]

    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=25,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    model.save('app/models/alphabet_model.keras')

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_save_alphabet_model()
