import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from .load_emnist import load_emnist

def train_and_save_alphabet_model():
    # Load training and testing data
    train_csv_path = 'app/data/emnist/emnist-letters-train.csv'
    test_csv_path = 'app/data/emnist/emnist-letters-test.csv'
    x_train, y_train = load_emnist(train_csv_path)
    x_test, y_test = load_emnist(test_csv_path)

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

    # More advanced model architecture
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('app/models/best_alphabet_model.keras', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, LearningRateScheduler(lr_schedule)]

    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=50,  # Increased epochs
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    model.save('app/models/alphabet_model.keras')

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_save_alphabet_model()
