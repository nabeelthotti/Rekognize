import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore # Callback to adjust learning rate during training
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Utility for real-time data augmentation


# Function to train the model and save it to a file
def train_and_save_model():
    # Load the MNIST dataset, which contains handwritten digits.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the training and test data to include the channel dimension (for grayscale images).
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Normalize the pixel values to the range [0, 1] to improve model performance.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Set up data augmentation to artificially increase the diversity of the training data.
    datagen = ImageDataGenerator(
        rotation_range=10,  # Randomly rotate images by up to 10 degrees
        zoom_range=0.1,  # Randomly zoom images in/out by up to 10%
        width_shift_range=0.1,  # Randomly shift images horizontally by up to 10% of the width
        height_shift_range=0.1  # Randomly shift images vertically by up to 10% of the height
    )

    # Define the CNN (Convolutional Neural Network) architecture.
    model = tf.keras.models.Sequential([
        # First convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        
        # MaxPooling layer to downsample the input, reducing its dimensions.
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Dropout layer to prevent overfitting by randomly setting 25% of inputs to zero.
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional layer with 64 filters and ReLU activation.
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Another MaxPooling layer to further downsample the input.
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Batch normalization to normalize the activations of the previous layer.
        tf.keras.layers.BatchNormalization(),
        
        # Third convolutional layer with 128 filters and ReLU activation.
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        
        # Another MaxPooling layer to further reduce the spatial dimensions.
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten layer to convert the 3D output of the last convolutional layer into 1D.
        tf.keras.layers.Flatten(),
        
        # Fully connected layer with 128 neurons and ReLU activation.
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Dropout layer to further prevent overfitting by randomly setting 50% of inputs to zero.
        tf.keras.layers.Dropout(0.5),
        
        # Output layer with 10 neurons (one for each digit) and softmax activation to get probabilities.
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model using Adam optimizer, sparse categorical crossentropy loss, and accuracy as a metric.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define a learning rate schedule to gradually reduce the learning rate as training progresses.
    def lr_schedule(epoch):
        lr = 1e-3  # Initial learning rate
        if epoch > 10:
            lr *= 0.5e-3  # Reduce learning rate after 10 epochs
        elif epoch > 20:
            lr *= 1e-4  # Further reduce learning rate after 20 epochs
        return lr

    # Define callbacks for the model, including the learning rate scheduler.
    callbacks = [
        LearningRateScheduler(lr_schedule)
    ]

    # Train the model on the augmented data for 25 epochs and validate on the test set.
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=25,  # Train for 25 epochs
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    # Save the trained model to a file.
    model.save('handwritten.model.keras')
    
    # Evaluate the model on the test set and print the accuracy.
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy}")


# Function to load a pre-trained model from a file.
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# Train the model and save it.
if __name__ == "__main__":
    train_and_save_model()
