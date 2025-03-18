import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from simple_net import SimpleNet

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_all_train_data():
    train_data = []
    for i in range(1, 6):
        file_path = os.path.join("cifar-10", f"data_batch_{i}")
        train_data.append(unpickle(file_path))

    train_dict = {b'data': [], b'labels': []}
    for data in train_data:
        train_dict[b'data'].extend(data[b'data'])
        train_dict[b'labels'].extend(data[b'labels']) 

    return train_dict

def preprocess_dataset(dict):
    num_images = len(dict[b'data'])

    # Convert CIFAR-10 data to TensorFlow format
    images = np.array(dict[b'data']).reshape((num_images, 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = dict[b'labels']
    images = images / 255.0  # Normalize images

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def create_model(activation_fn):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation=activation_fn, input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation_fn),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation=activation_fn),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train(dataset, test_dataset):
    # Initialize model from SimpleNet class
    model = create_model('relu')
   
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=1, validation_data=test_dataset)
    return model

def main():
    # Load and preprocess the train data
    train_dict = load_all_train_data()
    train_dataset = preprocess_dataset(train_dict)
    train_dataset = train_dataset.shuffle(buffer_size=50000).batch(32)

    # Load and preprocess the test data
    test_dict = unpickle("cifar-10/test_batch")
    test_dataset = preprocess_dataset(test_dict)
    test_dataset = test_dataset.batch(32)

    model = train(train_dataset, test_dataset)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()