import os
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow import keras

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
    # model = keras.Sequential([
    #     layers.Conv2D(32, (3, 3), activation=activation_fn, input_shape=(32, 32, 3)),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation=activation_fn),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Flatten(),
    #     layers.Dense(64, activation=activation_fn),
    #     layers.Dense(10, activation='softmax')
    # ])
    # return model
    model = keras.Sequential()

    # Use custom activations as layers
    if activation_fn == 'prelu':
        act_fn1 = PReLU(shared_axes=[1, 2])
        act_fn2 = PReLU(shared_axes=[1, 2])
        dense_act_fn = PReLU()
    elif activation_fn == 'leakyrelu':
        act_fn1 = LeakyReLU()
        act_fn2 = LeakyReLU()
        dense_act_fn = LeakyReLU()
    else:
        # Use string-based activations
        act_fn1 = layers.Activation(activation_fn)
        act_fn2 = layers.Activation(activation_fn)
        dense_act_fn = layers.Activation(activation_fn)

    # Build model using chosen activation layers
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(act_fn1)
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(act_fn2)
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(dense_act_fn)

    model.add(layers.Dense(10, activation='softmax'))

    return model

def train(dataset, activation_fn):
    # Initialize model from SimpleNet class
    model = create_model(activation_fn)
   
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=25)
    return model

def process(train_dataset, test_dataset, activation_fn):
    start = time.perf_counter()
    model = train(train_dataset, activation_fn)
    end = time.perf_counter()
    training_time = round(end - start, 3)
    print(f"Training time: {end - start:.2f} seconds")

    # Save the model
    model.save(f"resources/models/model_{activation_fn}.keras")

    # Evaluate the model on the test data
    start = time.perf_counter()
    test_loss, test_accuracy = model.evaluate(test_dataset)
    end = time.perf_counter()
    evaluation_time = round(end - start, 3)

    print(f"Evaluation time: {end - start:.2f} seconds")
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    return round(test_accuracy, 4), training_time, evaluation_time

def plot(activation_fn, accuracies, training_times, evaluation_times):
    colors = plt.cm.tab20.colors

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(accuracies) * 1.1)
    bars = plt.bar(activation_fn, accuracies, color=colors[:len(activation_fn)])
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Activation Functions')
    plt.xticks([])  # Remove x-axis labels
    # Add text above bars
    for bar, acc, fn in zip(bars, accuracies, activation_fn):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{acc:.4f}\n{fn}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    # Save the figure
    plt.savefig('resources/plots/activation_function_accuracy.png')
    plt.show()  


    # Plot training times
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(training_times) * 1.1)
    bars = plt.bar(activation_fn, training_times, color=colors[:len(activation_fn)])
    plt.xlabel('Activation Function')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time for Different Activation Functions')
    plt.xticks([])  # Remove x-axis labels
    # Add text above bars
    for bar, t, fn in zip(bars, training_times, activation_fn):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.2f}\n{fn}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    # Save the figure
    plt.savefig('resources/plots/activation_function_training_times.png')
    plt.show()


    # Plot evaluation times
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(evaluation_times) * 1.1)
    bars = plt.bar(activation_fn, evaluation_times, color=colors[:len(activation_fn)])
    plt.xlabel('Activation Function')
    plt.ylabel('Evaluation Time (s)')
    plt.title('Evaluation Time for Different Activation Functions')
    plt.xticks([])  # Remove x-axis labels
    # Add text above bars
    for bar, t, fn in zip(bars, evaluation_times, activation_fn):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.3f}\n{fn}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    # Save the figure
    plt.savefig('resources/plots/activation_function_eval_times.png')
    plt.show()

def main():
    # Load and preprocess the train data
    train_dict = load_all_train_data()
    train_dataset = preprocess_dataset(train_dict)
    train_dataset = train_dataset.shuffle(buffer_size=50000).batch(32)

    # Load and preprocess the test data
    test_dict = unpickle("cifar-10/test_batch")
    test_dataset = preprocess_dataset(test_dict)
    test_dataset = test_dataset.batch(32)

    activation_fn = ['sigmoid', 'tanh', 'softsign', 'relu', 'leakyrelu', 'prelu', 'softplus', 'softmax', 'swish']
    #activation_fn = ['relu', 'tanh', 'sigmoid']
    accuracies = []
    training_times = []
    evaluation_times = []
    for fn in activation_fn:
        print(f"Using activation function: {fn}")
        accuracy, tr_time, te_time = process(train_dataset, test_dataset, fn)
        accuracies.append(accuracy)
        training_times.append(tr_time)  
        evaluation_times.append(te_time)

    # print("Activation Functions: ", activation_fn)
    # print("Accuracies: ", accuracies)
    # print("Training Times: ", training_times)
    # print("Evaluation Times: ", evaluation_times)
    plot(activation_fn, accuracies, training_times, evaluation_times)

if __name__ == "__main__":
    main()