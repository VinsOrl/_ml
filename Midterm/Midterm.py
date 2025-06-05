import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    
    return (train_images, train_labels), (test_images, test_labels)

def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)  # 10 classes
    ])
    return model

def plot_sample_prediction(model, test_images, test_labels, class_names):
    predictions = model.predict(test_images)
    idx = 0  # show first test image
    predicted_label = np.argmax(predictions[idx])
    true_label = test_labels[idx]
    
    plt.figure(figsize=(5,5))
    plt.imshow(test_images[idx].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {class_names[predicted_label]}\nTrue: {class_names[true_label]}")
    plt.axis('off')
    plt.show()

def main():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Loading and preprocessing data...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    print("Building CNN model...")
    model = build_cnn_model()

    print("Compiling model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Training model...")
    model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

    print("Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    print("Showing a sample prediction...")
    plot_sample_prediction(model, test_images, test_labels, class_names)

if __name__ == "__main__":
    main()
