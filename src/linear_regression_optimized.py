import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def load_dataset():
    """
    Downloads the Auto MPG dataset and performs initial cleaning.
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    return dataset

def prepare_data(dataset):
    """
    Splits the dataset into training and testing sets, and separates labels.
    """
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    return train_features, test_features, train_labels, test_labels

def build_and_compile_model(norm):
    """
    Constructs a linear regression model with normalization.
    """
    model = tf.keras.Sequential([
        norm,
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return model

def train_model(model, train_features, train_labels, epochs=100):
    """
    Trains the model and returns the training history.
    """
    history = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        verbose=0,
        validation_split=0.2)
    return history

if __name__ == "__main__":
    # Pipeline execution
    raw_data = load_dataset()
    train_features, test_features, train_labels, test_labels = prepare_data(raw_data)

    # Normalization layer
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    # Model building and training
    linear_model = build_and_compile_model(normalizer)
    history = train_model(linear_model, train_features, train_labels)

    # Evaluation
    test_results = linear_model.evaluate(test_features, test_labels, verbose=0)
    print(f"Mean Absolute Error on Test Set: {test_results:.4f}")

    # Future-proofing: Placeholder for deep neural network comparison
    # dnn_model = build_and_compile_dnn_model(normalizer)
    # ...