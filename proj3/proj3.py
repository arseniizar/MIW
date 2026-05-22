import numpy as np
import matplotlib.pyplot as plt
import glob
import random


def load_and_split_data(filepath, split_ratio=0.8):
    try:
        data = np.loadtxt(filepath)
        print(f"Successfully loaded data from {filepath}")
    except OSError:
        print(f"File '{filepath}' not found. Generating dummy non-linear data for demonstration.")
        x_dummy = np.linspace(-3, 3, 200)
        y_dummy = np.sin(x_dummy) + np.random.normal(0, 0.1, 200)
        data = np.column_stack((x_dummy, y_dummy))

    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    split_index = int(len(X) * split_ratio)
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    return X_train, Y_train, X_test, Y_test


class NeuralNetwork:
    def __init__(self, input_size=1, hidden_size=10, output_size=1, activation='tanh', lr=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.lr = lr
        self.activation = activation

    def _activation(self, s):
        if self.activation == 'tanh':
            return np.tanh(s)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(s, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, s)

    def _activation_derivative(self, s):
        if self.activation == 'tanh':
            return 1 - np.tanh(s) ** 2
        elif self.activation == 'sigmoid':
            f = self._activation(s)
            return f * (1 - f)
        elif self.activation == 'relu':
            return 1 / (1 + np.exp(-np.clip(s, -500, 500)))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activation(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2
        return self.A2

    def backward(self, X, Y):
        E2 = Y - self.A2
        delta2 = E2 * 1.0

        E1 = np.dot(delta2, self.W2.T)
        delta1 = E1 * self._activation_derivative(self.Z1)

        dW2 = np.dot(self.A1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        m = X.shape[0]
        self.W2 += (self.lr / m) * dW2
        self.b2 += (self.lr / m) * db2
        self.W1 += (self.lr / m) * dW1
        self.b1 += (self.lr / m) * db1

    def train_batch(self, X, Y, epochs=1000, X_test=None, Y_test=None):
        print(f"--- Starting Batch Training | Activation: {self.activation.upper()} | Epochs: {epochs} ---")
        train_loss, test_loss = [], []
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, Y)

            train_loss.append(np.mean((Y - self.A2) ** 2))
            if X_test is not None:
                test_loss.append(np.mean((Y_test - self.forward(X_test)) ** 2))

            if epoch == 0 or (epoch + 1) % (epochs // 4) == 0:
                test_msg = f" | Test MSE: {test_loss[-1]:.4f}" if X_test is not None else ""
                print(f"Epoch {epoch + 1:4d}/{epochs} -> Train MSE: {train_loss[-1]:.4f}{test_msg}")
        print("Batch training complete.\n")
        return train_loss, test_loss

    def train_online(self, X, Y, epochs=1000, X_test=None, Y_test=None):
        print(f"--- Starting Online Training | Activation: {self.activation.upper()} | Epochs: {epochs} ---")
        train_loss, test_loss = [], []
        m = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)

            for i in indices:
                x_i = X[i:i + 1]
                y_i = Y[i:i + 1]
                self.forward(x_i)
                self.backward(x_i, y_i)

            train_loss.append(np.mean((Y - self.forward(X)) ** 2))
            if X_test is not None:
                test_loss.append(np.mean((Y_test - self.forward(X_test)) ** 2))

            if epoch == 0 or (epoch + 1) % (epochs // 4) == 0:
                test_msg = f" | Test MSE: {test_loss[-1]:.4f}" if X_test is not None else ""
                print(f"Epoch {epoch + 1:4d}/{epochs} -> Train MSE: {train_loss[-1]:.4f}{test_msg}")
        print("Online training complete.\n")
        return train_loss, test_loss


if __name__ == "__main__":
    data_files = glob.glob("Dane/dane*.txt")

    if data_files:
        selected_file = random.choice(data_files)
        print(f"Found {len(data_files)} dataset files. Randomly selected: {selected_file}")
    else:
        selected_file = "Dane/daneXX.txt"
        print("No dataset files found in 'Dane/' directory.")

    X_train, Y_train, X_test, Y_test = load_and_split_data(selected_file, split_ratio=0.8)

    nn_batch = NeuralNetwork(hidden_size=15, activation='tanh', lr=0.1)
    train_loss_batch, test_loss_batch = nn_batch.train_batch(X_train, Y_train, epochs=2000, X_test=X_test,
                                                             Y_test=Y_test)

    nn_online = NeuralNetwork(hidden_size=15, activation='tanh', lr=0.01)
    train_loss_online, test_loss_online = nn_online.train_online(X_train, Y_train, epochs=500, X_test=X_test,
                                                                 Y_test=Y_test)

    nn_relu = NeuralNetwork(hidden_size=15, activation='relu', lr=0.001)
    train_loss_relu, test_loss_relu = nn_relu.train_batch(X_train, Y_train, epochs=2000, X_test=X_test, Y_test=Y_test)

    print("Rendering loss evaluation plots...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_batch, label='Train Loss')
    plt.plot(test_loss_batch, label='Test Loss')
    plt.title("Batch (Tanh) - Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_loss_online, label='Train Loss')
    plt.plot(test_loss_online, label='Test Loss')
    plt.title("Online (Tanh) - Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_loss_relu, label='Train Loss')
    plt.plot(test_loss_relu, label='Test Loss')
    plt.title("Batch (ReLU) - Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
