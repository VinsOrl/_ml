import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

INPUT_SIZE = 7
HIDDEN_SIZE = 5
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1
EPOCHS = 5000
EPSILON = 1e-5

X = np.array([
    [1,1,1,1,1,1,0],
    [0,1,1,0,0,0,0],
    [1,1,0,1,1,0,1],
    [1,1,1,1,0,0,1],
    [0,1,1,0,0,1,1],
    [1,0,1,1,0,1,1],
    [1,0,1,1,1,1,1],
    [1,1,1,0,0,0,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,0,1,1],
])

y = np.eye(10)

np.random.seed(42)
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1
b1 = np.zeros((1, HIDDEN_SIZE))
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1
b2 = np.zeros((1, OUTPUT_SIZE))

def forward_pass(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)
    return output, a1

def numerical_gradient(param, compute_loss):
    grad = np.zeros_like(param)
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            original = param[i, j]
            param[i, j] = original + EPSILON
            loss_plus = compute_loss()
            param[i, j] = original - EPSILON
            loss_minus = compute_loss()
            grad[i, j] = (loss_plus - loss_minus) / (2 * EPSILON)
            param[i, j] = original
    return grad

for epoch in range(EPOCHS):
    output, hidden = forward_pass(X, W1, b1, W2, b2)
    loss = mean_squared_error(y, output)

    loss_w2 = lambda: mean_squared_error(y, sigmoid(hidden @ (W2 + 0) + b2))
    loss_b2 = lambda: mean_squared_error(y, sigmoid(hidden @ W2 + (b2 + 0)))
    loss_w1 = lambda: mean_squared_error(y, forward_pass(X, W1, b1, W2, b2)[0])
    loss_b1 = lambda: mean_squared_error(y, forward_pass(X, W1, b1, W2, b2)[0])

    dW2 = numerical_gradient(W2, loss_w2)
    db2 = numerical_gradient(b2, loss_b2)
    dW1 = numerical_gradient(W1, loss_w1)
    db1 = numerical_gradient(b1, loss_b1)

    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

final_output, _ = forward_pass(X, W1, b1, W2, b2)
predictions = np.argmax(final_output, axis=1)
print("\nPredicted Labels:", predictions)
