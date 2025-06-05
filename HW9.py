import torch
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    x_data = torch.rand(n_samples) * 10
    noise = (torch.rand(n_samples) - 0.5) * 5
    y_data = -x_data + noise
    return x_data, y_data

def train_model(x, y, lr=0.01, epochs=1000):
    weight = torch.randn(1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    optim = torch.optim.SGD([weight, bias], lr=lr)

    for epoch in range(epochs):
        optim.zero_grad()
        prediction = weight * x + bias
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss.backward()
        optim.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return weight.detach(), bias.detach()

def plot_results(x, y, weight, bias):
    y_hat = weight * x + bias
    plt.scatter(x, y, color='red', label='Noisy Data')
    plt.plot(x, y_hat, color='blue', label='Learned Line')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Fit using PyTorch")
    plt.show()

x_vals, y_vals = generate_data()
final_w, final_b = train_model(x_vals, y_vals)
plot_results(x_vals, y_vals, final_w, final_b)
