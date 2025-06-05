from micrograd.engine import Value

x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

learning_rate = 0.1
num_epochs = 100

for epoch in range(num_epochs):
    loss = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    loss.backward()

    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    x.grad = 0
    y.grad = 0
    z.grad = 0

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.data:.4f}, x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}')


print(f'Best result: x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}')