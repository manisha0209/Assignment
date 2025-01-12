import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv('linearX.csv', header=None).values
y = pd.read_csv('linearY.csv', header=None).values


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


x = normalize(x)

m = len(y)
x = np.hstack((np.ones((m, 1)), x))

t = np.zeros((2, 1))
lr = 0.05
iterations = 50


def compute_cost(x, y, t):
    predictions = np.dot(x, t)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


def gradient_descent(x, y, t, lr, tolerance=1e-6, iterations=50):
    cost_history = []
    for i in range(iterations):
        predictions = np.dot(x, t)
        errors = predictions - y
        t -= (lr / m) * np.dot(x.T, errors)
        cost = compute_cost(x, y, t)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            print(f"Final cost: {cost_history[-1]}")
            return t, cost_history

    print(f"Reached max iterations without convergence.")
    return t, cost_history
def batch_gradient_descent(x, y, t, lr, iterations):
    cost_history = []
    for _ in range(iterations):
        predictions = np.dot(x, t)
        errors = predictions - y
        t -= (lr / m) * np.dot(x.T, errors)
        cost_history.append(compute_cost(x, y, t))
    return t, cost_history

def stochastic_gradient_descent(x, y, t, lr, iterations):
    cost_history = []
    for _ in range(iterations):
        for i in range(m):
            xi = x[i, :].reshape(1, -1)  # Single example
            yi = y[i]
            prediction = np.dot(xi, t)
            error = prediction - yi
            t -= lr * xi.T * error
        cost_history.append(compute_cost(x, y, t))
    return t, cost_history

def mini_batch_gradient_descent(x, y, t, lr, iterations, batch_size):
    cost_history = []
    for _ in range(iterations):
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            predictions = np.dot(x_batch, t)
            errors = predictions - y_batch
            t -= (lr / batch_size) * np.dot(x_batch.T, errors)
        cost_history.append(compute_cost(x, y, t))
    return t, cost_history

# learning_rates = [0.005, 0.5, 5]
# iterations = 50



t, cost_history = gradient_descent(x, y, t, lr)
t_batch, cost_batch = batch_gradient_descent(x, y, np.zeros((2, 1)), lr, iterations)
t_sgd, cost_sgd = stochastic_gradient_descent(x, y, np.zeros((2, 1)), lr, iterations)
t_mini_batch, cost_mini_batch = mini_batch_gradient_descent(x, y, np.zeros((2, 1)), lr, iterations, batch_size=10)


print("Learning parameters (theta):")
print(t)
print(f"Cost function value after convergence: {cost_history[-1]}")

print("Final parameters (theta):")
print(t)

plt.plot(range(len(cost_history)), cost_history)
plt.title("Cost Function over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_batch)), cost_batch, label='Batch Gradient Descent', color='blue')
plt.plot(range(len(cost_sgd)), cost_sgd, label='Stochastic Gradient Descent', color='green')
plt.plot(range(len(cost_mini_batch)), cost_mini_batch, label='Mini-Batch Gradient Descent', color='red')
plt.title('Cost Function vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.scatter(x[:, 1], y, color='blue', label='Data points')
plt.plot(x[:, 1], np.dot(x, t), color='red', label='Regression line')
plt.xlabel("Normalized X")
plt.ylabel("Y")
plt.legend()
plt.show()

# plt.figure(figsize=(10, 8))
# for lr in learning_rates:
#     _, cost_history = gradient_descent(x, y, t, lr, iterations)
#     plt.plot(range(len(cost_history)), cost_history, label=f"learning_rate = {lr}")
# plt.title("Cost Function over first 50 iterations for different learning rates")
# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()