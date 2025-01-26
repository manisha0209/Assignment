import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

X = pd.read_csv("logisticX.csv").values
y = pd.read_csv("logisticY.csv").values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost_y1 = (1 / (2 * m)) * np.sum(-np.log(h[y.flatten() == 1]))
    cost_y0 = (1 / (2 * m)) * np.sum(-np.log(1 - h[y.flatten() == 0]))
    return cost_y1 + cost_y0

def gradient_descent_with_cost_tracking(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(theta, X, y))
    return theta, cost_history



theta = np.zeros((n + 1, 1))
learning_rate = 0.1
iterations = 50

theta, cost_history = gradient_descent_with_cost_tracking(X, y, theta, learning_rate, iterations)


def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

predictions = predict(X, theta)

c_matrix = confusion_matrix(y, predictions)


# accuracy = np.mean(predictions == y) * 100
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average="weighted")
recall = recall_score(y, predictions, average="weighted")
f1 = f1_score(y, predictions, average="weighted")





final_cost = compute_cost(theta, X, y)

print("Confusion Matrix")
print(c_matrix)

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"Final cost function value: {final_cost:.4f}")
print("Learning coefficients after convergence:")
print(theta)



plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), cost_history, marker='o', linestyle='-', color='blue')
plt.title("Cost Function vs. Iterations (First 50 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.grid(True)
plt.show()



def plot_decision_boundary(X, y, theta):
    plt.figure(figsize=(8, 6))

    pos = y.flatten() == 1
    neg = y.flatten() == 0

    plt.scatter(X[pos, 1], X[pos, 2], color='blue', label='Class 1')
    plt.scatter(X[neg, 1], X[neg, 2], color='red', label='Class 0')

    x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_values = -(theta[0] + theta[1] * x_values) / theta[2]

    plt.plot(x_values, y_values, color='black', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression - Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_decision_boundary(X, y, theta)

#######################################################################################################
# X_new = np.hstack((X, (X[:, 1:] ** 2)))
#
#
# y_new = y ** 2
#
#
# m, n = X_new.shape
# X_new = np.hstack((np.ones((m, 1)), X_new))
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# def compute_cost_new(theta_new, X_new, y_new):
#     m = len(y_new)
#     h = sigmoid(X_new @ theta_new)
#     cost_y1 = (1 / (2 * m)) * np.sum(-np.log(h[y_new.flatten() == 1]))
#     cost_y0 = (1 / (2 * m)) * np.sum(-np.log(1 - h[y_new.flatten() == 0]))
#     return cost_y1 + cost_y0
#
# def gradient_descent_with_cost_tracking_new(X_new, y_new, theta_new, learning_rate, iterations):
#     m = len(y_new)
#     cost_history_new = []
#     for _ in range(iterations):
#         gradient_new = (1 / m) * (X_new.T @ (sigmoid(X_new @ theta_new) - y_new))
#         theta_new -= learning_rate * gradient_new
#         cost_history_new.append(compute_cost_new(theta_new, X_new, y_new))
#     return theta_new, cost_history_new
#
#
#
# theta_new = np.zeros((X_new.shape[1], 1))
# learning_rate = 0.1
# iterations = 50
#
# theta_new, cost_history_new = gradient_descent_with_cost_tracking_new(X_new, y_new, theta_new, learning_rate, iterations)
#
#
# def predict_new(X_new, theta_new):
#     return sigmoid(X_new @ theta_new) >= 0.5
#
# predictions_new = predict_new(X_new, theta_new)
#
#
# accuracy_new = np.mean(predictions_new == y_new) * 100
#
#
# final_cost_new = compute_cost_new(theta_new, X_new, y_new)
#
#
# print(f"Accuracy: {accuracy_new:.2f}%")
# print(f"Final cost function value: {final_cost_new:.4f}")
# print("Learning coefficients after convergence:")
# print(theta_new)
#
#
#
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, iterations + 1), cost_history_new, marker='o', linestyle='-', color='blue')
# plt.title("Cost Function vs. Iterations after modifications")
# plt.xlabel("Iteration")
# plt.ylabel("Cost Function Value")
# plt.grid(True)
# plt.show()
#
#
#
# def plot_decision_boundary_new(X_new, y_new, theta_new):
#     plt.figure(figsize=(8, 6))
#
#     pos_new = y_new.flatten() == 1
#     neg_new = y_new.flatten() == 0
#
#     plt.scatter(X_new[pos_new, 1], X_new[pos_new, 2], color='blue', label='Class 1')
#     plt.scatter(X_new[neg_new, 1], X_new[neg_new, 2], color='red', label='Class 0')
#
#     x_values_new = np.linspace(X_new[:, 1].min(), X_new[:, 1].max(), 100)
#     y_values_new = -(theta_new[0] + theta_new[1] * x_values_new + theta_new[2] * (x_values_new ** 2)) / theta_new[3]
#
#     plt.plot(x_values_new, y_values_new, color='black', label='Decision Boundary')
#
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Logistic Regression - Decision Boundary after modification')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# plot_decision_boundary_new(X_new, y_new, theta_new)




