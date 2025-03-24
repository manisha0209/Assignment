import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_and_normalize_data(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found. Check the path and try again.")
    df = pd.read_csv(filename)
    data = df.values
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return data


def initialize_centroids(data, k):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])


def kmeans(data, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids


def plot_clusters(data, labels, centroids, k):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(k):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
    plt.legend()
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.title(f'K-Means Clustering (k={k})')
    plt.show()


def main():
    filename = r"C:\Users\KIIT\PycharmProjects\pythonProject\assignment 3\kmeans - kmeans_blobs.csv"
    data = load_and_normalize_data(filename)

    for k in [2, 3]:
        labels, centroids = kmeans(data, k)
        plot_clusters(data, labels, centroids, k)


if __name__ == '__main__':
    main()
