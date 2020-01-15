"""
Linear process that cannot generate models for the CIFAR-10 dataset.
"""

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA


def show_samples(x_train, y_train):
    for i in range(10):
        plt.imshow(x_train[i])
        print(y_train[i])
        plt.show()


def plot_handle(colors, legend_labels):
    handles = []
    for i in range(10):
        handles.append(mpatches.Patch(color=colors[i], label=legend_labels[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))


def plot_rgb_comparison(x, y, abs_index, ord_index, colors, legend_labels, title=None, xlabel=None, ylabel=None):
    for i in range(100):
        x_to_plot = x[i][abs_index]
        y_to_plot = x[i][ord_index]
        plt.plot(x_to_plot, y_to_plot, "-o", c=colors[(int(y[i]) - 1)])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plot_handle(colors, legend_labels)
    plt.show()


def plot_components_comparison(x, y, abs_index, ord_index, colors, legend_labels, title=None, xlabel=None, ylabel=None):
    for i in range(len(x)):
        x_to_plot = x[i][abs_index] / 255  # normalization
        y_to_plot = x[i][ord_index] / 255  # normalization
        plt.plot(x_to_plot, y_to_plot, "-o", c=colors[(int(y[i]) - 1)])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    handles = []
    for i in range(10):
        handles.append(mpatches.Patch(color=colors[i], label=legend_labels[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_pca(x_recovered_to_plot, y_train, colors, labels):
    plot_components_comparison(
        x_recovered_to_plot,
        y_train,
        1,
        2,
        colors,
        labels,
        xlabel="PCA component 1",
        ylabel="PCA component 2"
    )


def plot_best_case_scenario(colors, legend_labels):
    X_to_plot = np.zeros((10, 20))
    Y_to_plot = np.zeros((10, 20))
    for i in range(10):
        for j in range(20):
            X_to_plot[i][j] = (i / 10) + (j / 20)
            Y_to_plot[i][j] = (i / 10)

    for i in range(len(X_to_plot)):
        for j in range(len(X_to_plot[i])):
            plt.plot(X_to_plot[i][j], Y_to_plot[i][j], "-o", c=colors[i])

    plt.title("Ideal example of component to trace linear separations")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")

    plot_handle(colors, legend_labels)
    plt.show()


if __name__ == "__main__":
    colors = ['gray', 'rosybrown', 'darksalmon', 'bisque', 'tan', 'gold', 'darkkhaki', 'olivedrab', 'royalblue', 'plum']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train_flat = np.array(x_train[:100]).reshape((100, 32 * 32 * 3))
    pca_dims = PCA()
    pca_dims.fit(x_train_flat)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1

    pca = PCA(n_components=d)
    x_reduced = pca.fit_transform(x_train_flat)
    x_recovered = pca.inverse_transform(x_reduced)

    x_recovered_to_plot = x_recovered[:100].reshape((100, 32 * 32 * 3))
    y_to_plot = y_train[:100]
    plot_pca(x_recovered_to_plot, y_to_plot, colors, labels)

    plot_best_case_scenario(colors, labels)
