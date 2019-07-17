from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import ndarray


__all__ = ['visualize_data', 'visualize_tsne', 'visualize_mds', 'visualize_pca']


def visualize_data(data: ndarray, labels: ndarray, method):
    """
    Reduce dimension of data to 2D and visualize using a method

    :param data: data. a 2D numpy array (batch_size, dimension)
    :param labels: labels of data (for coloring)
    :param method: a method (e.g PCA). Should be a sklearn class
    """
    transformer = method(n_components=2)
    data = transformer.fit_transform(data)
    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
    plt.title('Scatter Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def visualize_tsne(data, labels): visualize_data(data, labels, TSNE)


def visualize_mds(data, labels): visualize_data(data, labels, MDS)


def visualize_pca(data, labels): visualize_data(data, labels, PCA)
