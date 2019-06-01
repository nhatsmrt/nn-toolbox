import numpy as np
import networkx as nx


def compute_normed_laplacian(G):
    '''
    :param G: the graph (a networkx object)
    :return: the normalized Laplacian of G
    '''
    adj_tilde = nx.adjacency_matrix(G).toarray() + np.eye(len(G.nodes), len(G.nodes))
    deg_tilde = np.sum(adj_tilde, axis=0)
    deg_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(deg_tilde))
    normed_laplacian = deg_tilde_inv_sqrt * adj_tilde * deg_tilde_inv_sqrt
    return normed_laplacian
