# -*- coding:utf-8 -*-
import numpy as np
import json
import time
from scipy.sparse.linalg import eigs
from mxnet import nd
from mxnet import autograd
from myglobals import *


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    
    '''
    
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis = 1))
    
    L = D - W
    
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    
    K: the maximum order of chebyshev polynomials
    
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    
    '''
    
    N = L_tilde.shape[0]
    
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        
    return cheb_polynomials


def get_adjacency_matrix(distance_df, num_of_vertices, normalized_k=0.1):
    """
    Parameters
    ----------
    distance_df: pd.DataFrame, contains distance between vertices, three columns [from, to, distance]

    num_of_vertices: int, number of vertices

    normalized_k: parameter of gaussian kernel
    
    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    A = np.zeros((num_of_vertices, num_of_vertices), dtype = np.float32)

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        i, j = int(row[0]), int(row[1])
        A[i, j] = row[2]
        A[j, i] = row[2]

    # Calculates the standard deviation as theta.
    # compute the variance of the all distances which does not equal zero
    tmp = A.flatten()
    var = np.var(tmp[tmp!=0])

    # normalization
    A = np.exp(- (A ** 2) / var)

    # drop the value less than threshold
    A[A < normalized_k] = 0
    
    return A


def get_graph_signal_matrix(num_of_vertices, ctx):
    """

    :param num_of_vertices:
    :param ctx:
    :return: X: np.ndarray, graph_signal_matrix, shape is (num_of_vertices, num_of_features, num_of_time_points)
    """

    # preprocessing graph signal data
    with open('data/graph_signal_data_small.txt', 'r') as f:
        data = json.loads(f.read().strip())

    # initialize the graph signal matrix, shape is (num_of_vertices, num_of_features, num_of_samples)
    X = nd.empty(shape=(num_of_vertices,  # num_of_vertices
                        len(data[list(data.keys())[0]].keys()),  # num_of_features
                        len(list(data[list(data.keys())[0]].values())[0])),  # num_of_samples
                 ctx=ctx)

    # i is the index of the vertice
    for i in range(num_of_vertices):
        X[i, 0, :] = nd.array(data[str(i)]['flow'], ctx=ctx)
        X[i, 1, :] = nd.array(data[str(i)]['occupy'], ctx=ctx)
        X[i, 2, :] = nd.array(data[str(i)]['speed'], ctx=ctx)

    return X


def make_dataset(graph_signal_matrix):
    '''
        Parameters
        ----------
        graph_signal_matrix: graph signal matrix, shape is (num_of_vertices, num_of_features, num_of_samples)

        Returns
        ----------
        features: list[graph_signal_matrix], shape of each element is (num_of_vertices, num_of_features, num_points_for_training)

        target: list[graph_signal_matrix], shape of each element is (num_of_vertices, num_points_for_predicting)
        '''

    # generate the beginning index and the ending index of a sample, which contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_points_for_train + num_points_for_predict)) for i in
               range(graph_signal_matrix.shape[2] - (num_points_for_train + num_points_for_predict) + 1)]

    # save samples
    features, target = [], []
    for i, j in indices:
        features.append(graph_signal_matrix[:, :, i: i + num_points_for_train].transpose((1, 0, 2)))
        target.append(graph_signal_matrix[:, 0, i + num_points_for_train: j])

    return features, target


def loss(output, target):
    """
    loss function: MSE

    Parameters
    ----------
    output: mx.ndarray, output of the network, shape is (batch_size, num_of_vertices, num_points_for_predicting)

    target: mx.ndarray, target value of the prediction, shape is (batch_size, num_of_vertices, num_points_for_predicting)

    """
    return nd.sqrt(nd.sum((output - target) ** 2) / np.prod(output.shape))  # 改成了RMSE


def get_net_output_step_by_step(net, x, y, step):
    output = x
    for i in range(step):
        # output = net(x)
        # l = loss(output, y)
        origin_output = output
        output = net(output)
        if i == y.shape[-1] - 1:
            break
        output = nd.concat(origin_output[:, :, :, 1:], nd.expand_dims(output.transpose((0, 2, 1)), axis=-1), dim=-1)
    return output


def train_model(net, trainer, training_dataloader, validation_dataloader, testing_dataloader):
    '''
    train the model

    Parameters
    ----------
    net: model which has been initialized

    trainer: mxnet.gluon.Trainer which has been initialized

    training_dataloader, validation_dataloader, testing_dataloader: gluon.data.dataloader.DataLoader

    Returns
    ----------
    train_loss_list: list(float), which contains loss values of training process

    val_loss_list: list(float), which contains loss values of validation process

    test_loss_list: list(float), which contains loss values of testing process

    '''
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        t = time.time()

        train_loss_list_tmp = []
        for x, y in training_dataloader:
            with autograd.record():
                output = get_net_output_step_by_step(net, x, y, y.shape[-1])
                l = loss(output[:, :, 0], y[:, :, -1])
            l.backward()
            train_loss_list_tmp.append(l.asscalar())
            trainer.step(batch_size)

        train_loss_list.append(sum(train_loss_list_tmp) / len(train_loss_list_tmp))

        val_loss_list_tmp = []
        for x, y in validation_dataloader:
            # output = net(x)
            # val_loss_list_tmp.append(loss(output, y).asscalar())
            output = get_net_output_step_by_step(net, x, y, y.shape[-1])
            l = loss(output[:, :, 0], y[:, :, -1])
            val_loss_list_tmp.append(l.asscalar())

        val_loss_list.append(sum(val_loss_list_tmp) / len(val_loss_list_tmp))

        test_loss_list_tmp = []
        for x, y in testing_dataloader:
            # output = net(x)
            # test_loss_list_tmp.append(loss(output, y).asscalar())
            output = get_net_output_step_by_step(net, x, y, y.shape[-1])
            l = loss(output[:, :, 0], y[:, :, -1])
            test_loss_list_tmp.append(l.asscalar())

        test_loss_list.append(sum(test_loss_list_tmp) / len(test_loss_list_tmp))

        print('current epoch is %s' % (epoch + 1))
        print('training loss(RMSE):', train_loss_list[-1])
        print('validation loss(RMSE):', val_loss_list[-1])
        print('testing loss(RMSE):', test_loss_list[-1])
        print('time:', time.time() - t)
        print()

        with open('results.log', 'a') as f:
            f.write('training loss(MSE): %s' % (train_loss_list[-1]))
            f.write('\n')
            f.write('validation loss(MSE): %s' % (val_loss_list[-1]))
            f.write('\n')
            f.write('testing loss(MSE): %s' % (test_loss_list[-1]))
            f.write('\n\n')

        if (epoch + 1) % 5 == 0:
            filename = params_file_prefix + str(epoch)
            net.save_parameters(filename)

        if (epoch + 1) % decay_interval == 0:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)

    return train_loss_list, val_loss_list, test_loss_list

