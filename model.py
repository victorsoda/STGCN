# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import Trainer
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

import pandas as pd
import os
from lib.utils import *
from myglobals import *


class LayerNorm(nn.Block):

    def __init__(self, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps
        self.gamma = self.params.get('gamma', allow_deferred_init=True, init=mx.init.One())
        self.beta = self.params.get('beta', allow_deferred_init=True, init=mx.init.Zero())

    def forward(self, x):
        if autograd.is_training():
            _, *tmp = x.shape
            self.gamma.shape = [1] + tmp
            self.gamma._finish_deferred_init()
            self.beta.shape = [1] + tmp
            self.beta._finish_deferred_init()

        mu = x.mean(axis=1, keepdims=True)
        sigma = nd.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True))
        return ((x - mu) / (sigma + self.eps)) * self.gamma.data() + self.beta.data()


class cheb_conv(nn.Block):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, num_of_filters, K, cheb_polys, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int
        
        num_of_features: int, num of input features
        
        K: int, up K - 1 order chebyshev polynomials will use in this convolution
        
        '''
        super(cheb_conv, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polys = cheb_polys
        
        # shape of theta is (self.K, num_of_features, num_of_filters)
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init = True)
    
    def forward(self, x):
        '''
        Chebyshev graph convolution operation
    
        Parameters
        ----------
        x: mx.ndarray, graph signal matrix, shape is (batch_size, N, F, T_{r-1}), F is the num of features

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})
        
        '''
        batch_size, num_of_features, num_of_vertices, num_of_timesteps = x.shape    # (50, 32, 307, 12 - 2)
        
        self.Theta.shape = (self.K, self.num_of_filters, num_of_features)   # (3, 16, 32)
        self.Theta._finish_deferred_init()
        
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]    # (50, 32, 307)
            output = nd.zeros(shape = (self.num_of_filters, num_of_vertices, batch_size), ctx = x.context)
            for k in range(self.K):
                T_k = self.cheb_polys[k]
                theta_k = self.Theta.data()[k]
                # print("c", nd.dot(graph_signal[0], T_k).shape)   # (32, 307)
                # print("c", nd.dot(graph_signal[0], T_k).expand_dims(-1).shape)   # (32, 307, 1)
                rhs = nd.concat(*[nd.dot(graph_signal[idx], T_k).expand_dims(-1) for idx in range(batch_size)], dim = -1)
                # print("c", rhs.shape)   # (32, 307, 50)
                output = output + nd.dot(theta_k, rhs)
                # print("c", output.shape)   # (16, 307, 50)
            outputs.append(output.transpose((2, 0, 1)).expand_dims(-1))   # (50, 16, 307, 1)
        return nd.relu(nd.concat(*outputs, dim = -1))   # (50, 32, 307, 10)
    
class temporal_conv_layer(nn.Block):  # 时间模块
    def __init__(self, num_of_filters, K_t, **kwargs):
        super(temporal_conv_layer, self).__init__(**kwargs)
        
        if isinstance(num_of_filters, int) and num_of_filters % 2 != 0:
            raise ValueError("num of filters in time convolution must be even integers")
            
        self.num_of_filters = num_of_filters
        with self.name_scope():
            self.conv = nn.Conv2D(channels = num_of_filters, kernel_size = (1, K_t))
            self.residual_conv = nn.Conv2D(channels = num_of_filters // 2, kernel_size = (1, K_t))  # //代表除法并下取整
        
    def forward(self, x):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        conv_output = self.conv(x)
        
        P = conv_output[:, : self.num_of_filters // 2, :, :]
        Q = conv_output[:, self.num_of_filters // 2: , :, :]
        assert P.shape == Q.shape
        
        return P * nd.sigmoid(Q) + self.residual_conv(x)

class ST_block(nn.Block):  # 时空卷积块
    def __init__(self, backbone, **kwargs):
        super(ST_block, self).__init__(**kwargs)
        
        num_of_time_conv_filters1 = backbone['num_of_time_conv_filters1']
        num_of_time_conv_filters2 = backbone['num_of_time_conv_filters2']
        K_t = backbone['K_t']
        num_of_cheb_filters = backbone['num_of_cheb_filters']
        K = backbone['K']
        cheb_polys = backbone['cheb_polys']
        
        with self.name_scope():
            self.time_conv1 = temporal_conv_layer(num_of_time_conv_filters1, K_t)
            self.cheb_conv = cheb_conv(num_of_cheb_filters, K, cheb_polys)
            self.time_conv2 = temporal_conv_layer(num_of_time_conv_filters2, K_t)
            # self.bn = nn.BatchNorm()
            self.bn = LayerNorm()

    def forward(self, x):   # x.shape = (50, 3, 307, 12)
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is batch_size, num_of_features, num_of_vertices, num_of_timesteps
        '''
        # print(self.time_conv1(x).shape)   # 1st/2nd ST-block: (50, 32, 307, 10/6)
        # print(self.cheb_conv(self.time_conv1(x)).shape)   # 1st/2nd ST-block: (50, 16, 307, 10/6)
        # print(self.time_conv2(self.cheb_conv(self.time_conv1(x))).shape)   # 1st/2nd ST-block: (50, 32, 307, 8/4)
        return self.bn(self.time_conv2(self.cheb_conv(self.time_conv1(x))))

class STGCN(nn.Block):
    def __init__(self, backbones, final_num_of_time_filters, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        
        self.final_num_of_time_filters = final_num_of_time_filters  # main函数里设为了64，相当于两层全连接层的隐层神经元数目
        
        self.st_blocks = []
        for backbone in backbones:
            self.st_blocks.append(ST_block(backbone))
            self.register_child(self.st_blocks[-1])
        
        with self.name_scope():
            self.final_time_conv_weight = self.params.get("conv_weight", allow_deferred_init = True)
            self.final_time_conv_bias = self.params.get('conv_bias', allow_deferred_init = True)
            self.final_fc_weight = self.params.get('fc_weight', allow_deferred_init = True)
            self.final_fc_bias = self.params.get('fc_bias', allow_deferred_init = True)
        
    def forward(self, x):
        output = x
        for block in self.st_blocks:
            output = block(output)

        # 之后经历两个全连接层：从num_of_features * num_of_vertices * time_points（3维） 到
        # num_of_vertices * num_of_channel_out（2维） 到 num_of_vertices（1维）

        batch_size, num_of_features, num_of_vertices, num_of_timesteps = output.shape
        
        self.final_time_conv_weight.shape = (num_of_features * num_of_timesteps, self.final_num_of_time_filters)   # (50, 32, 307, 4)
        self.final_time_conv_weight._finish_deferred_init()
        
        self.final_time_conv_bias.shape = (1, self.final_num_of_time_filters)
        self.final_time_conv_bias._finish_deferred_init()
        
        final_conv_output =  nd.dot(output.transpose((0, 2, 1, 3)).reshape(batch_size, num_of_vertices, -1), 
                                    self.final_time_conv_weight.data()) + self.final_time_conv_bias.data()
        # print(final_conv_output.shape)    # (50, 307, 64)

        batch_size, num_of_vertices, num_of_features = final_conv_output.shape
        
        self.final_fc_weight.shape = (num_of_features, num_points_for_predict)
        self.final_fc_weight._finish_deferred_init()
        self.final_fc_bias.shape = (num_points_for_predict, )
        self.final_fc_bias._finish_deferred_init()

        return nd.dot(final_conv_output, self.final_fc_weight.data()) + self.final_fc_bias.data()   # (50, 307, num_points_to_predict)
    
if __name__ == "__main__":
    ctx = mx.gpu(0)
    # a = nd.random_uniform(shape=(32, 45, 15))
    # a = nd.concat(*a, dim=1)
    # print(a.shape)
    # a = nd.random_uniform(shape=(32, 45, 15))
    # a = nd.concat(*a, dim=0)
    # print(a.shape)
    # a = nd.random_uniform(shape=(32, 45, 15))
    # a = nd.concat(*a, dim=-1)
    # print(a.shape)

    distance_df = pd.read_csv(distance_filepath, dtype={'from': 'int', 'to': 'int'})
    num_of_vertices = 307
    A = get_adjacency_matrix(distance_df, num_of_vertices, normalized_k_threshold)
    L_tilde = scaled_Laplacian(A)

    X = get_graph_signal_matrix(num_of_vertices, ctx)    # (#verticies, #features, #timepoints) = (307, 3, 500)

    # make training/valid/test dataset
    split_line1 = int(X.shape[2] * train_prop)
    split_line2 = int(X.shape[2] * (train_prop + valid_prop))
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1: split_line2]
    test_original_data = X[:, :, split_line2:]

    # 设定： 用于训练的序列长度：12    要预测的序列长度：9     则：
    # training_data = 265 * (307, 12, 3),    training_target = 265 * (307, 9)
    training_data, training_target = make_dataset(train_original_data)
    val_data, val_target = make_dataset(val_original_data)
    testing_data, testing_target = make_dataset(test_original_data)
    print(training_data[0].shape)
    # exit(2)

    # construct chebnet model
    cheb_polys = [nd.array(i, ctx = ctx) for i in cheb_polynomial(L_tilde, cheb_K)]
    backbones = [
    {
        'num_of_time_conv_filters1': 64,
        'num_of_time_conv_filters2': 64,
        'K_t': 3,
        'num_of_cheb_filters': 16,
        'K': cheb_K,
        'cheb_polys': cheb_polys
    },
    {
        'num_of_time_conv_filters1': 64,
        'num_of_time_conv_filters2': 64,
        'K_t': 3,
        'num_of_cheb_filters': 16,
        'K': cheb_K,
        'cheb_polys': cheb_polys
    }]
    net = STGCN(backbones, 64)
    net.initialize(ctx = ctx)

    # batch_size, num_of_features, num_of_vertices, num_of_timesteps = x.shape
    # print(net(nd.random_uniform(shape = (5, 3, 307, 12), ctx=ctx)).shape)
    # exit(2)

    trainer = Trainer(net.collect_params(), optimizer, {'learning_rate': learning_rate})
    training_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(training_data, training_target), batch_size=batch_size, shuffle=True)
    validation_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(val_data, val_target), batch_size=batch_size, shuffle=False)
    testing_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(testing_data, testing_target), batch_size=batch_size, shuffle=False)

    if not os.path.exists('stgcn_params'):
        os.mkdir('stgcn_params')

    train_loss_list, val_loss_list, test_loss_list = train_model(net, trainer, training_dataloader, validation_dataloader,
                                                                 testing_dataloader)
