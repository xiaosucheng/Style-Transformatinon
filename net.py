# -*- coding: UTF-8 -*-
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

class Net:
    def __init__(self, net):
        ''' load net

        net: VGG-16 caffe net
        '''
        self.net = net

    def conv(self, input, W, b):
        ''' convolution layer

        input: C x H x W tensor
            W: N x C x H x W tensor
            b: N-dimension vector

        return: N x H x W tensor
        '''
        conv_out = T.nnet.relu(conv2d(
            input=input,
            filters=W,
            filter_shape=None,
            input_shape=None,
            border_mode='valid'
        ) + b.reshape(1, -1, 1, 1))
        # .dimshuffle('x', 0, 'x', 'x')
        return conv_out

    def pooling(self, input):
        ''' pooing layer

        input: C x H x W tensor

        return: C x H/2 x W/2 tensor
        '''
        pooled_out = pool.pool_2d(
            input=input,
            ws=(2, 2),
            ignore_border=False
        )
        return pooled_out

    def forward(self, input=input):
        ''' net forward

        input: 3 x H x W image

        returns: All feature maps of every layers
        '''
        self.blob = {}
        self.blob['conv1_1'] = self.conv(input, self.net.params['conv1_1'][0].data, self.net.params['conv1_1'][1].data)
        self.blob['conv1_2'] = self.conv(self.blob['conv1_1'], self.net.params['conv1_2'][0].data, self.net.params['conv1_2'][1].data)
        self.blob['pool1'] = self.pooling(self.blob['conv1_2'])
        self.blob['conv2_1'] = self.conv(self.blob['pool1'], self.net.params['conv2_1'][0].data, self.net.params['conv2_1'][1].data)
        self.blob['conv2_2'] = self.conv(self.blob['conv2_1'], self.net.params['conv2_2'][0].data, self.net.params['conv2_2'][1].data)
        self.blob['pool2'] = self.pooling(self.blob['conv2_2'])
        self.blob['conv3_1'] = self.conv(self.blob['pool2'], self.net.params['conv3_1'][0].data, self.net.params['conv3_1'][1].data)
        self.blob['conv3_2'] = self.conv(self.blob['conv3_1'], self.net.params['conv3_2'][0].data, self.net.params['conv3_2'][1].data)
        self.blob['conv3_3'] = self.conv(self.blob['conv3_2'], self.net.params['conv3_3'][0].data, self.net.params['conv3_3'][1].data)
        self.blob['pool3'] = self.pooling(self.blob['conv3_3'])
        self.blob['conv4_1'] = self.conv(self.blob['pool3'], self.net.params['conv4_1'][0].data, self.net.params['conv4_1'][1].data)
        self.blob['conv4_2'] = self.conv(self.blob['conv4_1'], self.net.params['conv4_2'][0].data, self.net.params['conv4_2'][1].data)
        self.blob['conv4_3'] = self.conv(self.blob['conv4_2'], self.net.params['conv4_3'][0].data, self.net.params['conv4_3'][1].data)
        self.blob['pool4'] = self.pooling(self.blob['conv4_3'])
        self.blob['conv5_1'] = self.conv(self.blob['pool4'], self.net.params['conv5_1'][0].data, self.net.params['conv5_1'][1].data)