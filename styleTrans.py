# -*- coding: UTF-8 -*-
import numpy as np
import caffe
import theano
import theano.tensor as T
import cv2
from net import Net

def style_cost(a, x):
    ''' calculate cost

    :param a: style feature map
    :param x: generate feature map

    :return: loss
    '''
    M = a.shape[2] * a.shape[3]
    N = a.shape[1]
    A = T.dot(T.reshape(a, [N, M]), T.reshape(a, [N, M]).T)
    G = T.dot(T.reshape(x, [N, M]), T.reshape(x, [N, M]).T)
    loss = (1. / (4 * M ** 2)) * T.mean(T.pow((G - A), 2))
    return loss

# load caffe net
model_def = '/your/path/to/VGG16_deploy.prototxt'
model_weights = '/your/path/to/VGG16.caffemodel'
net = caffe.Net(model_def,      
                model_weights,  
                caffe.TEST)

# load imgs and preprocessing
height = 360
width = 360
content = cv2.imread('images/content.jpg')
style = cv2.imread('images/style.jpg')
content = np.cast['float32'](content)
style = np.cast['float32'](style)
mean_value = np.array([123.,117.,104.]).reshape(1,1,3)
content = content - mean_value
style = style - mean_value
np_noise = np.random.uniform(-20, 20, (height, width, 3)).astype('float32')
generate = 0.7*np_noise + 0.3*content
generate_f32 = np.asarray(generate.transpose(2, 0, 1).reshape(1, 3, height, width), dtype='float32')
generate_sh = theano.shared(value=generate_f32)

# instantiate 3 object and pass 3 imgs through the net respectively
content_class = Net(net)
style_class = Net(net)
generate_class = Net(net)
content_class.forward(content.transpose(2, 0, 1).reshape(1, 3, height, width))
style_class.forward(style.transpose(2, 0, 1).reshape(1, 3, height, width))
generate_class.forward(generate_sh)

# build graph
print "building graph..."
Layer = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
cost_style = 0.2 * sum(map(lambda l: style_cost(style_class.blob[l], generate_class.blob[l]), Layer))
cost = 8 * cost_style + 5 * T.mean(T.pow(generate_class.blob['conv4_2'] - content_class.blob['conv4_2'], 2))
g = T.grad(cost, generate_sh)
updates = [(generate_sh, generate_sh - T.cast(0.005, 'float32') * g)]
f = theano.function(inputs=[], outputs=cost, updates=updates)

# start training
print "start training..."
for step in range(100):
    loss = f()
    print "step:", step, "loss:", loss

# save result
toNpy = theano.function([], generate_sh.transpose(0, 2, 3, 1).reshape([height, width, 3]))
result = np.clip(toNpy() + mean_value, 0, 255).astype('uint8')
cv2.imwrite('output/result.png', result)
