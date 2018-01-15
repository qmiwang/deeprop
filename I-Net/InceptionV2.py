import sys

slim_path = '/root/workspace/DeepRop/slim'
if slim_path not in sys.path:
    sys.path.append(slim_path)
    
import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from nets.inception_v2 import *
from nets.inception_utils import inception_arg_scope

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

class InceptionV2(object):
    def __init__(self, is_training, config, scope):
        self._input_data = tf.placeholder(tf.float32, shape=(None, config.imgs_per_sample, config.img_height, config.img_width, config.img_channels),
                                          name='input_data_placeholder')
        self._label = tf.placeholder(tf.float32, shape=(None, config.num_classes), name='label_placeholder')
        self._dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob_placeholder')
        
        def process_feature(net, end_point='Mixed_5c', depth_multiplier=1.0, min_depth=16, concat_dim=3):
            if depth_multiplier <= 0:
                raise ValueError('depth_multiplier is not greater than zero.')
            depth = lambda d: max(int(d * depth_multiplier), min_depth)
            
            end_points = {}
            end_point = 'Mixed_5c'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                end_point = 'Mixed_5a'
                with slim.arg_scope([slim.conv2d], trainable=True):
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                            branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], scope='Conv2d_0b_3x3')
                            branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                        net = tf.concat( axis=concat_dim, values=[branch_0, branch_1, branch_2])
                        end_points[end_point] = net
                # 7 x 7 x 1024
                end_point = 'Mixed_5b'
                with slim.arg_scope([slim.conv2d], trainable=True):
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(320), [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(160), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
                        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                        end_points[end_point] = net
                # x 7 x 1024
                end_point = 'Mixed_5c'
                with slim.arg_scope([slim.conv2d], trainable=True):
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
                        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                        end_points[end_point] = net
                
            return net, end_points
        
        splitted_imgs = tf.split(self._input_data, config.imgs_per_sample, axis=1)
        
        with slim.arg_scope([slim.conv2d], trainable=True):
            with tf.variable_scope(scope) as inception_feature_scope:
                img_features = []
                for idx, img in enumerate(splitted_imgs):
                    if idx > 0:
                        inception_feature_scope.reuse_variables()
                    net, end_points = inception_v2_base(tf.squeeze(img, [1]),  # remove the second dimension 
                                            scope=inception_feature_scope, 
                                            final_endpoint='Mixed_4e',
                                            use_separable_conv=True)
                    img_features.append(net)

                img_features = tf.stack(img_features, axis=1)
        
        # get max feature per each exam
        with tf.variable_scope(scope):
            relevant_maxs = tf.reduce_max(img_features, axis=1)
            
        # process the max feature
        with tf.variable_scope(scope):
            final_feature, final_end_points = process_feature(relevant_maxs)    
            net = slim.avg_pool2d(final_feature, [8, 10], padding='VALID')
            net = tf.squeeze(net, [1, 2]) # squeeze the dimension which size is 1
            
        # final dense layer
        with tf.variable_scope(scope):
            with tf.variable_scope("logits"):
                net = tf.nn.dropout(net, self._dropout_keep_prob)
                dense = slim.fully_connected(net, config.num_classes, activation_fn=None, scope='dense')
        
        self._preds = dense
        self._preds_softmax = tf.nn.softmax(self._preds, name='preds_softmax')
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._label, logits=self._preds),
                                    name='softmax_cross_entropy_cost')
        tf.summary.scalar(self._cost.op.name, self._cost)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._preds, 1), tf.argmax(self._label, 1)), "float"), name='accuracy')
        tf.summary.scalar(self._accuracy.op.name, self._accuracy)
        self._pred_logits = tf.argmax(self._preds, 1)
        
        if not is_training:
            return
        
        self._tvars = tf.trainable_variables()
        self._nvars = 0
        for var in self._tvars:
            sh = var.get_shape().as_list()
            print(var.name, sh)
            self._nvars += np.prod(sh)
        #print('total variables', self._nvars)
        self._global_steps = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.lr, rho=0.95)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        #self._train_op = slim.learning.create_train_op(self._cost, optimizer)
        self._train_op = slim.learning.create_train_op(self._cost, optimizer, global_step=self._global_steps)
        
    @property
    def input_data(self):
        return self._input_data
    
    @property
    def label(self):
        return self._label
    
    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob
    
    @property
    def pred_logits(self):
        return self._pred_logits
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def tvars(self):
        return self._tvars
    
    @property
    def nvars(self):
        return self._nvars
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def accuracy(self):
        return self._accuracy
    
    @property
    def preds(self):
        return self._preds
    
    @property
    def preds_softmax(self):
        return self._preds_softmax
    @property
    def global_steps(self):
        return self._global_steps