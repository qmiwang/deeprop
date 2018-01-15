import cPickle as cpkl
import sys, logging, os
from sklearn import metrics 
slim_path = '/root/workspace/workspace/ROP_PAPER/slim'
if slim_path not in sys.path:
    sys.path.append(slim_path)
from nets.inception_v2 import *
from nets.inception_utils import *

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from sacred import Experiment
from data import *
from InceptionV2 import *

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL)
LOG_FORMAT = '%(asctime)s - %(message)s'
log = logging.getLogger('custom_logger')

ex = Experiment('retina-rop')
ex.logger = log

class Config:
    pass
C = Config()


@ex.config
def hyperparameters():
    data_path = '/mnt/data/lab-data/medical/ROP_data_list/20170828/verified_data_list(stage2,3)_consistent.txt.pkl'
    #data_path =  '/mnt/data/lab-data/medical/ROP_data_list/20170727/verified_data_list(normal,rop).txt.pkl'
    network = 'InceptionV2'
    gpu = 1
    num_classes = 2
    batch_size = 8
    num_epochs = 200
    imgs_per_sample = 12
    disp_batches = 10
    lr = 1.0
    optimizer = 'sgd'
    network_prefix = 'G-NET'
    save_checkpoint_path = './logs'
    drop_o=0.5 
    init_scale = 0.01
    img_height = 240
    img_width = 320
    img_channels = 3
    checkpoint_exclude_scopes = [network + '/logits', network + '/AuxLogits']
    checkpoint_path = '../slim/checkpoints/inception_v2.ckpt'
    data_balance = True
    fold = 4

@ex.capture
def get_config(_config):
    C.__dict__ = dict(_config)
    return C

@ex.capture
def get_logger(_log, save_checkpoint_path, network_prefix, network, seed):
    log_path = '%s/%s-%s/%d' % (save_checkpoint_path, network_prefix, network, seed)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(log_path + '/train.log')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(LOG_LEVEL)
    _log.addHandler(file_handler)
    return _log

## load pretrain model callback
def get_init_fn(checkpoint_path, checkpoint_exclude_scopes=[]):
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in tf.trainable_variables():#slim.get_model_variables():#
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=True), variables_to_restore


def train_epoch(sess, model, train_data, net_config, epoch_idx, log): 
    batch_idx = 0
    accs = 0.
    costs = 0.
    train_data.reset()
    for input_data, label in train_data:
        feed_dicts = {model.input_data:input_data, model.label:label, model.dropout_keep_prob:net_config.drop_o}
        acc, cost, _, step = sess.run([model.accuracy, model.cost, model.train_op, model.global_steps], feed_dict=feed_dicts)
        accs += acc
        costs += cost
        batch_idx += 1
        if batch_idx % net_config.disp_batches == 0:
            log.info('Epoch [%d] Batch [%d] : acc = [%.4f], cost = [%.6f]\r' % (epoch_idx, batch_idx, accs/batch_idx, costs/batch_idx))
    return accs/batch_idx, costs/batch_idx

def check_metric(preds, labels):
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if i == 0:
            p_l = pred
            l_l = label
        else:
            p_l = np.concatenate([p_l, pred])
            l_l = np.concatenate([l_l, label])
    idx_l_l = np.argmax(l_l, axis=1)
    idx_p_l = np.argmax(p_l, axis=1)
    precision_score = metrics.precision_score(idx_l_l, idx_p_l)
    recall_score = metrics.recall_score(idx_l_l, idx_p_l)
    accuracy_score = metrics.accuracy_score(idx_l_l, idx_p_l)
    fpr, tpr, thresholds = metrics.roc_curve(idx_l_l, p_l[:,1], pos_label=1)
    acu = metrics.auc(fpr, tpr)
    log_msg = 'precision_score:%.5f, recall_score:%.5f, accuracy_score:%.5f, acu:%.5f' % (precision_score,recall_score,accuracy_score,acu)
    return log_msg
                        
def test_epoch(sess, model, test_data, net_config, epoch_idx): 
    batch_idx = 0
    accs = 0.
    costs = 0.
    test_data.reset()
    preds = []
    labels = []
    for input_data, label in test_data:
        feed_dicts = {model.input_data:input_data, model.label:label, model.dropout_keep_prob:1.}
        acc, pred, cost = sess.run([model.accuracy, model.preds, model.cost,], feed_dicts)
        preds.append(pred)
        labels.append(label)
        batch_idx += 1
    log_msg = check_metric(preds, labels)
    return log_msg


@ex.automain
def main(seed, save_checkpoint_path, network_prefix, network, _run):
    net_config = get_config()
    log_path = '%s/%s-%s/%d' % (save_checkpoint_path, network_prefix, network, seed)

    log = get_logger()
    from sacred.commands import _format_config
    config_str = _format_config(_run.config, _run.config_modifications)
    log.info(config_str)
    np.random.seed(seed)
    
    arg_scope = inception_arg_scope(batch_norm_decay=0.9)
    initializer = tf.random_uniform_initializer(-net_config.init_scale, net_config.init_scale)

    train_datas, train_labels, test_datas, test_labels, syntext = get_data(net_config)
    cpkl.dump((train_datas, train_labels, test_datas, test_labels), open(log_path+'/data.pkl',"wb"))
    img_aug = ImageAug(net_config.img_height, net_config.img_width, net_config.img_channels)
    train_data = RetinaDataIter(train_datas, net_config.imgs_per_sample, train_labels, 
                                net_config.batch_size, 
                                net_config.img_height, net_config.img_width, net_config.img_channels,
                                image_aug=img_aug)
    test_data = RetinaDataIter(test_datas, net_config.imgs_per_sample, test_labels, net_config.batch_size, 
                               net_config.img_height, net_config.img_width, net_config.img_channels,
                               image_aug=img_aug)
    
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.visible_device_list = str(net_config.gpu)
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Graph().as_default(), tf.Session(config=gpu_config) as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.batch_norm], is_training=True):
                with tf.name_scope('train'):
                    with tf.variable_scope(net_config.network, reuse=None, initializer = initializer) as scope:
                        train_model = globals()[net_config.network](is_training=True, config=net_config, scope=scope)
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.batch_norm], is_training=False):
                with tf.name_scope('test'):
                    with tf.variable_scope(net_config.network, reuse=True, initializer = initializer) as scope:
                        test_model = globals()[net_config.network](is_training=False, config=net_config, scope=scope)
        
        log.info('[ Loading checkpoint ... ]')
        init_fn, restore_vars = get_init_fn(net_config.checkpoint_path, net_config.checkpoint_exclude_scopes)
        init_fn(sess)

        # init left variables in model
        log.info('init left...')
        uninitialized_vars = set(tf.global_variables()) - set(restore_vars)#set(tf.trainable_variables()) - set(restore_vars)
        sess.run(tf.variables_initializer(uninitialized_vars))
        saver = tf.train.Saver(max_to_keep=net_config.num_epochs)

        for epoch in range(net_config.num_epochs):
            time_start = time.time()
            acc, cost = train_epoch(sess, train_model, train_data, net_config, epoch, log)
            time_end = time.time()
            log.info('Epoch [%d] train acc = [%.4f], cost = [%.6f], time = %.2f' % (epoch, acc, cost, time_end-time_start))
            save_path = saver.save(sess, "%s/epoch-%d.ckpt" % (log_path, epoch))
            log.info("Saved to:%s" % save_path)
            #log_msg = test_epoch(sess, test_model, train_data, net_config, epoch)
            #log.info('Epoch [%d] test on test %s' % (epoch, log_msg))
            log_msg = test_epoch(sess, test_model, test_data, net_config, epoch)
            log.info('Epoch [%d] test on test %s' % (epoch, log_msg))