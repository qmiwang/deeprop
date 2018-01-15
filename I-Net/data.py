import mxnet as mx
import numpy as np
import time
import cv2 as cv
import cPickle as cpkl

import numpy as np
import cv2

INT_TYPES = [np.int8, np.int16, np.int32,
             np.uint8, np.uint16, np.uint32, np.uint64]
def convert_array_image_float32(image):
    orig_dtype = image.dtype
    image = np.array(image,dtype=np.float32)
    if orig_dtype in INT_TYPES:
        image /= np.iinfo(orig_dtype).max
    return image

def rotate_array(image, degree, interpolation=cv2.INTER_LINEAR):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    image = cv2.warpAffine(image,M,(cols,rows),flags=interpolation)
    return image

def central_array_crop(image, central_fraction=1):
    rows, cols, _ = image.shape
    off_frac = (1-central_fraction)/2
    row_start = int(off_frac*rows)
    row_end = int((1-off_frac)*rows)
    cols_start = int(off_frac*cols)
    cols_end = int((1-off_frac)*cols)
    image = image[row_start:row_end,cols_start:cols_end,:]
    return image

def inception_array_preprocessing(image, height, width, central_fraction=0.875):
    if image.dtype != np.float32:
        image = convert_array_image_float32(image)
    
    if central_fraction:
        image = central_array_crop(image, central_fraction=central_fraction)
    
    if height and width:
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LANCZOS4)
        
    image = (image - 0.5) * 2
    
    return image

class ImageAug(object):
    def __init__(self, img_height, img_width, img_channels, central_fraction=0.875):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.central_fraction = central_fraction
    
    def exec_aug(self, src):
        if isinstance(src, str):
            src = cv.imread(src)
        src = cv.resize(src, (self.img_width, self.img_height))
        return np.array(src, dtype=np.float32)
    pass

class RetinaDataIter(mx.io.DataIter):
    def __init__(self, samples, imgs_per_sample, labels, batch_size, img_height, img_width, img_channels,
                 image_aug=None, shuffle=True, shuffle_sample=False, last_batch_handle='discard'):
        assert len(samples) == len(labels), 'length of data and label should be equal'
        self.samples = samples
        self.imgs_per_sample = imgs_per_sample
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.image_aug = image_aug
        ###TODO assert img_shape == image_aug.image_shape
        self.idx_iter = mx.io.NDArrayIter(data=np.arange(len(samples)), label=np.ones([len(samples),1]),
                                          batch_size=batch_size, shuffle=shuffle, last_batch_handle=last_batch_handle)
        self.idx_iter.reset()
        self.shuffle_sample = shuffle_sample
        
        data_shape = [self.batch_size, self.imgs_per_sample, img_height, img_width, img_channels]
        self.data_buffer = np.zeros(data_shape)
    
    def iter_next(self):
        return self.idx_iter.iter_next()
    
    def next(self):
        if self.iter_next():
            return self.get_batch_new()
        else:
            raise StopIteration

    def reset(self):
        self.idx_iter.reset()
        pass
    def resample(self, sample):
        num_imgs = len(sample)
        img_idxs = np.random.permutation(max(num_imgs, self.imgs_per_sample))[:self.imgs_per_sample]
        sample = [sample[i%num_imgs] for i in img_idxs]
        return sample,
    
    def get_batch_new(self):
        sample_idxs = self.idx_iter.getdata()[0].asnumpy().astype(np.int32).tolist()
        for sample_idx_batch, sample_idx in enumerate(sample_idxs):
            #time_start = time.time()
            sample_imgs = self.samples[sample_idx]
            num_imgs = len(sample_imgs)
            for img_idx, img_true_idx in enumerate(np.random.permutation(num_imgs)[:min(num_imgs, self.imgs_per_sample)]):
                self.data_buffer[sample_idx_batch, img_idx, :, :, :] = \
                self.image_aug.exec_aug(sample_imgs[img_true_idx])
            self.resample_array(sample_idx_batch, num_imgs)
        self.data_buffer /= 255.0
        #self.data_buffer -= 0.5
        #self.data_buffer *= 2.0
        #print self.data_buffer[0,1:5,2:4,0]
        label_batch = [self.labels[i] for i in sample_idxs]
        
        return self.data_buffer, np.array(label_batch)#.reshape(-1,1)

    
    def resample_array(self, sample_idx_batch, num_imgs):
        #print num_imgs, self.imgs_per_sample
        for img_idx in range(min(num_imgs, self.imgs_per_sample), self.imgs_per_sample):
            self.data_buffer[sample_idx_batch, img_idx, :, :, :] = \
            self.data_buffer[sample_idx_batch, np.random.randint(min(num_imgs, self.imgs_per_sample)), :, :, :]
    pass

def get_data(conf):
    all_datas, syntext = cpkl.load(open(conf.data_path))
    num_samples = min([len(datas[0]) for datas in all_datas])
    train_datas, train_labels, test_datas, test_labels = [], [], [], []
    for datas in all_datas:
        idxs = np.random.permutation(len(datas[0]))
        if conf.data_balance:
            idxs = idxs[:num_samples]
        num_train = int(len(idxs) * conf.fold / (conf.fold + 1.0))
        train_datas += [datas[0][idx] for idx in idxs[:num_train]]
        test_datas += [datas[0][idx] for idx in idxs[num_train:]]
        train_labels += [[1,0] if datas[1] == 0 else [0,1]]*num_train
        test_labels += [[1,0] if datas[1] == 0 else [0,1]]*(len(idxs) - num_train)
    print 'there are %d samples in train_data and %d samples in test_data' % (len(train_labels), len(test_labels))
    return train_datas, train_labels, test_datas, test_labels, syntext