import pickle
import os
import numpy as np
#dir
#CIFAR_10_DIR = 'G:\work_dir\learn_DP\start-for-DP\data\cifar-10-batches-py'

class CIFAR_10_DATA():
    def __init__(self, CIFAR_10_DIR='G:\work_dir\learn_DP\start-for-DP\data\cifar-10-batches-py'
                 , one_hot=True, data_batch_name= "data_batch_1"):
        self.CIFAR_10_DIR = CIFAR_10_DIR
        self.one_hot = one_hot
        self.data_batch_name = data_batch_name
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = 10000
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_train_data(self):
        print(self.CIFAR_10_DIR, self.data_batch_name)
        db = self.unpickle(os.path.join(self.CIFAR_10_DIR, self.data_batch_name))
        self.data = db[b'data']
        self.lable = db[b'labels']
        return self
    def one_hot_label(self, label):
        label_one_hot_encoding = np.zeros((len(label),10))
        for i in range(len(label)):
            label_one_hot_encoding[i][label[i]] = 1
        return label_one_hot_encoding
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.lable = self.lable[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.one_hot_label(self.lable[start:end])


