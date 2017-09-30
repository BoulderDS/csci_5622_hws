
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches

        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length

        # normalize data to range [0, 1]
        self.train_x /= 255
        self.test_x /= 255

        # TODO: one hot encoding for train_y and test_y

        # TODO: build you CNN model

    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        pass

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")


    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)