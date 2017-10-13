import argparse
import numpy as np

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        train_set, valid_set, test_set = pickle.load(f)

        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]

        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]

        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]

        f.close()

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname:
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

    # TODO: Use the Sklearn implementation of support vector machines to train a classifier to
    # distinguish 4's from 9's (using the MNIST data from the KNN homework).
    # Use scikit-learn's Grid Search (http://scikit-learn.org/stable/modules/grid_search.html) to help determine
    # optimial hyperparameters for the given model (e.g. C for linear kernel, C and p for polynomial kernel, and C and gamma for RBF).

	# -----------------------------------
	# Plotting Examples
	# -----------------------------------

	# Display in on screen
	mnist_digit_show(data.x_train[ 0,:])

	# Plot image to file
	#mnist_digit_show(data.x_train[1,:], "mnistfig.png")
