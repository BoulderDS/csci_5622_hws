import unittest

import numpy as np
from logreg import LogReg, Numbers

data = Numbers('../data/mnist.pkl.gz')
neg_example_x = data.train_x[0]
neg_example_y = data.train_y[0]
pos_example_x = data.train_x[0]
pos_example_y = data.train_y[0]

class TestLogReg(unittest.TestCase):
    def setUp(self):
        self.logreg_learnrate = LogReg(784, 0.5)
        self.logreg_nolearnrate = LogReg(784, 1.0)

    def test_learnrate(self):
        print("\nTesting: Learning Rate Update")
        w = self.logreg_learnrate.sgd_update(pos_example_x, pos_example_y)
        self.assertAlmostEqual(w[208], 0.05371094)
        self.assertAlmostEqual(w[209], 0.14453125)
        self.assertAlmostEqual(w[210], 0.20507812)
        self.assertAlmostEqual(w[211], 0.24707031)
        self.assertAlmostEqual(w[212], 0.24707031)

    def test_nolearnrate(self):
        print("\nTesting: No Learning Rate Update")
        w = self.logreg_nolearnrate.sgd_update(neg_example_x, neg_example_y)
        self.assertAlmostEqual(w[208], 0.10742188)
        self.assertAlmostEqual(w[209], 0.2890625)
        self.assertAlmostEqual(w[210], 0.41015625)
        self.assertAlmostEqual(w[211], 0.49414062)
        self.assertAlmostEqual(w[212], 0.49414062)

if __name__ == '__main__':
    unittest.main()
