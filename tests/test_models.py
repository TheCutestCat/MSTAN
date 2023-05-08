import unittest
import numpy as np
from data.TestDataLoader import TestDataLoader


class MultiSourceProcess(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data_path = '../data/test_small_data.csv'
        self.dataloader = TestDataLoader(data_path, 'MultiSourceProcess')
        self.T0 = 10
        self.tau = 5
        self.batch_size = 32
    @classmethod
    def tearDownClass(self):
        # 清除数据加载器
        del self.dataloader

    def test_dataloder(self):
        for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader):
            self.assertTrue(np.array_equal(np.array(en_x.shape),np.array([self.batch_size,self.T0,2])))
            self.assertTrue(np.array_equal(np.array(wind_x.shape),np.array([self.batch_size,self.tau,3])))
            self.assertTrue(np.array_equal(np.array(other_x.shape),np.array([self.batch_size,self.tau,6])))
            self.assertTrue(np.array_equal(np.array(y.shape),np.array([self.batch_size,self.T0+self.tau])))
            # break

if __name__ == '__main__':
    unittest.main()
