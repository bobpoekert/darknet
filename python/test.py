import unittest
import yolo
from PIL import Image
import c_yolo
import numpy as np

class TestYoloNet(unittest.TestCase):

    _test_image = None

    @property
    def test_image(self):
        if self._test_image is None:
            test_image = Image.open('test.jpg').convert('RGB')
            self._test_image = np.asarray(test_image)
        return self._test_image

    def test_resize(self):
        img = c_yolo.Image()
        img.set_ndarray(self.test_image)
        resized = img.resize(200, 300)
        self.assertEqual(resized.width(), 200)
        self.assertEqual(resized.height(), 300)

    def test_predict(self):
        predictions = yolo.yolo_net().predict(self.test_image)
        self.assertIsInstance(predictions, list)
        self.assertTrue(any([v > 0 for row in predictions for v in row[4]]), 'probs are all zero')

if __name__ == '__main__':
    unittest.main()
