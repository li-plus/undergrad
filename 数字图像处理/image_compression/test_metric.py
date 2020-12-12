import numpy as np
import metric


class TestMetric(object):
    def setup(self):
        self.pred = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        self.test = np.array([[1, 2, 5], [4, 2, 6], [7, 8, 9]], dtype=np.uint8)
        self.mse = 1.4444444444444444
        self.psnr = 46.533795180003985
        self.rms = 1.2018504251546631

    def test_mse(self):
        mse = metric.get_mse(self.pred, self.test)
        assert np.isclose(mse, self.mse)

    def test_psnr(self):
        psnr = metric.get_psnr(self.pred, self.test)
        assert np.isclose(psnr, self.psnr)

    def test_rms(self):
        rms = metric.get_rms(self.pred, self.test)
        assert np.isclose(rms, self.rms)


def test_dct():
    import cv2
    x = np.array([[1, 2, 3, 4]], dtype=np.float64)
    output = cv2.dct(x)
    answer = [5, -2.2304424973876626, 0, -0.15851266778110815]
    assert np.isclose(output, answer).all()
