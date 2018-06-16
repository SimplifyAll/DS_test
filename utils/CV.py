import numpy as np
from skimage.feature import hog

class HogExtractor():
    
    @staticmethod
    def get_hog(image):
        return hog(image, orientations=8, pixels_per_cell=(2, 2),
                   cells_per_block=(1, 1), multichannel=True, block_norm="L2-Hys")

    def transform(self, X, *_):
        result = np.empty((X.shape[0], 2048))
        for i, img in enumerate(X[:3]):
            result[i] = self.get_hog(img)
        return result