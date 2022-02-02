import cv2
import time
import numpy as np
import onnxruntime as ort

DEBUG= os.getenv('debug')
p = lambda x: DEBUG and print(x, flush=True)

class Predictor():
    def __init__(self,
        model_path,
        input_size=(512,512),
        normalize=True,
        **kwargs
        ):
        self.class_mapping = {}
        self.model_path = model_path
        self.model = ort.InferenceSession(self.model_path)
        self.normalize = normalize

        if type(input_size) == int: #wxh
            self.input_size = [input_size, input_size]
        else:
            self.input_size = input_size

        if self.normalize:
            assert 'mean' in kwargs and 'std' in kwargs, 'Must provide mean and std'
            self.mean = kwargs['mean']
            self.std = kwargs['std']
            if isinstance(self.mean, list):
                self.mean = np.array(self.mean)
            if isinstance(self.std, list):
                self.std = np.array(self.std)

    def crop_center(self, img):
         width_expand = 0.234375
         height_expand = 0.13888888
         center = img.shape[0]//2, img.shape[1]//2
         crop_width = int(width_expand*img.shape[1])
         crop_height = int(height_expand*img.shape[0])
         crop = img[center[0]-crop_height:center[0], center[1]:center[1]+crop_width, :]
         crop = cv2.resize(crop, (128,128))
         return crop

    @staticmethod
    def norm(x, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        x = (x - mean)/std
        return x

    def prep_image(self, img):
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32)
        img = img/255
        if self.normalize:
            img = self.norm(img)
        img = np.transpose(img, (2,0,1))
        return np.expand_dims(img, 0)

    def predict(self, x):
        inp = self.prep_image(x, self.input_size, self.normalize, two_stream=self.two_stream)
        start = time.time()
        pred = self.model.run(None, {'input':inp})
        end = time.time()
        pred = pred[0][0]
        out = {v:pred[k] for k, v in self.class_mapping.items()}
        p(f"Prediction time: {(end-start):.4f} ")
        return out
