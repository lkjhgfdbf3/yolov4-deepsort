
import glob
import os


import cv2
import numpy as np
import onnxruntime as ort


class ReidHelper:
    def __init__(self, settings):
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        #os.environ["CUDA_VISIBLE_DEVICES"]="0"

        #EP_list = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']

        model_path = settings.MODEL_PATH
        self.size = settings.SIZE
        self.sess = ort.InferenceSession(model_path, providers=providers)
        # print('-------------------------------------------')
        # print(ort.get_device())

    def infer(self, image_np):
        input_name = self.sess.get_inputs()[0].name

        image = self.preprocess(image_np)

        feat = self.sess.run(None, {input_name: image})[0]
        feat = self.normalize(feat, axis=1)
        
        return feat

    def preprocess(self, image_np):

        # the model expects RGB inputs
        original_image = image_np[:, :, ::-1]

        # Apply pre-processing to image.
        resize_width = self.size[0]
        resize_height = self.size[1]
        img = cv2.resize(original_image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        return img


    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)



