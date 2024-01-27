import numpy as np
import tensorflow as tf


class LandmarkClassifier(object):
    def __init__(
            self,
            model_path='model/landmark_classifier/landmark-matome-case10.tflite',
            num_threads=10
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()

        # input details and output details from the model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']

        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list][0], dtype=np.float32))

        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
