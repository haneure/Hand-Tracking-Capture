import numpy as np
import tensorflow as tf


class LandmarkClassifier(object):
    def __init__(
            self,
            # model_path='model/landmark_classifier/landmark-matome-added-left-right-v2.tflite',
            # model_path='model/landmark_classifier/landmark-matome-v1.2.tflite',
            # model_path='model/landmark_classifier/landmark-matome-v1.3.tflite',
            model_path='model/landmark_classifier/landmark-matome-case3.tflite',
            # model_path='model/landmark_classifier/landmark-matome-75-25.tflite',
            # model_path='model/landmark_classifier/landmark-matome.tflite',`
            # model_path='model/landmark_classifier/landmark - 2 class.tflite',
            # model_path='model/landmark_classifier/landmark_modifier.tflite',
            num_threads=5
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        # print(self.interpreter.allocate_tensors())

        # input details dan output details ini dari model
        self.input_details = self.interpreter.get_input_details()
        print(self.input_details)
        self.output_details = self.interpreter.get_output_details()
        print(self.output_details)

    def __call__(
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        # print(self.input_details)
        # print(self.input_details[0]['index'])

        # ini landmark dengan shape 1, 63
        # print(np.array([landmark_list][0], dtype=np.float32))

        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list][0], dtype=np.float32))

        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        # print(self.output_details[0]['index'])

        result = self.interpreter.get_tensor(output_details_tensor_index)

        # print(result)

        result_index = np.argmax(np.squeeze(result))

        print(result_index)

        return result_index
