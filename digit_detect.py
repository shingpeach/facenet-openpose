#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models
import argparse
import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.classify as cls

detect_model = "model/digit-detector/detector_model.hdf5"
recognize_model = "model/digit-detector/recognize_model.hdf5"

mean_value_for_detector = 107.524
mean_value_for_recognizer = 112.833

model_input_shape = (32,32,1)

preproc_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
preproc_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

char_detector = cls.CnnClassifier(detect_model, preproc_for_detector, model_input_shape)
char_recognizer = cls.CnnClassifier(recognize_model, preproc_for_recognizer, model_input_shape)

digit_spotter = detector.DigitSpotter(char_detector, char_recognizer, rp.MserRegionProposer())

if __name__ == "__main__":
    img = cv2.imread('C:/Users/shing/Desktop/25.jpg')
    result = digit_spotter.run(img, threshold=0.5, do_nms=True, show_result=False, nms_threshold=0.1)
    print(result)







