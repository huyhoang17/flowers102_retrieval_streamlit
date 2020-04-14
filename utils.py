import os
import pickle

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import grpc
from tensorflow_serving.apis import (
    prediction_service_pb2_grpc,
    predict_pb2
)

from consts import (
    TRAIN_FD,
    TRAIN_PKL_FP,
    TRAIN_LABEL_FP
)


@st.cache
def load_prec_embs():
    with open(TRAIN_PKL_FP, "rb") as f:
        train_embs = pickle.load(f)

    with open(TRAIN_LABEL_FP, "rb") as f:
        train_labels = pickle.load(f)

    train_img_fps = wfile(TRAIN_FD)
    assert len(train_img_fps) == train_embs.shape[0]

    return train_img_fps, train_embs, train_labels


def wfile(root):
    img_fps = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            img_fps.append(os.path.join(path, name))

    return sorted(img_fps)


class FlowerArc:

    def __init__(self,
                 host="localhost",
                 port=8500,
                 model_name="flower",
                 model_signature="flower_signature",
                 input_name="input_image",
                 output_name="emb_pred"):

        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(
            self.host, self.port
        ))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )
        self.input_name = input_name
        self.output_name = output_name

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def norm_mean_std(self,
                      img):

        img = img / 255
        img = img.astype('float32')

        mean = np.mean(img, axis=(0, 1, 2))
        std = np.std(img, axis=(0, 1, 2))
        img = (img - mean) / std

        return img

    def test_preprocess(self,
                        img,
                        img_size=(384, 384),
                        expand=True):

        img = cv2.resize(img, img_size)

        # normalize image
        img = self.norm_mean_std(img)

        if expand:
            img = np.expand_dims(img, axis=0)

        return img

    def predict(self, img):

        assert img.ndim == 3

        img = self.test_preprocess(img)

        self.request.inputs[self.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=tf.float32,
                shape=img.shape
            )
        )

        result = self.stub.Predict(self.request, 10.0)

        emb_pred = tf.contrib.util.make_ndarray(
            result.outputs[self.output_name]
        )
        return emb_pred


class Saliency:

    def __init__(self,
                 host="localhost",
                 port=8500,
                 model_name="saliency",
                 model_signature="serving_default",
                 input_name="input_image",
                 output_name="pred_mask"):

        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(
            self.host, self.port
        ))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )
        self.input_name = input_name
        self.output_name = output_name

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def test_preprocess(self,
                        img,
                        img_size=(320, 240),
                        expand=True):

        img = cv2.resize(img, img_size)

        if expand:
            img = np.expand_dims(img, axis=0)

        return img

    def predict(self, img):

        assert img.ndim == 3

        img = self.test_preprocess(img)

        self.request.inputs[self.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=tf.float32,
                shape=img.shape
            )
        )

        result = self.stub.Predict(self.request, 10.0)

        pred_mask = tf.contrib.util.make_ndarray(
            result.outputs[self.output_name]
        )
        return pred_mask
