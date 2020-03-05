import os
import pickle

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

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


def norm_mean_std(img):

    img = img / 255
    img = img.astype('float32')

    mean = np.mean(img, axis=(0, 1, 2))
    std = np.std(img, axis=(0, 1, 2))
    img = (img - mean) / std

    return img


def test_preprocess(img, img_size=(384, 384), expand=True):

    img = cv2.resize(img, img_size)

    # normalize image
    img = norm_mean_std(img)

    if expand:
        img = np.expand_dims(img, axis=0)

    return img


def grpc_infer(img,
               host="192.168.19.96",
               port=8700,
               model_name="flower",
               model_signature="flower_signature",
               input_name="input_image",
               output_name="emb_pred"):

    assert img.ndim == 3

    channel = grpc.insecure_channel("{}:{}".format(host, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = model_signature

    test_img = test_preprocess(img)

    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            test_img,
            dtype=tf.float32,
            shape=test_img.shape
        )
    )

    result = stub.Predict(request, 10.0)

    emb_pred = tf.contrib.util.make_ndarray(result.outputs[output_name])
    return emb_pred
