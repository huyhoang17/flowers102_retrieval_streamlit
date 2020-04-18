import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


def _encoder(input_layer):
    inner = layers.Conv2D(64, (3, 3), padding="same",
                          activation="relu")(input_layer)
    inner = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inner)
    inner = layers.MaxPooling2D((2, 2))(inner)

    # block 02
    for no_filter in [64, 128]:
        inner = layers.Conv2D(
            no_filter, (3, 3),
            padding="same",
            activation="relu",
        )(inner)
    inner = layers.MaxPooling2D((2, 2))(inner)

    # block 03
    for _ in range(3):
        inner = layers.Conv2D(
            256, (3, 3),
            padding="same",
            activation="relu",
        )(inner)
    layer_10 = layers.MaxPooling2D((2, 2), name='layer_10')(inner)

    # block 04
    layer_11 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu")(layer_10)
    layer_12 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu")(layer_11)
    layer_13 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu")(layer_12)
    layer_14 = layers.MaxPooling2D((1, 1))(layer_13)

    layer_15 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu", dilation_rate=(2, 2))(layer_14)
    layer_16 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu", dilation_rate=(2, 2))(layer_15)
    layer_17 = layers.Conv2D(512, (3, 3), padding="same",
                             activation="relu", dilation_rate=(2, 2))(layer_16)
    layer_18 = layers.MaxPooling2D((1, 1))(layer_17)

    layer_19 = layers.concatenate([layer_10, layer_14, layer_18])

    return layer_19


def _upsample(stack, shape, factor=1):
    # image, size
    stack = tf.image.resize_bilinear(
        stack,
        (shape[1] * factor, shape[2] * factor)
    )
    return stack


# ASPP
def _aspp(inner_layer):
    branch_01 = layers.Conv2D(
        256, (1, 1), padding="same", activation=None
    )(inner_layer)
    branch_01 = layers.BatchNormalization()(branch_01)
    branch_01 = layers.Activation("relu")(branch_01)

    branch_02 = layers.Conv2D(
        256, (3, 3), padding="same", activation=None, dilation_rate=(4, 4)
    )(inner_layer)
    branch_02 = layers.BatchNormalization()(branch_02)
    branch_02 = layers.Activation("relu")(branch_02)

    branch_03 = layers.Conv2D(
        256, (3, 3), padding="same", activation=None, dilation_rate=(8, 8)
    )(inner_layer)
    branch_03 = layers.BatchNormalization()(branch_03)
    branch_03 = layers.Activation("relu")(branch_03)

    branch_04 = layers.Conv2D(
        256, (3, 3), padding="same", activation=None, dilation_rate=(12, 12)
    )(inner_layer)
    branch_04 = layers.BatchNormalization()(branch_04)
    branch_04 = layers.Activation("relu")(branch_04)

    # image-level feature
    branch_05 = layers.GlobalAveragePooling2D()(inner_layer)
    branch_05 = layers.Reshape((1, 1, branch_05.get_shape()[1]))(branch_05)

    # reduce depth size of feature map: 1280 --> 256
    branch_05 = layers.Conv2D(
        256, (1, 1), padding="valid", activation=None
    )(branch_05)
    branch_05 = layers.BatchNormalization()(branch_05)
    branch_05 = layers.Activation("relu")(branch_05)

    # bilinear upsampling
    shape = inner_layer.get_shape()
    branch_05 = _upsample(branch_05, shape, 1)

    branch_aspp = layers.concatenate(
        [branch_01, branch_02, branch_03, branch_04, branch_05]
    )
    branch_aspp = layers.Conv2D(
        256, (1, 1), padding="same", activation=None
    )(branch_aspp)
    branch_aspp = layers.BatchNormalization()(branch_aspp)
    branch_aspp = layers.Activation("relu")(branch_aspp)

    return branch_aspp


# DECODER
def _decoder(aspp_layer):
    deco_01 = _upsample(aspp_layer, aspp_layer.get_shape(), 2)
    deco_01 = layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    )(deco_01)

    deco_02 = _upsample(deco_01, deco_01.get_shape(), 2)
    deco_02 = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    )(deco_02)

    deco_03 = _upsample(deco_02, deco_02.get_shape(), 2)
    deco_03 = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    )(deco_03)

    deco_out = layers.Conv2D(
        1, (3, 3), padding="same", activation=None
    )(deco_03)
    print(deco_out.get_shape())

    return deco_out


def saliency_net(img_size=(240, 320, 3)):
    input_layer = layers.Input(
        name='input_image', shape=img_size, dtype='float32')

    layer_19 = _encoder(input_layer)
    aspp_layer = _aspp(layer_19)
    deco_out = _decoder(aspp_layer)

    net = models.Model(inputs=[input_layer], outputs=deco_out)
    return net


if __name__ == '__main__':
    net = saliency_net()
    print(net.count_params(), net.inputs, net.outputs)
