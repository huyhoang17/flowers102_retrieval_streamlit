import itertools

import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import streamlit as st
from utils import grpc_infer, load_prec_embs


def main():

    st.title("Flower retrieval")
    train_img_fps, train_embs, train_labels = load_prec_embs()
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption='Uploaded Image.',
            use_column_width=True
        )
        image = Image.open(uploaded_file)
        img_arr = np.array(image)

        # query emb
        test_emb = grpc_infer(img_arr)

        dists = cdist(test_emb, train_embs, metric='euclidean')[0]
        top_k = 18
        min_dist_indexes = dists.argsort()[:top_k]
        label_indexes = [train_labels[index] + 1 for index in min_dist_indexes]
        img_fps = [train_img_fps[index] for index in min_dist_indexes]

        indices_on_page, images_on_page = \
            map(list, zip(*itertools.islice(zip(label_indexes, img_fps), 0, top_k)))  # noqa
        st.image(images_on_page, width=200, caption=indices_on_page)


if __name__ == '__main__':
    main()
