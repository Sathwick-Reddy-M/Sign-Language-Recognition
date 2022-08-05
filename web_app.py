from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def get_best_model():
    best_model = keras.models.load_model('models/experiment-dropout-0')
    return best_model

@st.cache
def get_label_binarizer():
    train_df = pd.read_csv('data/alphabet/sign_mnist_train.csv')
    y = train_df['label']
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    return label_binarizer

def preprocess_image(image, image_file, best_model, label_binarizer):
    # image: numpy array

    image_width = image.shape[0]

    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    image = image/255
    image = tf.image.resize(image, [28, 28], preserve_aspect_ratio=True)
    
    preprocessed_image = np.ones((1, 28, 28, 1))
    preprocessed_image[0, :image.shape[0], :image.shape[1], :] = image
    
    prediction = best_model.predict(preprocessed_image)
    
    index_to_letter_map = {i:chr(ord('a') + i) for i in range(26)}
    letter = index_to_letter_map[label_binarizer.inverse_transform(prediction)[0]]

    st.image(image_file, caption='Uploaded Image', width=max(image_width, 100))
    st.write(f'The image is predicted as {letter}')

best_model = get_best_model()
label_binarizer = get_label_binarizer()

image_file = st.file_uploader('Choose the ASL Image', ['jpg', 'png'])

if image_file is not None:
    image = Image.open(image_file).convert('L')
    image = np.array(image, dtype='float32')
    preprocess_image(image, image_file, best_model, label_binarizer)

test_df = pd.read_csv('data/alphabet/sign_mnist_test.csv')
X_test, y_test = test_df.drop('label', axis=1), test_df['label']

for i in range(10):
    plt.imsave(f'test-images/{i+1}.jpg', np.array(X_test.iloc[i]).reshape(28, 28))

index_to_letter_map = {i:chr(ord('a') + i) for i in range(26)}
