# --------Les imports ----------
import streamlit as st
import pandas as pd
import cv2
import os
from keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D
import skimage
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np # linear algebra
import matplotlib.cm as cm
from skimage import io, transform
import skimage
from skimage.io import imread
import lime
from lime import lime_image, explanation
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from bokeh.plotting import figure
import keras
#-----Model de Deep Leaning------
@st.cache
def importerImages():
    labels = ['PNEUMONIA', 'NORMAL']
    img_size = 180
    def get_training_data(data_dir):
        data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)

    train = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/train')
    test = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/test')
    val = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/val')

    IMG_SIZE = 180
    x_train = []
    y_train = []

    x_val = []
    y_val = []

    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)



    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255


    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(y_test)

    x_global = np.concatenate((x_train,x_test),axis=0)
    y_global = np.concatenate((y_train,y_test),axis=0)



    X_train, X_test, Y_train, Y_test = train_test_split(x_global, y_global, test_size=0.33, random_state=42)
    return X_train, X_test, Y_train, Y_test

def generationModel(X_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Dropout(0.2))

    model.add(Activation("sigmoid"))
    metrics = [
      'accuracy',
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
    ]

def lancementModel(model,metrics,X_train, Y_train,X_test,Y_test):
    model.compile(loss='binary_crossentropy', # loss function
                  optimizer='adam',                # optimization algorithm
                  metrics=metrics)

    training_history = model.fit(X_train, Y_train,
                                 validation_split = 0.2,
                                 epochs = 25,
                                 batch_size = 200)

    scores = model.evaluate(X_test, Y_test)

def importerHistorique(adresse):
    r = pd.read_json(adresse)
    return r

def print_results(y_test, y_pred):
    try:

        liste = []
        liste.append(['Accuracy',accuracy_score(y_pred , y_test)])
        liste.append(['AUC',roc_auc_score(y_test , y_pred)])
        liste.append(['Precision',precision_score(y_test , y_pred)])
        liste.append(['Recall',recall_score(y_test , y_pred)])
        liste.append(['F1',f1_score(y_test , y_pred)])
        list_df = pd.DataFrame(liste)
        st.dataframe(list_df)

        #st.write('Accuracy   : {:.5f}'.format(accuracy_score(y_pred , y_test)))
        #st.write('AUC        : {:.5f}'.format(roc_auc_score(y_test , y_pred)))
        #st.write('Precision  : {:.5f}'.format(precision_score(y_test , y_pred)))
        #st.write('Recall     : {:.5f}'.format(recall_score(y_test , y_pred)))
        #st.write('F1         : {:.5f}'.format(f1_score(y_test , y_pred)))
        st.write('Confusion Matrix : \n', confusion_matrix(y_test, y_pred))


    except:
        pass

def printHistorique(history,epochs):

    epochs_array = [i for i in range(epochs)]
    fig, ax = plt.subplots(1, 3)
    train_precision = history['precision']
    train_recall = history['recall']
    train_loss = history['loss']

    val_precision = history['val_precision']
    val_recall = history['val_recall']
    val_loss = history['val_loss']
    fig.set_size_inches(20, 5)

    ax[0].plot(epochs_array, train_loss, 'g-o', label='Training Loss')
    ax[0].plot(epochs_array, val_loss, 'r-o', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)

    ax[1].plot(epochs_array, train_precision, 'go-', label='Training Precision')
    ax[1].plot(epochs_array, val_precision, 'ro-', label='Validation Precision')
    ax[1].set_title('Training & Validation Precision')
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Precision')
    ax[1].grid(True)

    ax[2].plot(epochs_array, train_recall, 'go-', label='Training Recall')
    ax[2].plot(epochs_array, val_recall, 'ro-', label='Validation Recall')
    ax[2].set_title('Training & Validation Recall')
    ax[2].legend()
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Recall')
    ax[2].grid(True)

    st.pyplot(fig)

def printAccuracy(history,epochs):
    epochs_array = [i for i in range(epochs)]
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    p = figure(
        title='Précision (acc) des jeux de train et de validation.',
        x_axis_label='Epochs',
        y_axis_label='Précision (acc)')
    p.line(epochs_array, train_acc, legend_label='Précision du train (acc)', line_width=2,color='green')
    p.line(epochs_array, val_acc, legend_label='Précision de la validation (acc)', line_width=2,color='red')

    p.circle(epochs_array, train_acc, line_width=2, color='green',fill_color="white")
    p.circle(epochs_array, val_acc, line_width=2, color='red',fill_color="white")

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"

    st.bokeh_chart(p)

def printImage_incorrect(y_test,y_pred,x_test,IMG_SIZE):
    st.write('Images des erreurs de predictions du modèle')
    incorrect = np.nonzero(y_test != y_pred)[0]
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.25, hspace=0.75)
    plt.tight_layout()
    i = 0
    labels = ['PNEUMONIA', 'NORMAL']

    for c in incorrect[:6]:
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(x_test[c].reshape(IMG_SIZE, IMG_SIZE), cmap='gray', interpolation='none')
        ax[i].set_title('Prédiction de Classe: {}, {} Véritable Classe: {}, {}'.format(y_pred[c],labels[int(y_pred[c])], y_test[c],labels[int(y_test[c])]))
        i += 1
    st.pyplot(fig)
def printImage_correct(y_test,y_pred,x_test,IMG_SIZE):
    st.write('Images des bonnes predictions du modèle')
    correct = np.nonzero(y_test == y_pred)[0]
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.25, hspace=0.75)
    plt.tight_layout()
    i = 0
    labels = ['PNEUMONIA', 'NORMAL']
    for c in correct[:6]:
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(x_test[c].reshape(IMG_SIZE, IMG_SIZE), cmap='gray', interpolation='none')
        ax[i].set_title('Prédiction de Classe: {}, {} Véritable Classe: {}, {}'.format(y_pred[c],labels[int(y_pred[c])], y_test[c],labels[int(y_test[c])]))
        i += 1
    st.pyplot(fig)

def importerImage():
    return 0

def detailModeleUtilise(model_pneumonie,IMG_SIZE):

    agree = st.checkbox('Afficher tous les détails du modèle (plus long)')
    #Lancement du modèle.
    epochs = 45
    #model = keras.models.load_model("model_pneumonie_1_KAGGLE.h5") Modèle Simple initial
    historique = importerHistorique("./Dfs/historique_model.json")
    #Streamlit affichage des résultats du modèle.
    printHistorique(historique,epochs)
    printAccuracy(historique, epochs)
    if agree:
        X_train, X_test, Y_train, Y_test = importerImages()
        predictions = model_pneumonie.predict(x=X_test)
        y_pred = np.round(predictions).reshape(1, -1)[0]
        print_results(Y_test, y_pred)
        printImage_incorrect(Y_test,y_pred,X_test,IMG_SIZE)
        printImage_correct(Y_test,y_pred,X_test,IMG_SIZE)

def pretraitementImage(uploaded_file_pred,color_on,ISIZE):
    file_bytes = np.asarray(bytearray(uploaded_file_pred), dtype=np.uint8)
    if color_on == True:
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    #st.image(opencv_image, caption='Image chargée origiale')
    imageResize = cv2.resize(opencv_image, (ISIZE, ISIZE))

    #st.image(imageResize, caption='Image chargée redimensionnée')
    l = []
    l.append(imageResize)

    #image255 = np.array(np.array(l)) / 255
    imagePredction = np.array(l).reshape(-1, ISIZE, ISIZE, 1)
    return imagePredction

def pretraitementImage224(uploaded_file_pred,color_on):
    file_bytes2 = np.asarray(bytearray(uploaded_file_pred), dtype=np.uint8)
    opencv_image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

    st.image(opencv_image2, caption='Image chargée origiale')
    imageResize = cv2.resize(opencv_image2, (224, 224))
    st.image(imageResize, caption='Image chargée redimensionnée')
    l = []
    l.append(imageResize)
    image255 = np.array(np.array(l)) / 255
    #imagePredction = image255.reshape(-1,224, 224)
    imagePredction = tf.reshape(image255,(1, 224, 224,3))
    return imagePredction

def predictionModel(imagePredction,model_pred):

    predictions = model_pred.predict(x=imagePredction)

    return predictions


# Fonctions Grad-Cam
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).
                                       output,
                                        model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.1):
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


# Fonctions Lime
def read_and_transform_img(path,IMG_SIZE):
    img = skimage.io.imread(path)
    img = skimage.transform.resize(img, (IMG_SIZE[0], IMG_SIZE[0], 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img
def afficherGramCAM(uploaded_file2,model_pneumonie_2_KAGGLE_bis,dernierecouche,segnmente_ON_OFF):

    if segnmente_ON_OFF:
        # Grad_Cam
        # uploaded_file2 = cv2.resize(uploaded_file2, (1,512, 512,1))
        # plt.subplot(5, 3, 10)
        # heatmap_orig = make_gradcam_heatmap(uploaded_file2, model_pneumonie_2_KAGGLE_bis,dernierecouche
        #                               )
        #
        #file_bytes2 = np.asarray(bytearray(uploaded_file2), dtype=np.uint8)
        #opencv_image2 = cv2.imdecode(uploaded_file2, cv2.IMREAD_GRAYSCALE)

        img_array = cv2.resize(uploaded_file2, (model_pneumonie_2_KAGGLE_bis.input.shape[1], model_pneumonie_2_KAGGLE_bis.input.shape[2]))
        img_array = np.expand_dims(img_array, axis=0)

        plt.subplot(5, 3, 10)

        heatmap_orig = make_gradcam_heatmap(tf.reshape(img_array, (1, 512, 512, 3)), model_pneumonie_2_KAGGLE_bis,
                                            dernierecouche
                                            )


        img_orig_grad = save_and_display_gradcam(uploaded_file2, heatmap_orig, alpha=0.4)
        st.image(img_orig_grad)
    else:
        # Grad_Cam
        file_bytes2 = np.asarray(bytearray(uploaded_file2), dtype=np.uint8)
        opencv_image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
        imageResize = cv2.resize(opencv_image2, (224, 224))

        plt.subplot(5, 3, 10)

        heatmap_orig = make_gradcam_heatmap(tf.reshape(imageResize, (1, 224, 224, 3)), model_pneumonie_2_KAGGLE_bis,
                                            dernierecouche
                                            )
        img_orig_grad = save_and_display_gradcam(imageResize, heatmap_orig, alpha=0.4)
        st.image(img_orig_grad)


# def read_and_transform_img(path,IMG_ SIZE):
#     img = skimage.io.imread(path)
#     img = skimage.transform.resize(img, (IMG_SIZE[0], IMG_SIZE[0], 3))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     return img


def graph1():
    from skimage.segmentation import mark_boundaries

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=True,
                                                    num_features=1,  # 5
                                                    hide_rest=True)

    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=False,
                                                    num_features=1,  # 10
                                                    hide_rest=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax2.imshow(mark_boundaries(temp_2, mask_2))

    ax1.axis('off')
    ax2.axis('off')


def graph1_1():
    from skimage.segmentation import mark_boundaries

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=False,
                                                    num_features=1,  # 10
                                                    hide_rest=False)

    plt.imshow(mark_boundaries(temp_1, mask_1))

    plt.axis('off')


def graph2():
    ind = explanation.top_labels[0]

    dict_heatmap = dict(explanation.local_exp[ind])

    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())

    plt.colorbar()




# Functions for image segmentation
# Main function is segment_image
# Other functions are used by segment_image

# Utility functions for segmentation

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
def image_to_train(img):
    npy = img / 255
    npy = np.reshape(npy, npy.shape + (1,))
    npy = np.reshape(npy, (1,) + npy.shape)
    return npy
def train_to_image(npy):
    img = (npy[0, :, :, 0] * 255.).astype(np.uint8)
    return img
# Main function for segmenting image

def segmentation_image(segmentation_model, uploaded_file2,seg_img_ext, save_to = None):
    """
Segment image using segmentation_model: extract lungs from image

Parameters:
   - segmentation_model: trained model used for image segmentation
   - img_path: path of image to segment
   - save_to (Optional): destination path for segmented_image. Image not saved if argument is not passed.

Returns:
   - segmented image
    """
    #pid, fileext = os.path.splitext(os.path.basename(img_path))
    file_bytes2 = np.asarray(bytearray(uploaded_file2), dtype=np.uint8)
    opencv_image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(opencv_image2, (512, 512))


    segm_ret = segmentation_model.predict(image_to_train(img),
                                          verbose=0)
    img = cv2.bitwise_and(img,
                          img,
                          mask=train_to_image(segm_ret))
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img_color

