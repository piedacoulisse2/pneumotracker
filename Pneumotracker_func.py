# Build dataframe for dataset

import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.applications.imagenet_utils import decode_predictions

import skimage
from skimage.segmentation import mark_boundaries
from skimage import io, transform
from skimage.io import imread

import lime
from lime import lime_image

def get_and_create_dirs():
    
    """
Asks user for directories for original and segmented images, returns corresponding paths
   
Returns:
   - orig_path: path for original images
   - seg_path: path for segmented images
    """

    print('Checks for paths for original and segmented images.\nOriginal and segmented images folders must be organized in the exact same structure (images in the same folders and sub-folders).\n')
    
    orig_path = None

    while orig_path is None:
        orig_path = input('Input root path for original images (default github folder):\n') or r'.\chest_xray'
        if not os.path.exists(orig_path):
            orig_path = None
            print('Path doesn\'t exist, please input a valid directory path.\n')

    seg_path = input('\nInput root path for segmented images if exists or needs to be created (default github folder):\n') or r'.\segmentation'
    if (seg_path is not None) and (not os.path.exists(seg_path)):
        create_dir = 'Z'
        while create_dir not in ['Y', 'N']:
            create_dir = input('Path doesn\'t exist, would you like to create folder structure for ' + seg_path + ' (Y or N)?\n')
        if create_dir == 'Y':
            for dirname, _, filenames in os.walk(orig_path):
                os.makedirs(dirname.replace(orig_path, seg_path))           
            if os.path.exists(seg_path):
                print('Directory created.')
            else:
                print('Unknown error while attempting to create directory.')
        else:
            print('Directory not created')
    
    orig_file_ext = input('\nWhat is the file extension for original images (default jpeg) ?\n').replace('.', '') or 'jpeg'
    
    seg_model = input('\nWhat is the path to the segmentation model checkpoint (default github folder) ?\n') or r'.\Models\unet_lung_seg.hdf5'
            
    return orig_path, seg_path, orig_file_ext, seg_model

def build_df(path_orig, orig_img_ext, path_seg = None, seg_img_ext = None, df = None):
    
    """
Build DataFrame for model training/validation
        
Parameter :
   - path_orig: root path for original images, which can be organized in sub-folders
   - orig_img_ext: extension of original images files
   - path_seg (Optional): root path for original images, which can be organized in sub-folders
   - seg_img_ext (Optional): root path for original images, which can be organized in sub-folders
   - df (Optional): if passed, Filename_seg and Filepath_seg will be added
   
Returns DataFrame df with columns:
   - Label_name: "NORMAL" or "PNEUMONIA"
   - Label_int: 0 if "NORMAL", 1 if "PNEUMONIA"
   - Label_pathology: "normal", "bacteria", "virus"
   - Label_pathology_int: 0 if "normal", 1 if "bacteria", 2 if "virus"
   - Filename_orig: filename of original image with extension
   - Filpath_orig: full path of original image (directory + filename)
   - Filename_seg: filename of segmented image with extension
   - Filepath_seg: full path of segmented image (directory + filename)
    """
    
    list_df = []
    
    if df is not None:
        if (path_seg is None) or (seg_img_ext is None):
            print('Arguments path_seg and seg_img_ext are mandatory if df is passed.')
            print('Execute function with mandatory parameters.')
            return df
        else:
            df['Filename_seg'] = df['Filename_orig'].apply(lambda x: x.replace(orig_img_ext, seg_img_ext))
            df['Filepath_seg'] = df['Filepath_orig'].apply(lambda x: x.replace(path_orig, path_seg).replace(orig_img_ext, seg_img_ext))
    else:
        for dirname, _, filenames in os.walk(path_orig):
            for filename in tqdm(filenames, disable=len(filenames)==0):
                if ('.' + orig_img_ext) in filename:
                    list_val = []
                    list_val.append('PNEUMONIA' if 'PNEUMONIA' in dirname else 'NORMAL')
                    list_val.append(1 if 'PNEUMONIA' in dirname else 0)
                    list_val.append('bacteria' if 'bacteria' in filename.lower() else 'virus' if 'virus' in filename.lower() else 'normal')
                    list_val.append(1 if 'bacteria' in filename.lower() else 2 if 'virus' in filename.lower() else 0)
                    list_val.append(filename)
                    list_val.append(os.path.join(dirname, filename))
                    
                    if (path_seg is not None) and (seg_img_ext is not None):
                        list_val.append(filename.replace(orig_img_ext, seg_img_ext))
                        list_val.append(os.path.join(dirname.replace(path_orig, path_seg), filename.replace(orig_img_ext, seg_img_ext)))
                    else:
                        list_val += ['', '']
                    
                    list_df.append(list_val)

        df = pd.DataFrame(list_df, columns = ['Label_name', 'Label_int', 'Label_pathology', 'Label_pathology_int', 'Filename_orig', 'Filepath_orig', 'Filename_seg', 'Filepath_seg'])
        
    print('Done')
    
    return df
    
def build_train_model(X, Y, model, classes = 2, batch_size = 32, epochs = 20, checkpoint_name = 'checkpoint.h5', history_name = 'history', train_model = True):
    
    """
Builds, trains and evaluates model. Saves training history in a DataFrame and model in a checkpoint h5 file :
    - Splits data into training, validation and testing set.
    - Builds data generators.
    - Computes class weights.
    - Builds callbacks.

Parameters:
    - X: pandas Series containing filepaths to images
    - Y: pandas Series containing labels for images
    - model: model to train and evaluate
    - classes (Optional): number of classes, 2 by default (NORMAL & PNEUMONIA)
    - batch_size (Optional): batch_size for training, validation and testing datasets, 32 by default
    - epochs (Optional): number of epochs for model training, 20 by default
    - checkpoint_name (Optional): name for model training checkpoint, "checkpoint.h5" by default
    - history_name (Optional): name for history DataFrame, "checkpoint.h5" by default.
    - train_model : model is trained if True, else simply loaded from checkpoint and metrics are computed
    
Returns:
    - model evaluation
    - model training history
    - classification_report
    - confusion matrix
    - model
    """

    print('\nSplitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.30, random_state = 1234)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify = y_test, test_size = 0.50, random_state = 1234)

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_test = pd.concat([X_test, y_test], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)
    
    print('\nBuilding Data Generators')
    image_size = (model.input.shape[1], model.input.shape[2])
    
    if model.input.shape[3] == 3:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'
        
    if classes == 2:
        class_mode = 'binary'
        classes_list = ['NORMAL', 'PNEUMONIA']
        loss_function = 'binary_crossentropy'
    else:
        class_mode = 'categorical'
        classes_list = ['normal', 'bacteria', 'virus']
        loss_function = 'categorical_crossentropy'
    
    gen = ImageDataGenerator(rescale = 1./255)

    batch_size = batch_size

    train_generator = gen.flow_from_dataframe(df_train,
                                              directory = None,
                                              x_col = X.name,
                                              y_col = Y.name,
                                              target_size = image_size,
                                              color_mode = color_mode,
                                              batch_size = batch_size,
                                              class_mode = class_mode,
                                              shuffle = True,
                                              seed = 42)

    valid_generator = gen.flow_from_dataframe(df_test,
                                              directory = None,
                                              x_col = X.name,
                                              y_col = Y.name,
                                              target_size = image_size,
                                              color_mode = color_mode,
                                              batch_size = batch_size,
                                              class_mode = class_mode,
                                              shuffle = False)

    test_generator = gen.flow_from_dataframe(df_val,
                                             directory = None,
                                             x_col = X.name,
                                             y_col = Y.name,
                                             target_size = image_size,
                                             color_mode = color_mode,
                                             batch_size = batch_size,
                                             class_mode = class_mode,
                                             shuffle = False)
    
    if train_model:
    
        print('\nComputing Class weights')
        train_count = df_train.shape[0]
        class_weight = {}
        for i in range(classes):
            class_weight[i] = (1 / df_train[df_train[Y.name] == classes_list[i]][Y.name].count())*(train_count)/float(classes)
            print('Weight for class', classes_list[i], ': {:.2f}'.format(class_weight[i]))

        print('\nBuilding callbacks:')
        print('Checkpoint')
        checkpoint = ModelCheckpoint(checkpoint_name,
                                     monitor="loss",
                                     verbose=2,
                                     save_best_only=True,
                                     save_weights_only=False)

        print('Reduce learning rate on plateau')
        lr_plateau = ReduceLROnPlateau(monitor = 'loss',
                                       patience = 2,
                                       verbose = 2,
                                       mode = 'min')

        print('Early stopping for val_loss')
        early_stop = EarlyStopping(monitor = 'val_loss',
                                   patience = 5,
                                   verbose = 2,
                                   mode = 'min')
        
        print('\nBuilding model')
        optimizer = Adam()
        model.compile(optimizer = optimizer,
                      loss = loss_function,
                      metrics = ['accuracy'])
        model.summary()

        print('\nTraining model')
        history = model.fit(train_generator,
                            epochs = epochs,
                            steps_per_epoch = train_generator.n//batch_size,
                            validation_data = valid_generator,
                            validation_steps = valid_generator.n//batch_size,
                            callbacks = [checkpoint, lr_plateau, early_stop],
                            class_weight = class_weight)
    
        hist_df = pd.DataFrame(history.history)

        with open(history_name, mode='w') as f:
            hist_df.to_csv(f)

    if os.path.exists(history_name):
        history = pd.read_csv(history_name, index_col = 0)
    else:
        print('history doesn\'t exist')
        history = None
    
    model = load_model(checkpoint_name)

    print('\nEvaluating model')
    model_eval = model.evaluate(test_generator)

    print("Loss: " , model_eval[0])

    print("Accuracy: " , model_eval[1]*100 , "%")
    
    df_val['predicted_value'] = model.predict(test_generator)
    df_val['predicted_int'] = (df_val['predicted_value'] > 0.5).apply(int)
    df_val['predicted_str'] = df_val['predicted_int'].apply(lambda x: 'NORMAL' if x == 0 else 'PNEUMONIA')
    
    class_report = pd.DataFrame(classification_report(df_val['Label_name'], df_val['predicted_str'], output_dict = True))
    class_report = class_report.style.set_table_attributes("style='display:inline; font-size:110%; color:black; font-weight: bold'").set_caption('Classification report for model ' + checkpoint_name.replace('.h5', ''))
    
    conf_matrix = pd.crosstab(df_val['Label_name'], df_val['predicted_str'], rownames=['Real'], colnames=['Predicted'])
    conf_matrix = conf_matrix.style.set_table_attributes("style='display:inline; font-size:110%; color:black; font-weight: bold'").set_caption('Confusion matrix for model' + checkpoint_name.replace('.h5', ''))
    
    return model_eval, history, class_report, conf_matrix, model

    
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
    npy = np.reshape(npy,(1,) + npy.shape)
    return npy

def train_to_image(npy):
    img = (npy[0,:, :, 0] * 255.).astype(np.uint8)
    return img
    

# Main function for segmenting image

def segment_image(segmentation_model, img_path, save_to = None):
    
    """
Segment image using segmentation_model: extract lungs from image
        
Parameters:
   - segmentation_model: trained model used for image segmentation 
   - img_path: path of image to segment
   - save_to (Optional): destination path for segmented_image. Image not saved if argument is not passed.
   
Returns:
   - segmented image
    """

    pid, fileext = os.path.splitext(os.path.basename(img_path))
    img = cv2.resize(cv2.imread(img_path,
                                cv2.IMREAD_GRAYSCALE),
                     (512, 512))
    segm_ret = segmentation_model.predict(image_to_train(img),
                                          verbose=0)
    img = cv2.bitwise_and(img,
                          img,
                          mask=train_to_image(segm_ret))
    if save_to is not None:
        cv2.imwrite(os.path.join(save_to,
                                 "%s.png" % pid),
                    img)
    return img
    

# Intrpretability functions
# Grad-Cam

def get_heatmap_gradcam(model, last_conv_layer_name, img_path = None, img = None, heatmap_quant = None, alpha = 0.4):
    
    """
Computes and returns Grad-Cam heatmap and superimposed image:
    - Deactivates last convolution layer
    - Computes gradcam heatmap
    - Superimposes original image and heatmap
    - Reactivates last convolution layer
        
Parameters:
   - model: model for Grad-Cam
   - last_conv_layer_name: name of last convolution layer from model
   - img_path (Optional): path of image to interpret. Optional if img is passed
   - img (Optional): image to interpret. Optional if img_path is passed
   - heatmap_quant (Optional): if passed, quantile of heatmap pixel intesity to keep (between 0 and 1)
   - alpha (Optional): opacity of superimposed heatmap
   
Returns:
   - Superimposed image
   - Heatmap   

Example 1:
    heatmap, gradcam = get_heatmap_gradcam(model = random_model,
                                           last_conv_layer_name = 'conv2d_10',
                                           img_path = r'./images/image_1.jpeg',
                                           alpha = 0.4)
    
    Returns heatmap and all pixels of superimposed image from image "image_1.jpeg" located in "./images" with model "random_model", which last convolution layer is "conv2d_10"

Example 2:
    heatmap, gradcam = get_heatmap_gradcam(model = random_model,
                                           last_conv_layer_name = 'conv2d_10',
                                           img_path = r'./images/image_1.jpeg',
                                           heatmap_quant = 0.75
                                           alpha = 0.4)
    
    Returns heatmap and top 25% of pixels of superimposed image from image "image_1.jpeg" located in "./images" with model "random_model", which last convolution layer is "conv2d_10"
    """
    
    if (img_path is None) and (img is None):
        print('One of "img_path" or "img" is required')
        return None, None
    elif img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
    img_array = cv2.resize(img, (model.input.shape[1], model.input.shape[2]))        
    img_array = np.expand_dims(img_array, axis=0)
    
    model_activation = model.layers[-1].activation
    model.layers[-1].activation = None
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    if heatmap_quant is not None:
        for i in range(3):
            jet_heatmap[:, :, i] = np.where(jet_heatmap[:, :, i] > np.quantile(jet_heatmap[:, :, i], heatmap_quant), jet_heatmap[:, :, i], 0)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    model.layers[-1].activation = model_activation

    return heatmap, superimposed_img
    
# Lime

def lime_heatmap(model, img_path = None, img = None, colorbar = True, explanation = None):
    
    """
Diplays lime heatmap for model and image passed

Parameters:
    - model: model for Lime computations
    - img_path (Optional): path of image to interpret. Optional if img is passed
    - img (Optional): image to interpret. Optional if img_path is passed
    - colorbar (Optional): if True, colorbar is displayed next to heatmap
    - explanation (Optional): used if passed, computed otherwise
    
Returns:
    - Lime explanation
    
Example 1:

    explanation1 = lime_heatmap(model = random_model,
                                img_path = r'./images/image_1.jpeg',
                                explanation = None)
    Computes Lime explanation, displays lime heatmap with colorbar and returns explanation
    
Example 1:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    explanation1 = lime_heatmap(model = random_model,
                                img = img,
                                explanation = explanation1)
    Uses explanation1 to diplay  lime heatmap with colorbar and returns explanation1
    """
    
    if (img_path is None) and (img is None):
        print('One of "img_path" or "img" is required')
        return None, None
    elif img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (model.input.shape[1], model.input.shape[2]))
    
    img = np.expand_dims(img, axis=0)
    
    if explanation is None:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img[0].astype('double'), 
                                                 model.predict,  
                                                 top_labels=3, 
                                                 hide_color=0, 
                                                 num_samples=1000)
                                                 
    ind =  explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                    positive_only=False, 
                                                    num_features=1,
                                                    hide_rest=False)
                                                    
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    
    if colorbar:
        plt.colorbar()
    
    return explanation

def lime_outline(model, img_path = None, img = None, explanation = None):

    """
Diplays image superimposed with outline of Lime top label

Parameters:
    - model: model for Lime computations
    - img_path (Optional): path of image to interpret. Optional if img is passed
    - img (Optional): image to interpret. Optional if img_path is passed
    - explanation (Optional): used if passed, computed otherwise
    
Returns:
    - Lime explanation
    
Example 1:

    explanation = lime_outline(model = random_model,
                               img_path = r'./images/image_1.jpeg',
                               explanation = None)
    Computes Lime explanation, displays superimposed image and returns explanation
    
Example 2:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    explanation1 = lime_outline(model = random_model,
                                img = img,
                                explanation = explanation1)
    Uses explanation1 to diplay superimposed image and returns explanation1
    """
    
    from skimage.segmentation import mark_boundaries
    if (img_path is None) and (img is None):
        print('One of "img_path" or "img" is required')
        return None, None
    elif img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    img = skimage.transform.resize(img, (model.input.shape[1], model.input.shape[2], 3))    
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    if explanation is None:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img[0].astype('double'), 
                                                 model.predict,  
                                                 top_labels=3, 
                                                 hide_color=0, 
                                                 num_samples=1000)

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                    positive_only=False, 
                                                    num_features=1,
                                                    hide_rest=False)
                                                    
    plt.imshow(mark_boundaries(temp_1, mask_1))

    plt.axis('off')
    
    return explanation


# Models
# Variables containing models used in Pneumotracker project

# model_smpl : simple model with 2 convolution layers and one dense

model_smpl = Sequential()
model_smpl.add(Conv2D(filters = 30,
                      kernel_size = (5, 5),
                      input_shape = (224, 224, 3),
                      activation = 'relu',
                      padding = 'valid'))
model_smpl.add(MaxPooling2D(pool_size = (2, 2)))
model_smpl.add(Conv2D(filters = 16,
                      kernel_size = (3, 3),
                      activation = 'relu',
                      padding = 'valid'))
model_smpl.add(MaxPooling2D(pool_size = (2, 2)))
model_smpl.add(Dropout(rate = 0.2))
model_smpl.add(Flatten())
model_smpl.add(Dense(units = 128,
                     activation = 'relu'))
model_smpl.add(Dense(units = 1,
                     activation = 'sigmoid'))
optimizer = Adam()
model_smpl.compile(optimizer = optimizer,
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# model_cmpl : complex model with 10 convolution layers and one dense

model_cmpl = Sequential()    
model_cmpl.add(Conv2D(filters = 16,
                      kernel_size = (3, 3),
                      input_shape = (224, 224, 3),
                      activation = 'relu'))
model_cmpl.add(Conv2D(filters = 16,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(AveragePooling2D(pool_size = (2, 2)))
model_cmpl.add(BatchNormalization())    
model_cmpl.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(AveragePooling2D(pool_size = (2, 2)))
model_cmpl.add(BatchNormalization())
model_cmpl.add(Conv2D(filters = 64,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(Conv2D(filters = 64,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(AveragePooling2D(pool_size = (2, 2)))
model_cmpl.add(BatchNormalization())
model_cmpl.add(Conv2D(filters = 128,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(Conv2D(filters = 128,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(AveragePooling2D(pool_size = (2, 2)))
model_cmpl.add(BatchNormalization())
model_cmpl.add(Conv2D(filters = 256,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(Conv2D(filters = 256,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_cmpl.add(AveragePooling2D(pool_size = (2, 2)))
model_cmpl.add(BatchNormalization())
model_cmpl.add(Flatten())
model_cmpl.add(Dropout(0.2))
model_cmpl.add(Dense(units = 380,
                     activation = 'relu'))
model_cmpl.add(Dropout(0.2))
model_cmpl.add(Dense(units = 1,
                     activation = 'sigmoid'))
optimizer = Adam()
model_cmpl.compile(optimizer = optimizer,
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])