# Build dataframe for dataset
    
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

def get_and_create_dirs():
    
    """
Asks user for directories for original and segmented images, returns corresponding paths
   
Returns:
   - orig_path: path for original images
   - seg_path: path for segmented images
    """

    print('Checks for paths for original and segmented images.\nOriginal and segmented images must be organized in the exact same structure (images in the same folders and sub-folders).\n')
    
    orig_path = None

    while orig_path is None:
        orig_path = input('Input root path for original images:\n')
        if not os.path.exists(orig_path):
            orig_path = None
            print('Path doesn\'t exist, please input a valid directory path.\n')

    seg_path = input('\nInput root path for segmented images if exists or needs creation:\n')
    if not os.path.exists(seg_path):
        create_dir = 'Z'
        while create_dir not in ['Y', 'N']:
            create_dir = input('Path doesn\'t exist, would you like to create it (Y or N)?\n')
        if create_dir == 'Y':
            os.makedirs(seg_path)
            if os.path.exists(seg_path):
                print('Directory created.')
            else:
                print('Unknown error while attempting to create directory.')
        else:
            print('Directory not created')
            
    return orig_path, seg_path

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
    
    print('Building dataset DataFrame')
    
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
            for filename in filenames:
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
    
    return df
    
def build_train_model(X, Y, model, classes = 2, batch_size = 32, epochs = 20, checkpoint_name = 'checkpoint.h5', history_name = 'history'):
    
    """
Builds, trains and evaluates model. Saves training history in a DataFrame and model in a checkpoint h5 file :
    - Splits data into training, validation and testing set.
    - Builds data generators.
    - Computes class weights.
    - Builds callbacks.

Parameters:
    - X: pandas Series containing filepaths to images.
    - Y: pandas Series containing labels for images.
    - model: model to train and evaluate.
    - classes (Optional): number of classes, 2 by default (NORMAL & PNEUMONIA).
    - batch_size (Optional): batch_size for training, validation and testing datasets, 32 by default.
    - epochs (Optional): number of epochs for model training, 20 by default.
    - checkpoint_name (Optional): name for model training checkpoint, "checkpoint.h5" by default.
    - history_name (Optional): name for history DataFrame, "checkpoint.h5" by default.
    
Returns:
    - model evaluation
    - model training history
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

    model = load_model(checkpoint_name)

    print('\nEvaluating model')
    model_eval = model.evaluate(test_generator)

    print("Loss: " , model_eval[0])

    print("Accuracy: " , model_eval[1]*100 , "%")

    hist_df = pd.DataFrame(history.history)

    with open(history_name + '.csv', mode='w') as f:
        hist_df.to_csv(f)
    
    return model_eval, history
    

# Models
# Variables containing models used in Pneumotracker project

# model_smpl : simple model with 2 convolution layers and one dense

model_smpl = Sequential()
model_smpl.add(Conv2D(filters = 30,
                      kernel_size = (5, 5),
                      input_shape = (224, 224, 1),
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

# model_comp : complex model with 10 convolution layers and one dense

model_comp = Sequential()    
model_comp.add(Conv2D(filters = 16,
                      kernel_size = (3, 3),
                      input_shape = (224, 224, 3),
                      activation = 'relu'))
model_comp.add(Conv2D(filters = 16,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(AveragePooling2D(pool_size = (2, 2)))
model_comp.add(BatchNormalization())    
model_comp.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(AveragePooling2D(pool_size = (2, 2)))
model_comp.add(BatchNormalization())
model_comp.add(Conv2D(filters = 64,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(Conv2D(filters = 64,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(AveragePooling2D(pool_size = (2, 2)))
model_comp.add(BatchNormalization())
model_comp.add(Conv2D(filters = 128,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(Conv2D(filters = 128,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(AveragePooling2D(pool_size = (2, 2)))
model_comp.add(BatchNormalization())
model_comp.add(Conv2D(filters = 256,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(Conv2D(filters = 256,
                      kernel_size = (3, 3),
                      activation = 'relu'))
model_comp.add(AveragePooling2D(pool_size = (2, 2)))
model_comp.add(BatchNormalization())
model_comp.add(Flatten())
model_comp.add(Dropout(0.2))
model_comp.add(Dense(units = 380,
                     activation = 'relu'))
model_comp.add(Dropout(0.2))
model_comp.add(Dense(units = 1,
                     activation = 'sigmoid'))
optimizer = Adam()
model_comp.compile(optimizer = optimizer,
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])