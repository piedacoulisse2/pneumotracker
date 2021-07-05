# -*- coding: utf-8 -*-

import os
import cv2

from PIL import Image

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def build_df(path, img_ext):
    
    """Build DataFrame for model training/validation
        
       Parameter :
           - path : root path for images, which can be organized in sub-folders
           
       Returns DataFrame df with columns:
           - Label_name: "NORMAL" or "PNEUMONIA"
           - Label_int: 0 if "NORMAL", 1 if "PNEUMONIA"
           - Label_pathology: "normal", "bacteria", "virus"
           - Label_pathology_int: 0 if "normal", 1 if "bacteria", 2 if "virus"
           - Filename: filename with extension
           - Filpath: full path (directory + filename)"""
    
    list_df = []
    
    print('Build')
    
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if ('.' + img_ext) in filename:
                list_val = []
                list_val.append('PNEUMONIA' if 'PNEUMONIA' in dirname else 'NORMAL')
                list_val.append(1 if 'PNEUMONIA' in dirname else 0)
                list_val.append('bacteria' if 'bacteria' in filename.lower() else 'virus' if 'virus' in filename.lower() else 'normal')
                list_val.append(1 if 'bacteria' in filename.lower() else 2 if 'virus' in filename.lower() else 0)
                list_val.append(filename)
                list_val.append(os.path.join(dirname, filename))
                list_df.append(list_val)

 

    df = pd.DataFrame(list_df, columns = ['Label_name', 'Label_int', 'Label_pathology', 'Label_pathology_int', 'Filename', 'Filepath'])
    df.head()
    
    return df


def add_infos(df):
  
    rep = []
    height = []
    width = []
    size = []
    mode = []
    mean = []
    median = []
    std = []
    seuil100 = []
    
 
    for index, row in df.iterrows(): 
    
        full_path = row['Filepath']
    
        img = Image.open(full_path)
 
		#img = Image.grayscale(img)
 
        # Nom du fichier
        head, tail = os.path.split(full_path)
        
        # Répertoire 'NORMAL' OU 'PNEUMONIA'
        head, tail = os.path.split(head)
   
        # Répertoire 'train', 'test' ou 'val'
        head, tail = os.path.split(head)
        
        rep.append(tail)
 
        # Dimension de l'image
        height.append(img.height)
 
        width.append(img.width)
 
        # Size: nombre total de pixels
        img_size = img.height * img.width
        
        size.append(img_size)
 
        # Mode : couleur ou N&B
        mode.append(img.mode)
 
        # Conversion en array numpy            
        np_img = np.array(img)
        
        # Statistiques
        mean.append(round(np.mean(np_img), 2))
                
        median.append(round(np.median(np_img), 2))
                
        std.append(round(np.std(np_img), 2))
        
        # Seuillage
        img_seuil = np_img > 100
        
        up_to_seuil = round(img_seuil.sum() / img_size * 100, 2)
        
        seuil100.append(up_to_seuil)
 
        
    df['Rep'] = rep
    df['Height'] = height
    df['Width'] = width
    df['Size'] = size
    df['Mode'] = mode
    df['Mean'] = mean
    df['Median'] = median
    df['Std'] = std
    df['Seuil100'] = seuil100

 
    return df    



def planche_contact(list_images):

    f, ax = plt.subplots(2, 5, figsize=(30, 10))

    for i in range(10):

        img = cv2.imread(list_images[i])

        lig = i // 5
        col = i % 5

        ax[lig, col].imshow(img)

        if i < 5:
            ax[lig, col].set_title("Normal")
        else:
            ax[lig, col].set_title("Pneumonie")

        ax[lig, col].axis('off')
        ax[lig, col].set_aspect('auto')

    plt.show()


    
def show_image(image_path):

    img = cv2.imread(image_path)

    titre = os.path.split(image_path)[1]    

    fig = plt.figure(figsize=(10, 3))

    ax = fig.add_subplot(121) # ------------------------
    
    ax.set_title(titre)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])    
    
    ax.imshow(img)
    
    ax = fig.add_subplot(122)
    
    ax.hist(img.ravel(), bins = 256, alpha = 0.7)
    ax.set_title('Histogramme')
    
    bottom, top = plt.ylim() # --------------------------   
    
    ax.vlines(x = np.mean(img), 
              ymin = bottom, ymax = top,
              color = 'blue',
              label = 'Moyenne',
              linestyles  = 'dashed')
    
    ax.vlines(x = np.median(img), 
              ymin = bottom, ymax = top,
              label = 'Médiane',
              linestyles  = 'dashed')    

    ax.set_aspect('auto')    
    
    ax.legend()
    
    plt.show()  
    
