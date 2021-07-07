# Build dataframe for dataset

import os
import pandas as pd
import numpy as np
import cv2

from scipy.stats import chi2_contingency, shapiro, mannwhitneyu


# Build DataFrame of dataset information

def build_stats_df(df, filepath_col, labelname_col, labelpat_col, save_path = '.\df_stats.csv'):
    
    """
Build or reads DataFrame for dataset statistics computation and saves to "save_path" if passed
        
Parameter :
   - df: DataFrame containing paths for images and classes of images
   - filepath_col: name of column containing paths to images
   - labelname_col: name of column containing pathology (NORMAL or PNEUMONIA)
   - labelpat_col: name of column containing type of pathogen
   - save_path (Optional): path to read or save DataFrame
   
Returns DataFrame df with columns:
   - Pathology: "NORMAL" or "PNEUMONIA"
   - Pathogen: "normal", "bacteria", "virus"
   - Mean: mean pixel for image
   - Filename_orig: filename of original image with extension
   - Filpath_orig: full path of original image (directory + filename)
   - Filename_seg: filename of segmented image with extension
   - Filepath_seg: full path of segmented image (directory + filename)
    """
       
    read_df = 'C'
    list_df = []

    means = {'NORMAL' : 0,
             'PNEUMONIA' : 0,
             'normal' : 0,
             'bacteria' : 0,
             'virus' : 0}
    
    if os.path.exists(save_path):
        read_df = input('DataFrame was found, would you like to read it (R) or recreate it (C) (default Read)?\n') or 'R'
        if read_df == 'R':
            df = pd.read_csv(save_path, index_col = 0)
            new_cols = ['Pathology', 'Pathogen', 'Mean', 'Nb_pixels', 'Sup_thr_100', 'Per_thr_100'] + list(np.arange(0, 256, 1))
            df.columns = new_cols
            return df
    
    if read_df == 'C':
        for index, row in df.iterrows():
            img = cv2.imread(row[filepath_col], cv2.IMREAD_GRAYSCALE)
            list_mean = [row[labelname_col],
                         row[labelpat_col],
                         img.mean(),
                         img.shape[0] * img.shape[1],
                         img[img > 100].shape[0],
                         img[img > 100].shape[0] / (img.shape[0] * img.shape[1])] + list(np.unique(np.array(img), return_counts=True)[1])
            list_df.append(list_mean)
            
        df = pd.DataFrame(list_df, columns = ['Pathology', 'Pathogen', 'Mean', 'Nb_pixels', 'Sup_thr_100', 'Per_thr_100'] + list(np.arange(0, 256, 1))).fillna(0)
        df.to_csv(save_path)
        
    print('Done')
    
    return df


# Compare populations mean or distribution

def compare_pops(dataset1, dataset2, label1, label2, alpha = 0.05):
    
    """
Compares datasets with T-test or Mann-Whitney test depending on datasets normality.
        
Parameter :
   - dataset1: numpy array of first test data
   - dataset2: numpy array of second test data
   - label: label of first test data
   - label: label of second test data
   - alpha (Optional): threshold for test p-value. Default = 0.05
    """
       
    shapiro_data1 = shapiro(dataset1)
    shapiro_data2 = shapiro(dataset2)
    normal = True
    
    print('1. Are populations normally distributed?')
    for i, j in zip([shapiro_data1, shapiro_data2], [label1, label2]):
        if i.pvalue > alpha:
            print('H0 (normal distribution) not rejected (p-value = {:.5f}).\nDataset "{}" distribution could be normal.\n'.format(i.pvalue, j))
            normal = True if normal else False
        else:
            print('H0 (normal distribution) rejected (p-value = {:.5f}).\nDataset "{}" distribution does not appear normal.\n'.format(i.pvalue, j))
            normal = False
            
    if normal:
        print('2. Do populations have equal mean?')
        ttest_result = ttest_ind(dataset1, dataset2)
        if ttest_result.pvalue > alpha:
            print('H0 (equal population means) not rejected (p-value = {:.5f}).\n{} and {} populations appear to have equal means.\n'.format(ttest_result.pvalue, label1, label2))
        else:
            print('H0 (equal population means) rejected (p-value = {:.5f}).\n{} and {} populations appear to have different means.\n'.format(ttest_result.pvalue, label1, label2))
    else:
        print('2. Do populations have similar distributions?')
        mannwhit_test = mannwhitneyu(dataset1, dataset2)
        if mannwhit_test.pvalue > alpha:
            print('H0 (identical population distributions) not rejected (p-value = {:.5f}).\n{} and {} populations appear to have similar distributions.\n'.format(mannwhit_test.pvalue, label1, label2))
        else:
            print('H0 (identical population distributions) rejected (p-value = {:.5f}).\n{} and {} populations appear to have different distributions.\n'.format(mannwhit_test.pvalue, label1, label2))


# Assess variable independence

def chi_test(cont_table, variable1_name, variable2_name, N, alpha = 0.05):
    
    """
Computes chi-squared test to assess independence of variables
        
Parameter :
   - cont_table: contingency table (DataFrame)
   - variable1_name: name of first variable
   - variable2_name: name of second variable
   - alpha (Optional): threshold for test p-value. Default = 0.05
    """
    
    chi2 = chi2_contingency(cont_table, correction=False)
    
    if chi2[1] > alpha:
        print('H0 (variables are independant) not rejected (p-value = {:.5f}).\n{} and {} variables appear to be dependant.\n'.format(chi2[1], variable1_name, variable2_name))
    else:
        print('H0 (variables are independant) rejected (p-value = {:.5f}).\n{} and {} variables appear to be independant.\n'.format(chi2[1], variable1_name, variable2_name))
    
    k = cont_table.shape[0]
    r = cont_table.shape[1]
    phi = max(0,(chi2[0] / N)-((k-1)*(r-1)/(N-1)))
    k_corr = k - (np.square(k-1)/(N-1))
    r_corr = r - (np.square(r-1)/(N-1)) 
    cramer_v = np.sqrt(phi/min(k_corr - 1,r_corr - 1))

    print('Cramer\'s V value is {:.2f}:'.format(cramer_v))
    
    if cramer_v < 0.1:
        print('There is a very weak link between "{}" and "{}".'.format(variable1_name, variable2_name))
    elif cramer_v < 0.2:
        print('There is a weak link between "{}" and "{}".'.format(variable1_name, variable2_name))
    elif cramer_v < 0.5:
        print('There is a moderate link between "{}" and "{}".'.format(variable1_name, variable2_name))
    elif cramer_v < 0.2:
        print('There is a very strong link between "{}" and "{}".'.format(variable1_name, variable2_name))