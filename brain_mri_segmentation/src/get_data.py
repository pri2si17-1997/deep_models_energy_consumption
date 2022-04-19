import os
import pandas as pd
import numpy as np
import glob2
import random
import cv2
from sklearn.model_selection import train_test_split

def has_mask(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0: 
        return 1
    else:
        return 0

DATASET_PATH = os.path.abspath('../dataset/lgg-mri-segmentation/kaggle_3m/')

def get_data():
    #data = pd.read_csv(os.path.join(DATASET_PATH + '/data.csv'))

    images = sorted(glob2.glob(DATASET_PATH + '/**/*.tif'))
    patient_id = [x.split('/')[-2] for x in images]

    df = pd.DataFrame(list(zip(patient_id, images)), columns=['patient_id', 'image_path'])

    df_imgs = df[~df['image_path'].str.contains("mask")] # if have not mask
    df_masks = df[df['image_path'].str.contains("mask")]# if have mask

    # File path line length images for later sorting
    BASE_LEN = len(DATASET_PATH + '/TCGA_DU_6408_19860521/TCGA_DU_6408_19860521_')
    END_IMG_LEN = 4
    END_MASK_LEN = 9

    # Data sorting
    imgs = sorted(df_imgs["image_path"].values, key=lambda x : int(x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(df_masks["image_path"].values, key=lambda x : int(x[BASE_LEN:-END_MASK_LEN]))

    # Sorting check
    idx = random.randint(0, len(imgs)-1)
    print("Path to the Image:", imgs[idx], "\nPath to the Mask:", masks[idx])

    brain_df = pd.DataFrame({"patient_id": df_imgs.patient_id.values,
                            "image_path": imgs,
                            "mask_path": masks
                            })
    
    brain_df['mask'] = brain_df['mask_path'].apply(lambda x: has_mask(x))
    brain_df_mask = brain_df[brain_df['mask'] == 1]

    X_train, X_val = train_test_split(brain_df_mask, test_size=0.15)
    X_test, X_val = train_test_split(X_val, test_size=0.5)

    print("Train size is {}, valid size is {} & test size is {}".format(len(X_train), len(X_val), len(X_test)))

    return X_train, X_val, X_test
