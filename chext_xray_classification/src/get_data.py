import os
import pandas as pd
import glob2
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(dataset_path):
    data_entry_csv_path = os.path.join(dataset_path, 'Data_Entry_2017.csv')
    data = pd.read_csv(data_entry_csv_path)
    print(f"Data Shape : {data.shape}")
    
    # Removing patients with age greater than 100
    data = data[data['Patient Age']<100]
    print(f"New dataset dimensions: {data.shape}")

    data = data[['Image Index', 'Finding Labels']]
    all_images = sorted(glob2.glob(dataset_path + '/**/*.png'))
    print(f'Number of Images: {len(all_images)}')

    all_image_paths = {os.path.basename(x): x for x in all_images}

    #Add path of images as column to the dataset
    data['Path'] = data['Image Index'].map(all_image_paths.get)
    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = np.delete(all_labels, np.where(all_labels == 'No Finding'))
    all_labels = [x for x in all_labels]

    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            # Add a column for each desease
            data[c_label] = data['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)
        
    print(f"Dataset Dimension: {data.shape}")

    data = data.groupby('Finding Labels').filter(lambda x : len(x)>11)
    #label_counts = data['Finding Labels'].value_counts()

    return data, all_labels

def split_train_dev_test(data):
    train_and_valid_df, test_df = train_test_split(data,
                                               test_size = 0.30,
                                               random_state = 2018,
                                              )

    train_df, valid_df = train_test_split(train_and_valid_df,
                                      test_size=0.30,
                                      random_state=2018,
                                     )
    print(f'Training: {train_df.shape[0]} Validation: {valid_df.shape[0]} Testing: {test_df.shape[0]}')

    return train_df, valid_df, test_df

def get_data_generator(dataframe, all_labels, batch_size = 32):
    IMG_SIZE = (224, 224)
    base_generator = ImageDataGenerator(rescale=1./255)
    df_gen = base_generator.flow_from_dataframe(dataframe,
                                                 x_col='Path',
                                                 y_col=all_labels,
                                                 target_size=IMG_SIZE,
                                                 classes=all_labels,
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 batch_size=batch_size)
    return df_gen
