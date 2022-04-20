from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
import numpy as np
from itertools import chain


def check_dict_subset(subset, superset):
    """Checks if one nested dictionary is a subset of another

    :param subset: subset dictionary
    :param superset: superset dictionary
    :return: if failed: gives helpful print statements and assertion error
             if successful, prints 'Your parameter choice is valid'
    """
    print("superset keys:", superset.keys())
    print("subset keys:", subset.keys())
    assert all(item in superset.keys() for item in subset.keys())
    print("Subset keys is a subset of superset keys", all(item in superset.keys() for item in subset.keys()))
    for key in subset.keys():
        print("superset key items:", superset[key])
        print("subset key items:", subset[key])
        if type(superset[key]) == dict:
            assert type(subset[key]) == type(superset[key])
            check_dict_subset(subset[key], superset[key])
        elif type(superset[key]) == list:
            assert subset[key] in superset[key]
            print("subset[key] item:", subset[key], " is in superset[key] items:", superset[key])
        else:
            print("Something went wrong. Uncomment the print statements in check_dict_subset() for easier debugging.")
            return type(superset[key]), superset[key]

    return 'Your parameter choice is valid'


def preprocess(imgs, img_rows, img_cols, to_rows=None, to_cols=None):
    """Resize all images in a 4D tensor of images of the shape (samples, rows, cols, channels).

    :param imgs: a 4D tensor of images of the shape (samples, rows, cols, channels)
    :param to_rows: new number of rows for images to be resized to
    :param to_cols: new number of rows for images to be resized to
    :return: a 4D tensor of images of the shape (samples, to_rows, to_cols, channels)
    """
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (to_rows, to_cols), preserve_range=True)
    return imgs_p

def prep(img, image_rows, image_cols):
    """Prepare the image for to be used in a submission

    :param img: 2D image
    :return: resized version of an image
    """
    img = img.astype('float32')
    img = resize(img, (image_rows, image_cols), preserve_range=True)
    img = (img > 0.5).astype(np.uint8)  # threshold
    return img


def run_length_enc(label):
    """Create a run-length-encoding of an image

    :param label: image to be encoded
    :return: string with run-length-encoding of an image
    """
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]

    # consider empty all masks with less than 10 pixels being greater than 0
    if len(y) < 10:
        return ''

    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

# Only change ALLOWED_PARS if adding new functionality
ALLOWED_PARS = {
    'outputs': [1, 2],
    'activation': ['elu', 'relu'],
    'pooling_block': {
        'trainable': [True, False]},
    'information_block': {
        'inception': {
            'v1': ['a', 'b'],
            'v2': ['a', 'b', 'c'],
            'et': ['a', 'b']},
        'convolution': {
            'simple': ['not_normalized', 'normalized'],
            'dilated': ['not_normalized', 'normalized']}},
    'connection_block': ['not_residual', 'residual']
}

# for reference: in combination, these parameter choice showed the best performance
BEST_OPTIMIZER = Adam(lr=0.0045)
BEST_PARS = {
    'outputs': 2,
    'activation': 'elu',
    'pooling_block': {'trainable': True},
    'information_block': {'inception': {'v2': 'b'}},
    'connection_block': 'residual'
}

OPTIMIZER = Adam(lr=0.0045)

# DO NOT CHANGE THE NAME, you can change the parameters
PARS = {
    'outputs': 1,
    'activation': 'relu',
    'pooling_block': {'trainable': False},
    'information_block': {'convolution': {'simple': 'normalized'}},
    'connection_block': 'not_residual'
}

# DO NOT REMOVE THESE LINES, they checks if your parameter choice is valid
assert PARS.keys() == ALLOWED_PARS.keys()
print(check_dict_subset(PARS, ALLOWED_PARS))