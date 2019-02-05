import numpy as np
import keras
from keras import backend as k
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as ply


mobile = keras.applications.mobilenet.MobileNet()

def prep_img(file):
	img = image.load_img('test4.jpg', target_size = (224,224))
	img_array = image.img_to_array(img)
	img_arr_exp_dims = np.expand_dims(img_array, axis = 0)
	return keras.applications.mobilenet.preprocess_input(img_arr_exp_dims)

from IPython.display import Image
Image(filename = 'test4.jpg', width = 300, height = 300)

preprocessed_img = prep_img('test4.jpg')
predictions = mobile.predict(preprocessed_img)
results = imagenet_utils.decode_predictions(predictions)
print(results)
