from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
from numpy import asarray
from Model import load_sample
from PIL import Image
import numpy as np

def predictEmotion(paintingName):
    model = load_model('saved_model')
    model.summary() 
    root = "Edvard_Munch/"
    image = Image.open(root + paintingName)
    image = image.resize((256, 256))
    image = np.array([asarray(image)])
    result = model.predict(image)
    if result[0][0] == 1:
        return "happiness", result
    elif result[0][1] == 1:
        return "sadness", result
    elif result[0][2] == 1:
        return "fear", result
    elif result[0][3] == 1:
        return "anger", result
    return "awe", result
    
print(predictEmotion("galloping-horse-1912.jpg"))