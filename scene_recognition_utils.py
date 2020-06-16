# Taken from https://github.com/GKalliatakis/Keras-VGG16-places365
import os
import urllib.request
import numpy as np
from PIL import Image
from cv2 import resize
from pathlib import Path

from Keras_VGG16_places365.vgg16_places_365 import VGG16_Places365

