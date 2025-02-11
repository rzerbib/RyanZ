import cv2
import numpy
img = numpy.load('/workspace/WildfireSpreadTS/2018/fire_21458798/2018-01-01.tif',allow_pickle = True)
#img = cv2.imread('/workspace/WildfireSpreadTS/2018/fire_21458798/2018-01-01.tif')
height, width, channels = img.shape
print(height, width, channels)