from air_draw import draw


a = draw()


import cv2

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./dense/')

a = cv2.resize(a,(28,28))
print(a.shape)
a = a.reshape(-1,28,28,1)
a = a/255.0
print(a.shape)
result = model.predict(a)
result = np.array(result)
#print(result)
#print(np.where(result == result.max()))
z = np.argmax(result)
img = np.ones((504,504,1), np.uint8)

org = (200, 250)

font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 8

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 3

cv2.putText(img,str(z),org, font,fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("abc",img)
cv2.waitKey(0)
