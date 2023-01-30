import time
from subprocess import run
from random import randrange, choice

import cv2
import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf

proc = run(['ls', 'images'], capture_output = True)
class_labels = sorted(proc.stdout.decode().split())
model = models.load_model('faces_model_save')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera.")
    quit()
else:
    print("Opened camera.")

time.sleep(1)

frames = 0
start = time.time()
while True:
    ret, img = cap.read()
    if ret: 
        frames += 1
    else:
        break
    # new_img = img[220:520, 540:840]
    new_img = cv2.resize(img, (250, 250))
    img_array = utils.img_to_array(new_img)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    fps = frames / (time.time() - start)
    cv2.putText(
        img, 
        f"{fps:0f} fps, {class_labels[prediction]}", 
        (10,30), 
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, 
        (0, 255, 0),
        2
    )
    print(f"{frames:5} {class_labels[prediction]:10}", end=" ")
    if frames % 5 == 0:
        print()
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()
quit()
