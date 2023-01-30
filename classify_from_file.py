import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from subprocess import run
from random import randrange, choice

model = models.load_model('faces_model_save')

proc = run(['ls', 'images'], capture_output = True)
class_labels = sorted(proc.stdout.decode().split())

correct = 0
num_test_images = 50
for i in range(num_test_images):
    person = choice(class_labels)
    img_num = randrange(100)
    img = utils.load_img(
        f'images/{person}/img_{img_num}.jpg',
        target_size = (250, 250),
        )
    img_array = utils.img_to_array(img)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    print(f"{person} {class_labels[prediction]}")
    if person == class_labels[prediction]:
        correct += 1

print(f"Number of test images: {num_test_images}, Correct: {correct}, % correct: {correct/num_test_images * 100:.0f}%")