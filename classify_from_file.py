import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from subprocess import run
from random import randrange, choice

load_path = 'faces_model_save'
model = models.load_model(load_path)

with open(f'{load_path}/class_names.data', 'rb') as f:
    class_names = pickle.load(f)
print(class_names)

correct = 0
num_test_images = 50
for i in range(num_test_images):
    person = choice(class_names)
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
    print(f"{person} {class_names[prediction]}")
    if person == class_names[prediction]:
        correct += 1

print(f"Number of test images: {num_test_images}, Correct: {correct}, % correct: {correct/num_test_images * 100:.0f}%")