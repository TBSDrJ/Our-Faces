from pprint import pprint

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.callbacks as callbacks
from tensorflow.data import Dataset
from tensorflow.train import Checkpoint
import tensorflow as tf
import numpy as np
import cv2

SZ = 250

train = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (SZ, SZ),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (SZ, SZ),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)


class_names = train.class_names

print("Class Names:")
pprint(class_names)

# If you are loading from a checkpoint, you need to define the model
# before loading.  If you are loading a saved model, you can comment out
# the entire class definition because the model architecture is built 
# into the saved model.

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()
        # Input: SZ x SZ x 3
        # First layer is convolution with:
        # Frame/kernel: 13 x 13, Stride: 3 x 3, Depth: 8
        self.model.add(layers.Conv2D(8, 13, strides = 3,
            input_shape = image_size, activation = 'relu'))
        # Output: 80 x 80 x 8
        # Next layer is maxpool, Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 40 x 40 x 8
        # self.model.add(layers.Dropout(0.3))
        # Next up Conv with Frame: 3 x 3, Strides: 1, Depth: 8
        self.model.add(layers.Conv2D(8, 3, activation = 'relu'))
        # Output: 38 x 38 x 16
        # Next up maxpool again. Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 19 x 19 x 16
        # Now, flatten
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Flatten())
        # Output length: 5776
        self.model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(256, activation = 'relu'))
        self.model.add(layers.Dense(64, activation = 'relu'))
        # Softmax activation will turn values into probabilities
        self.model.add(layers.Dense(17, activation = 'softmax'))
        # Also try CategoricalCrossentropy()
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )
    def __str__(self):
        self.model.summary()
        return ""


for person in class_names:
    # Get the first image of that person and set it up
    img = cv2.imread(f'images/{person}/img_0.jpg')
    img = cv2.resize(img, (SZ, SZ))
    img = utils.img_to_array(img)
    img = img[tf.newaxis, ...]

    # Did checkpoints every 2 epochs up to 40.
    #   or every 4 epochs up to 80 or every 8 up to 200 or every 10 to 380.
    for k in range(10, 321, 10):
        # Set up the architecture and load in the checkpoint weights
        net = Net((SZ, SZ, 3))
        # print(net)
        checkpoint = Checkpoint(net.model)
        checkpoint.restore(f'checkpoints/checkpoint_5_{k:02d}').expect_partial()
        # Get the first conv layer, feed the image and set it up for viewing
        filters = net.model.layers[0](img)[0]
        shape = filters.shape
        filters = filters.numpy()
        # Put all filters in one big mosaic image with 2 rows, padded by 
        #   20px gray strips.  
        # Scaling up the filters by 3x to make them easier to see
        cols = shape[2] // 2
        mosaic = np.zeros(
            (6*shape[0] + 20, 3*cols*shape[1] + (cols - 1)*20)
        )  
        # Print the filter max and average to screen so we can see how much
        #   the classification uses this filter.
        print(f'{person:>12} Chkpt {k:02d} Maxes:', end = ' ')
        second_str = '                      Avgs: '
        # Shape[2] = number of filters
        for i in range(shape[2]):
            # Get just one filter
            filter = filters[0:shape[0],0:shape[1],i]
            # Calculate and print max and avg
            maxes = []
            avgs = []
            for j in range(shape[0]):
                maxes.append(max(filter[j]))
                avgs.append(sum(filter[j])/len(filter[j]))
            print(f'{max(maxes):8.4f}', end = ' ')
            second_str += f'{sum(avgs)/len(avgs):9.4f}'
            # Triple the filter size to make it easier to see
            filter = cv2.resize(filter, (3*shape[0], 3*shape[1]))
            # Rescale so the grayscale is more useful
            filter = filter / max(maxes) * 2
            # Locate the filter in the mosaic and copy the values in
            offset = ((i % 2)*(3*shape[0] + 20), (i // 2)*(3*shape[1] + 20))
            mosaic[
                offset[0]:offset[0] + 3*shape[0], 
                offset[1]:offset[1] + 3*shape[1]] = filter  
        print()
        print(f'{second_str}')
        # Make the gray stripes that separate the filters
        # Vertical Stripes
        for i in range(1, cols):
            start_vert_stripe = 3*i*shape[1] + (i - 1)*20
            mosaic[
                0:mosaic.shape[0], 
                start_vert_stripe:start_vert_stripe + 20] = np.ones(
                    (mosaic.shape[0], 20)) * 0.5  
        # Horizontal Stripe
        mosaic[3*shape[0]:3*shape[0] + 20, 0:mosaic.shape[1]] = np.ones(
                (20, mosaic.shape[1])) * 0.5
        # Display the image
        cv2.imshow(f'{person} Checkpoint {k}', mosaic)
        if chr(cv2.waitKey(0)) == 'q':
            quit()
        cv2.destroyAllWindows()
