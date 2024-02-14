import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.callbacks as callbacks
from tensorflow.data import Dataset
import numpy as np

train = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (250, 250),
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
    image_size = (250, 250),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)

# # Data Augmentation Section
# # You can have multiple layers, but first one always needs the input_shape
# rotation = models.Sequential([
#     layers.RandomRotation(0.25, input_shape = (250, 250, 3))
# ])
# # train.map() applies the transformation in parentheses to each pair x,y 
# # in the dataset.  We only need to transform the x-values, we just pass
# # the y-values along passively.  Notice that the output of the lambda
# # function is a 2-tuple, which is the transformed image followed
# rotated = train.map(lambda x, y: (rotation(x), y))
# # Make sure you create *all* the transformed copies before assembling
# # them into a new training set.
# train = train.concatenate(rotated)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()
        # Input: 250 x 250 x 3
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
        # Output: 38 x 38 x 8
        # Next up maxpool again. Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 19 x 19 x 8
        # Now, flatten
        self.model.add(layers.Flatten())
        # Output length: 2888
        self.model.add(layers.Dense(1024, activation = 'relu'))
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

net = Net((250, 250, 3))
# print(net)

print(type(train))
quit()

# callbacks = [
#     callbacks.ModelCheckpoint(
#         'checkpoints{epoch:02d}', 
#         verbose = 1, 
#         save_freq = 80
#     )
# ]
net.model.fit(
    train,
    batch_size = 32,
    epochs = 40,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
    # callbacks = callbacks
)
