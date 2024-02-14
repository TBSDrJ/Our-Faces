import pickle
from pprint import pprint
import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.callbacks as callbacks
import tensorflow.data as data

train = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (250, 250),
    seed = 37,
    validation_split = 0.3, 
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (250, 250),
    seed = 37,
    validation_split = 0.3, 
    subset = 'validation',
)

print("Class Names:")
pprint(train.class_names)
class_names = train.class_names

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = data.AUTOTUNE)

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
        # Next up Conv with Frame: 3 x 3, Strides: 1, Depth: 8
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Conv2D(8, 3, activation = 'relu'))
        # Output: 38 x 38 x 16
        # Next up maxpool again. Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 19 x 19 x 16
        # Now, flatten
        self.model.add(layers.Flatten())
        # Output length: 5776
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
print(net)

callbacks = [
    callbacks.ModelCheckpoint(
        'checkpoints/checkpoint_1_{epoch:02d}', 
        verbose = 2, 
        save_freq = 152,
    )
]

net.model.fit(
    train,
    batch_size = 32,
    epochs = 80,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
    callbacks = callbacks,
)

# save_path = 'saves/faces_model_save_2023_02_08__80_epochs_dropout'
# net.model.save(save_path)
# with open(f'{save_path}/class_names.data', 'wb') as f:
#     pickle.dump(class_names, f)
