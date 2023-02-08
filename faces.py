import pickle
from pprint import pprint
import tensorflow.keras.utils as utils
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
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

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = data.AUTOTUNE)

class Net():
    def __init__(self, input_shape):
        self.model = Sequential()
        # self.model.add(layers.ZeroPadding2D(
        #     padding = ((1,0), (1,0)),
        # ))
        self.model.add(layers.Conv2D(
            8, # filters
            13, # kernel
            strides = 3, # a.k.a. step size
            activation = 'relu',
        )) # Output: 80 x 80 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,
        )) # Output: 40 x 40 x 8
        self.model.add(layers.Conv2D(
            8, # filters
            3, # kernel
            strides = 1, 
            activation = 'relu',
        )) # Output: 38 x 38 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,
        )) # Output: 19 x 19 x 8
        self.model.add(layers.Flatten(
        )) # Output: 2888
        self.model.add(layers.Dense(
            1024, 
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            256,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            64,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            17, # Exactly equal to number of classes
            activation = 'softmax', # Always use softmax on your last layer
        ))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy']
        )
    def __str__(self):
        self.model.summary()
        return ""

    # allows you to use net.save() instead of net.model.save()
    def save(self, filename):
        self.model.save(filename)

net = Net((250, 250, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 40,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)
