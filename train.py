# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(200, 200, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # binary_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(200, 200),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(200, 200),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='binary') 

STEP_SIZE_TRAIN=training_set.n
STEP_SIZE_TEST=test_set.n

classifier.fit(
        training_set,
        epochs=1,
        validation_data=test_set)


# Saving the model
model_json = classifier.to_json()
with open("./models/model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('./models/model-bw.h5')

