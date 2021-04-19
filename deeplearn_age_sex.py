from google.colab import drive 
drive.mount('/content/drive')

##
import os
if not os.path.exists("/content/drive/MyDrive/age_gender_classification/UTKFace"):
  os.makedirs("/content/drive/MyDrive/age_gender_classification/UTKFace")
##
  cd /content/drive/MyDrive/age_gender_classification/UTKFace


  #
  pwd
  # dataset_folder_name = '/media/kongbe/New Volume1/dataset/UTKFace'   # data path in local computer
dataset_folder_name = "/content/drive/MyDrive/age_gender_classification/UTKFace"# Data's path in google colab
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198

dataset_dict = {
    # 'race_id': {
    #     0: 'white', 
    #     1: 'black', 
    #     2: 'asian', 
    #     3: 'indian', 
    #     4: 'others'
    # },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

# dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
# dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())


print(dataset_dict)


import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##
def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, _, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)]
        except Exception as ex:
            return None, None

    # files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    files = [os.path.join(dataset_folder_name,i) for i in os.listdir(dataset_path)]
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'path_file']
    df = df.dropna()
    
    return df
    # return files
##
    df = parse_dataset(dataset_folder_name)
df.head()



#


import plotly.graph_objects as go
from IPython.display import display, Image

def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()
    
    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribution for %s' % pd_series.name)
        
    img_bytes = fig.to_image(format="png")
    display(Image(img_bytes))



    ##

import plotly.express as px
fig = px.histogram(df, x="age", nbins=20)
fig.update_layout(title_text='Age distribution')
fig.show()

##
from tensorflow.keras.utils import to_categorical
from PIL import Image
import tensorflow.keras
import cv2

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        # self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (IM_WIDTH, IM_HEIGHT))
        im = np.array(im, dtype=np.float32) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages, genders = [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = person['age']
                if person['gender'] == 'male':
                  gender = 0
                else:
                  gender = 1 
                file = person['path_file']
                
                im = self.preprocess_image(file)
                
                ages.append(age / self.max_age)
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(genders)]
                    images, ages, genders = [], [], []
                    
            if not is_training:
                break
                
data_generator = UtkFaceDataGenerator(df)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()


##
batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
print(len(train_idx))


##
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

class UtkMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def make_default_hidden_layers(self, inputs):                 
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    # def build_race_branch(self, inputs, num_races):
    #     """
    #     Used to build the race branch of our face recognition network.
    #     This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
    #     followed by the Dense output layer.
    #     """
    #     x = self.make_default_hidden_layers(inputs)

    #     x = Flatten()(x)
    #     x = Dense(128)(x)
    #     x = Activation("relu")(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.5)(x)
    #     x = Dense(num_races)(x)
    #     x = Activation("softmax", name="race_output")(x)

    #     return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):   
        """
        Used to build the age branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.

        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        age_branch = self.build_age_branch(inputs)
        # race_branch = self.build_race_branch(inputs, num_races)
        gender_branch = self.build_gender_branch(inputs)

        model = Model(inputs=inputs,
                     outputs = [age_branch, gender_branch],
                     name="face_net")

        return model
    
model = UtkMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)



#
%matplotlib inline

from keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plot_model(model, to_file='model.png')
img = mpimg.imread('model.png')

plt.figure(figsize=(40, 30))
plt.imshow(img)

from keras.optimizers import Adam

init_lr = 1e-4
epochs = 200

opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer=opt, 
              loss={
                  'age_output': 'mse', 
                  'gender_output': 'binary_crossentropy'},
              loss_weights={
                  'age_output': 4.,  
                  'gender_output': 0.1},
              metrics={
                  'age_output': 'mae', 
                  'gender_output': 'accuracy'})


###
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

checkpoint_path = "/content/drive/MyDrive/age_gender_classification/model_checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)

callbacks = [
    ModelCheckpoint(filepath=checkpoint_path, 
                    monitor="val_loss",
                    verbose=1,
                    save_weights_only=True)
]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//batch_size)


##
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

test_model = UtkMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)
init_lr = 1e-4
epochs = 20

opt = Adam(lr=init_lr, decay=init_lr / epochs)

test_model.compile(optimizer=opt, 
              loss={
                  'age_output': 'mse', 
                  'gender_output': 'binary_crossentropy'},
              loss_weights={
                  'age_output': 4.,  
                  'gender_output': 0.1},
              metrics={
                  'age_output': 'mae', 
                  'gender_output': 'accuracy'})
test_model.load_weights("/content/drive/MyDrive/age_gender_classification/model_checkpoint/cp-0070.ckpt")
# test_gen = data_generator.generate_images(test_idx, 
#                                           is_training= True,
#                                           batch_size=batch_size)
# results = test_model.evaluate(test_gen, 
#                               steps=len(test_idx)//batch_size,
#                               verbose=1)
print(len(valid_idx))
test_batch_size = 128
test_generator = data_generator.generate_images(test_idx[:1000], is_training=False, batch_size=test_batch_size)
age_pred, gender_pred = test_model.predict_generator(test_generator, 
                                                steps=len(test_idx[:1000])//test_batch_size,
                                                verbose=1)




test_generator = data_generator.generate_images(test_idx[:1000], is_training=False, batch_size=test_batch_size)
samples = 0
images, age_true, gender_true = [], [], []
for test_batch in test_generator:
    image = test_batch[0]
    labels = test_batch[1]
    
    images.extend(image)
    age_true.extend(labels[0])
    gender_true.extend(labels[1])
    
age_true = np.array(age_true)
gender_true = np.array(gender_true)

age_true = age_true * data_generator.max_age
age_pred = age_pred * data_generator.max_age

for i in gender_pred:
  for j in range(0,2):
    if i[j]<0.5:
      i[j]=0
    else: 
      i[j]=1
# print(gender_pred)
# np.expand_dims(age_true, axis=0)
# print(age_true.shape)
# print(gender_true[0][0])
import math
import numpy as np
import matplotlib.pyplot as plt

n = 16
random_indices = np.random.permutation(n)
n_cols = 4
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 17))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB))
   
    # plt.imshow(cv2.cvtColor(image[img_idx], cv2.COLOR_BGR2RGB))

    cur_age_pred = age_pred[img_idx][0]
    cur_age_true = age_true[img_idx]
    
    cur_gender_pred = gender_pred[img_idx][0]
    cur_gender_true = gender_true[img_idx][0]
    
    age_threshold = 10
    if cur_gender_pred == cur_gender_true and abs(cur_age_pred - cur_age_true) <= age_threshold:
        ax.xaxis.label.set_color('green')
    elif cur_gender_pred != cur_gender_true  and abs(cur_age_pred - cur_age_true) > age_threshold:
        ax.xaxis.label.set_color('red')
    else:
        ax.xaxis.label.set_color('purple')
    ax.set_xlabel('a: {}, g: {}'.format(int(age_pred[img_idx]),
                            dataset_dict['gender_id'][gender_pred[img_idx][1]]))
    
    ax.set_title('a: {}, g: {}'.format(int(age_true[img_idx]),
                            dataset_dict['gender_id'][gender_true[img_idx][1]]))
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.savefig('preds.png')