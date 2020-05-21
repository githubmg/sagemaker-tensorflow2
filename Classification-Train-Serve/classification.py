import argparse
import numpy as np
import os
import tensorflow as tf
#from tensorflow.contrib.eager.python import 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers

#para prediccion
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard



train_data_gen_args = dict(rescale=1./255,
                    shear_range=0.01,
                    rotation_range = 20,
                    zoom_range=0.2,
                    height_shift_range = 0.2,
                    width_shift_range = 0.2,
                    brightness_range=[0.1, 1.9],
                    horizontal_flip=True)

data_gen_args = dict(target_size=(224, 224),
        batch_size=16,
        shuffle=True,
        #color_mode='grayscale',
        class_mode='categorical')

num_classes = 4
def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_train_val_data(train_dir,validation_dir):
    train_datagen = ImageDataGenerator(**train_data_gen_args)
    train_generator = train_datagen.flow_from_directory(train_dir, **data_gen_args)
    val_generator = train_datagen.flow_from_directory(validation_dir, **data_gen_args)
    
    print('train shape:', train_generator[0][0].shape,'val shape:', val_generator[0][0].shape)

    return train_generator, val_generator


def get_model():
    base_model = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
    #x = GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False
        
    return model


if __name__ == "__main__":

    args, _ = parse_args()
    
    train_generator, val_generator = get_train_val_data(args.train,args.validation)
    
    model = get_model()
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
       
    callbacks_list = []
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    
    model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=args.epochs,  validation_data=val_generator, validation_steps=STEP_SIZE_VALID, callbacks=callbacks_list)

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    model.save(args.model_dir + '/1')
    #tf.keras.models.save_model(model, args.model_dir)


