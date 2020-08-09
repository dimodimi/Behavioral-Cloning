from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

INPUT_SHAPE = (160, 320, 3)
keep_prob   = 0.3

def nvidia_net():
    model = Sequential()
    
    # Crop 50 pixels from top and 20 pixels from bottom of image
    model.add( Cropping2D( cropping=((50, 20), (0, 0)), input_shape=INPUT_SHAPE ) )
    
    # Normalize
    model.add( Lambda( lambda x: (x / 127.5) - 1 ) )
    
    # Conv layers
    model.add( Convolution2D(24,5,5,subsample=(2,2),activation="relu") )
    model.add( Convolution2D(36,5,5,subsample=(2,2),activation="relu") )
    model.add( Convolution2D(48,5,5,subsample=(2,2),activation="relu") )
    model.add( Convolution2D(64,3,3,                activation="relu") )
    model.add( Convolution2D(64,3,3,                activation="relu") )
    
    model.add( Flatten() )
    
    # FC layers with dropout
    # fully connected layers with dropouts
    model.add( Dense(100) )
    model.add( Dropout(keep_prob) )
    model.add( Dense(50) )
    model.add( Dropout(keep_prob) )
    model.add( Dense(10) )
    model.add( Dropout(keep_prob) )
    model.add( Dense(1) )
    
    return model