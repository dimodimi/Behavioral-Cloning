import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Model
from scipy import ndimage

from nvidia_model import nvidia_net

EPOCHS = 5
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
ANGLE_CORRECTION = 0.2 # radians

def main():
    # Read the data
    # Each sample contains the three images and steering angle
    # We will do full augmentation using all three original images + flipped versions
    samples = []
    with open('/home/workspace/CarND-Behavioral-Cloning-P3/new_data/IMG/IMG/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip headers
        for line in reader:
            samples.append(line)
    
    # Shuffle once and split into train/val sets
    samples = sklearn.utils.shuffle(samples)
    train_samples = samples[:int(TRAIN_SPLIT*len(samples))]
    val_samples   = samples[int(TRAIN_SPLIT*len(samples)):]
    
    print("Number of train images: " + str(2 * len(train_samples)))
    print("Number of validation images: " + str(2 * len(val_samples)))
    
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(val_samples, batch_size=BATCH_SIZE)

    model = nvidia_net()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, 
                        samples_per_epoch=2*len(train_samples),
                        validation_data=validation_generator, 
                        nb_val_samples=2*len(val_samples), 
                        nb_epoch=EPOCHS, 
                        verbose=1)

    # save the model
    model.save('model.h5')
    
    return
    
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Add center + side images + flipped versions
                add_images(images, angles, batch_sample)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def add_images(images, angles, sample):
    # Filenames in sample may have a leading space which we need to skip
    # Add center image to samples
    center_img_filename = '/home/workspace/CarND-Behavioral-Cloning-P3/new_data/IMG/IMG/IMG/' + sample[0].split('/')[-1]
    img = ndimage.imread(center_img_filename) # RGB format
    center_angle = float(sample[3])
    images.append(img)
    angles.append(center_angle)
    # Add flipped version
    flipped_img = np.fliplr(img)
    images.append(flipped_img)
    angles.append(-center_angle)
    
    
    # Same for left and right images
    #left_img_filename = '/opt/carnd_p3/data/IMG/' + sample[1].split('/')[-1]
    #img = ndimage.imread(left_img_filename)
    #left_angle = center_angle + ANGLE_CORRECTION
    #images.append(img)
    #angles.append(left_angle)
    #flipped_img = np.fliplr(img)
    #images.append(flipped_img)
    #angles.append(-left_angle)
    
    #right_img_filename = '/opt/carnd_p3/data/IMG/' + sample[2].split('/')[-1]
    #img = ndimage.imread(right_img_filename)
    #right_angle = center_angle - ANGLE_CORRECTION
    #images.append(img)
    #angles.append(right_angle)
    #flipped_img = np.fliplr(img)
    #images.append(flipped_img)
    #angles.append(-right_angle)
    
if __name__ == '__main__':
    main()
