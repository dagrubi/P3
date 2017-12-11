from keras.layers import Flatten, Dense
from keras.layers import Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import model_from_json
import json
from sklearn.model_selection  import train_test_split
from sklearn.utils import shuffle as shuffle
import random
from keras import backend as K 
import os
import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import math
import helper



def correct_filename(file_name):
	file_split = file_name.split("windows_sim")
	res_filename = '.' + file_split[1].replace('\\','/')
	return res_filename
	
	

def generator(samples, batch_size):
	correct_steering = [0.0, 0.27, -0.27 ]
	
	while 1:
		for part_batch in range(0, len(samples), batch_size):
			x_dat = []
			y_dat = []
			cnt_0 = 0
			print_out = 0
			batch = samples[part_batch:part_batch+batch_size]
			for meas in batch:
				steering = float(meas[3])
				speed = float(meas[6])
				#if speed > 2:
				#if abs(steering) < 0.1:
				#	cnt_0 += 1
				#if (cnt_0 > 0.5*batch_size):
				#	rand_idx = random.randrange(len(samples))
				#	while abs(samples[rand_idx][3]) < .1:
				#		rand_idx = random.randrange(len(samples))
				#	sel_img = samples[rand_idx]
				#	steering = samples[rand_idx][3]
				#else:
				sel_img = meas	
				for ii in range(3):
					pic = cv2.imread(correct_filename(sel_img[ii]))
					angle = steering + correct_steering[ii]
					# pic = helper.preprocess(pic)
					# convert to rgb-picture
					pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
					# randomly change brightness
					pic = cv2.cvtColor(pic, cv2.COLOR_RGB2HSV)
					brightness = 0.25 + np.random.uniform()
					pic[:,:,2] = pic[:,:,2] * brightness
					pic = cv2.cvtColor(pic, cv2.COLOR_HSV2RGB)
					# cropping
					pic2 = pic[40:-20,:]
					img_resize = cv2.resize(pic2, (200, 66), interpolation=cv2.INTER_AREA)
					
					# convert to yuv
					img = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)
					#if print_out == 0:
					#print_out += 1
					#print("#### pic shape ####")
					#print(pic.shape)
					#shape (66, 200, 3)
					# 50% of the pics are flipped
					if random.randrange(2) == 1:
						x_dat.append(img)
						y_dat.append(angle)
					else:
						# augmented pic
						x_dat.append(cv2.flip(img,1))
						y_dat.append(angle * -1.0)
								
			x_data = np.array(x_dat)
			y_data = np.array(y_dat)
			yield x_data, y_data

def build_model(part_drop):
	model = Sequential()
	
	#model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
	model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(66,200,3)))
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid", activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid", activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode="valid", activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", activation='relu'))
	model.add(Flatten())
	model.add(Dropout(part_drop))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(part_drop))
	model.add(Dense(1))
	return model
	
def train_model(train_sample, valid_sample, nr_batch, nr_epochs, dropout_rate, nr_model):

	print(" Generator ")
	train_generator = generator(train_sample, nr_batch)
	valid_generator = generator(valid_sample, nr_batch)
	model = build_model(dropout_rate)
	print(model.summary())
	model.compile(loss='mse', optimizer='adam')
	checkpoint = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)

	print(" #### Training ###### ")
	hist_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_sample)*10, 
		validation_data=valid_generator, nb_val_samples=len(valid_sample), nb_epoch=nr_epochs, verbose=1,
		callbacks=[checkpoint])

	print("#### Save Model ###### ")
	# Save model data
	model.load_weights("./weights.hdf5")
	json_string = model.to_json()
	with open("model.json", 'w') as f:
		f.write(json_string)
	model.save_weights("model.h5")

	return hist_obj


if __name__=="__main__":
	
	## parameter
	size_batch = 128
	nr_epoch = 24
	drop_fact = 0.15
	NR_MODEL = 9

	# load data
	x_data = []
	y_data = []

	## data sources
	list_files = []
	#list_files.append('./record_2')
	# list_files.append('./record_1')
	list_files.append('./data')

	# create list of samples
	samples = []
	cnt_0 = 0
	cnt_all = 0
	for file in list_files:
		log_file = file + '/driving_log.csv'
		with open(log_file) as csvfile:
			reader = csv.reader(csvfile, skipinitialspace=True)
			for line in reader:
				# filter
				cnt_all += 1
				steering = float(line[3])
				if abs(steering) < 0.1:
					cnt_0 +=1
				if abs(steering) > 0.1 or cnt_0 < 0.4*len(samples):
					if float(line[6]) > 2:
						samples.append(line)
						
	print("Number of samples: ", len(samples), " of ", cnt_all)
				
	# get rain data
	sample_train, sample_valid = train_test_split(samples, test_size=0.2)
	print("No Train - Test")
	print(len(sample_train), " - ", len(sample_valid))

	# train model
	hist_obj = train_model(sample_train, sample_valid, size_batch, nr_epoch, drop_fact, NR_MODEL)
	
	K.clear_session()
	
