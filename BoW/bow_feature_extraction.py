import numpy as np
import cv2 
import matplotlib.pyplot as plt
import glob, os
from multiprocessing import Pool, Queue, Process
from sklearn.cluster import KMeans
import os
import _pickle as pickle
import re


class ExtractFeaturesBoW:
	def getNearestCentroidIndex(self, number_of_clusters, centroids, keypoint_descriptor_vector):
		for centroid in centroids:
		    distance = np.linalg.norm(np.array(keypoint_descriptor_vector)-np.array(centroid))
		    print(distance)
		    distances_array.append(distance)
		print (len(distances_array))
		print(distances_array)
		print ('index of centroid having min distance: %i'%distances_array.index(min(distances_array)))
		min_index = distances_array.index(min(distances_array))
		return min_index

	def evaluatingHistogram(self, number_of_clusters, centroids, list_of_feature_descriptor_vectors):
		number_of_demention_of_histogram = number_of_clusters
		histogram = np.zeros(number_of_demention_of_histogram)
		for descriptor in list_of_feature_descriptor_vectors:
			min_index = self.getNearestCentroidIndex(number_of_clusters, centroids, descriptor)
			histogram[min_index] += 1
		print('feature vector under histogram form: ', histogram)
		return histogram

	def bowFearuteExtraction(self, number_of_clusters, centroids, img_path):
		img = cv2.imread(img_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeature2d.SITF_create()
		keypoints, descriptors = sift.detectAndCompute(gray,None)
		histogram_type_feature = self.evaluatingHistogram(number_of_clusters, centroids, descriptors)
		return histogram_type_feature

	def bowExtractAllFeatureAndSave(self, img_paths_list, return_features=False):
		number_of_clusters = 20000
		# load file pickle
		with open('codeword/codewords.dat', 'rb') as handle:
			centroids = pickle.load(handle)
		
		features = []
		for img_path in img_paths_list:
			histogram_type_feature = 
			print("Extract Feature from file: ", img_path)
            feature = self.bowFearuteExtraction(number_of_clusters, centroids, image_path)

            # print feature shape
            if count == 0:
                print("Feature shape: ", feature.shape)

            # store feature to storage
            path_split = re.split('[. //]', img_path)
            # print("path split 0: ", path_split[0])
            # print("paht split 1: ", path_split[1])
            file_path = "features/" + model_name + '/' + path_split[1] + "/"
            file_name = path_split[2] + '.pkl'

            # create directory if not exist
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            # using pickle to save variable to file
            with open(file_path + file_name, mode='wb') as feature_file:
                pickle.dump(feature, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

            count += 1

            if return_features:
                features.append(feature)

if __name__ == '__main__':
	extractor = ExtractFeaturesBoW()
    image_paths_list = glob.glob('dataset/101_ObjectCategogies//*//*.jpg')
	extractor.bowExtractAllFeatureAndSave(image_paths_list)