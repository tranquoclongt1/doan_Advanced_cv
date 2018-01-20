import cv2
import matplotlib.pyplot as plt
import numpy as np
from LoadData import LoadData
from multiprocessing import Pool, Queue, Process
from sklearn.cluster import KMeans
import os
import _pickle as pickle


class BoWBuilder:
	def featureExtraction(img_paths_list, start_point, end_point, queue):
		i = start_point
		sift = cv2.xfeatures2d.SIFT_create()        
		all_keypoints = []
		all_descriptors = []
		if start_point <= len(img_paths_list) :
			if end_point <= len(img_paths_list): 
				path_list = img_paths_list[start_point:end_point]
			else:
				path_list = img_paths_list[start_point:len(img_paths_list)]
		else:
			path_list = []

		for image_path in path_list:
			print(str(i),'  --  ',image_path)
			i = i + 1
			image = cv2.imread(image_path)
			gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# key_points = sift.detect(gray_img, None)
			key_points, descriptors = sift.detectAndCompute(gray_img, None)
			all_keypoints = all_keypoints + key_points
			all_descriptors.append(descriptors)
		print("Finish process in 1 core!")
		queue.put(all_descriptors)

	def multiprocessingFeatureDescripting(img_paths_list):
		# multicores processing
		num_of_cores = 32
		distance = int(len(img['trainImage'])/num_of_cores)
		queue = Queue()
		similarities = [Process(target=featureExtraction, args=(img_paths_list, x*distance, (x+1)*distance, queue)) for x in range (0,num_of_cores+1)]

		for similar in similarities:
			similar.start()

		result = []
		print('Appending in result!')
		for similar in similarities:
			similar.join(10)
			result.append(queue.get(True))

		return result

    # --- COLLECT ALL KEYPOINT AND THEIR DESCRIPTORS ---
	def noneMulticoreFeatureDescripting(img_paths_list):
		all_keypoints = []
		all_descriptors = np.array([], dtype=np.float32).reshape(0,128)
		sift = cv2.xfeatures2d.SIFT_create()
		i = 1
		for image_path in img_paths_list:
			print(str(i), ' --- Processing on:', image_path)
			i = i + 1
			image = cv2.imread(image_path)
			gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# key_points = sift.detect(gray_img, None)
			key_points, descriptors = sift.detectAndCompute(gray_img, None)
			all_keypoints = all_keypoints + key_points
			# print('descriptors:\n', descriptors)
			# all_descriptors = all_descriptors + descriptors
			# all_descriptors.append(descriptors)
			# print (all_descriptors.shape)
			# print ( descriptors.shape)
			# all_descriptors = np.concatenate((all_descriptors, descriptors), axis=0)
			if descriptors is not None:
				all_descriptors = np.concatenate((all_descriptors, descriptors), axis=0)
		return all_descriptors, all_keypoints

	# --- KMEANS CLUSTERING -- 

	# all_descriptor_vectors = multiprocessingFeatureDescripting()

	def buildingCodeWord(img_paths_list):
		all_descriptor_vectors, all_keypoints = noneMulticoreFeatureDescripting(img_paths_list)
		print('Number of descriptors: ',len(all_descriptor_vectors))
		print(all_descriptor_vectors)
		print('Finish extracting descriptors!')
		# Number of clusters
		kmeans = KMeans(n_clusters=20000)
		# Fitting the input data
		kmeans = kmeans.fit(np.array(all_descriptor_vectors))

		# Getting the cluster labels
		labels = kmeans.predict(np.array(all_descriptor_vectors))
		# Centroid values
		centroids = kmeans.cluster_centers_
		print('Finish clustering')
		# Comparing with scikit-learn centroids
		print(centroids) # From sci-kit learn

		# Write to file:
		data_path = "codeword"
		if not os.path.exists(data_path):
			os.makedirs(data_path)
		codewords_file_path = data_path + '/codewords.dat'   
		with open(codewords_file_path, mode='wb') as codeword_file:
			# for centroid in centroids:
			# 	codeword_file.write(str(centroid))
			# 	codeword_file.write('\n')
			# np.save(codeword_file, centroids)
			pickle.dump(centroids, codeword_file)

		print("Finish building codebook!")

if __name__ == '__main__':
	builder = BoWBuilder()
	img_paths_list = LoadData().getImagePath('db1')
	builder.buildingCodeWord(img_paths_list)

# # load file pickle
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)
