import _pickle as pickle
import os
from multiprocessing import Queue, Process

import cv2
import numpy as np
from sklearn.cluster import KMeans

from LoadData import LoadData


def write_to_file(storage_path, value_object, file_name='save.pkl'):
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    codewords_file_path = storage_path + file_name
    with open(codewords_file_path, mode='wb') as saved_file:
        pickle.dump(value_object, saved_file)


class BoWBuilder:

    # main processing in extracting features
    def featureExtraction(self, img_paths_list, start_point, end_point, core_index, queue):
        i = start_point
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=15, nOctaveLayers=6,
                                           contrastThreshold=0.08, edgeThreshold=15,
                                           sigma=1.6)
        all_keypoints = []
        # innitial for passing ValueError: all the input arrays must have same number of dimensions
        all_descriptions = np.array([], dtype=np.float32).reshape(0, 128)
        if start_point <= len(img_paths_list):
            if end_point <= len(img_paths_list):
                path_list = img_paths_list[start_point:end_point]
            else:
                path_list = img_paths_list[start_point:len(img_paths_list)]
        else:
            path_list = []

        for image_path in path_list:
            print(str(i), '  --  ', image_path)
            # print(".", end='', flush=True)
            i = i + 1
            image = cv2.imread(image_path)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # key_points = sift.detect(gray_img, None)
            key_points, descriptions = sift.detectAndCompute(gray_img, None)
            all_keypoints = all_keypoints + key_points
            if descriptions is not None:
                all_descriptions = np.concatenate((all_descriptions, descriptions), axis=0)
        print("Finish process in core No." + str(core_index) + " !")
        queue.put(all_descriptions)

    # using multicore to optimize the preocessing speed
    def multiprocessingFeatureDescripting(self, img_paths_list):
        # multicores processing
        num_of_cores = 8
        if len(img_paths_list) % num_of_cores:
            distance = int(len(img_paths_list) / num_of_cores) + 1
        else:
            distance = int(len(img_paths_list) / num_of_cores)
        queue = Queue()
        # processes spliting
        similarities = [
            Process(target=self.featureExtraction, args=(img_paths_list, x * distance, (x + 1) * distance, x, queue))
            for x in range(0, num_of_cores + 1)]

        print('Extracting descriptions: ', end='', flush=True)
        for similar in similarities:
            similar.start()

        queue_result = []
        # print('Appending in result! ', end='', flush=True)
        for similar in similarities:
            similar.join(10)
            queue_result.append(queue.get(True))
            print(".", end='', flush=True)

        # innitial for passing ValueError: all the input arrays must have same number of dimensions
        all_descriptions = np.array([], dtype=np.float32).reshape(0, 128)
        for ele in queue_result:
            all_descriptions = np.concatenate((all_descriptions, ele), axis=0)

        self.write_to_file(storage_path='extracted_feature_descriptions/', file_name='all_descriptors.pkl',
                           value_object=all_descriptions)
        return all_descriptions

        # --- COLLECT ALL KEYPOINT AND THEIR DESCRIPTORS ---

    def noneMulticoreFeatureDescripting(self, img_paths_list):
        all_keypoints = []
        all_descriptors = np.array([], dtype=np.float32).reshape(0, 128)
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=15, nOctaveLayers=6,
                                           contrastThreshold=0.08, edgeThreshold=15,
                                           sigma=1.6)
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
        data_path = "backup"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        codewords_file_path = data_path + '/all_descriptors.dat'
        with open(codewords_file_path, mode='wb') as backup_file:
            # for centroid in centroids:
            # 	codeword_file.write(str(centroid))
            # 	codeword_file.write('\n')
            # np.save(codeword_file, centroids)
            pickle.dump(all_descriptors, backup_file)

        return all_descriptors, all_keypoints

    # BUILDING CODEBOOK
    # --- using KMEANS CLUSTERING --
    def buildingCodeWord(self, img_paths_list, multi_core=False, need_to_extract_features=True):
        if need_to_extract_features:
            if multi_core:
                all_descriptor_vectors = self.multiprocessingFeatureDescripting(img_paths_list)
            else:
                all_descriptor_vectors, all_keypoints = self.noneMulticoreFeatureDescripting(img_paths_list)
        else:
            with open('extracted_feature_descriptions/all_descriptors.pkl', 'rb') as handle:
                all_descriptor_vectors = pickle.load(handle)
        # print(all_descriptor_vectors)
        print('Finish extracting descriptors!')
        print('Number of descriptors: ', len(all_descriptor_vectors))

        # Number of clusters
        kmeans = KMeans(n_clusters=500)
        # Fitting the input data
        kmeans = kmeans.fit(np.array(all_descriptor_vectors))

        # Getting the cluster labels
        labels = kmeans.predict(np.array(all_descriptor_vectors))
        # Centroid values
        centroids = kmeans.cluster_centers_
        print('Finish clustering')
        # Comparing with scikit-learn centroids
        print(centroids)  # From sci-kit learn

        # Write to file:
        self.write_to_file(storage_path='BoW_codebook/', file_name='Codebook.pkl', value_object=centroids)
        # data_path = "codeword"
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)
        # codewords_file_path = data_path + '/codewords.dat'
        # with open(codewords_file_path, mode='wb') as codeword_file:
        #     # for centroid in centroids:
        #     # 	codeword_file.write(str(centroid))
        #     # 	codeword_file.write('\n')
        #     # np.save(codeword_file, centroids)
        #     pickle.dump(centroids, codeword_file)

        print("Finish building codebook!")


if __name__ == '__main__':
    builder = BoWBuilder()
    img_paths_list = LoadData().getImagePath('db1')
    # print(img_paths_list)
    builder.buildingCodeWord(img_paths_list['trainImage'], multi_core=True, need_to_extract_features=False)

# # load file pickle
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)
