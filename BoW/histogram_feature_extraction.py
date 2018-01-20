import _pickle as pickle
import glob
import re
from multiprocessing import Queue, Process

import cv2
import numpy as np

from BoWCodeWordBuilder import write_to_file


class HistogramFeature():
    def compute_feature_histogram(self, centroids, image_path):
        # inititalize sift detector
        sift = cv2.xfeatures2d.SIFT_create()
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=15, nOctaveLayers=6,
        #                                    contrastThreshold=0.08, edgeThreshold=15,
        #                                    sigma=1.6)
        # read image and extract sift features descriptor:
        image = cv2.imread(image_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = sift.detectAndCompute(gray_img, None)

        # Initialize the historgram ( ~ feature vector)
        number_of_demention_of_histogram = len(centroids)
        histogram = np.zeros(number_of_demention_of_histogram)

        for keypoint_descriptor in descriptors:
            # print ('.', end='', flush=True)
            # print('---------------')
            # print('des_vector: ', keypoint_descriptor)
            distances_array = []
            # i = 0 # print for debugging
            for centroid in centroids:
                #     distance = distance.euclidean(a,vector)
                # Euclidean computation
                distance = np.linalg.norm(np.array(keypoint_descriptor) - np.array(centroid))
                # print(i, ',  ', distance)
                # i += 1
                #     np.concatenate((distances_array, distance), axis=0)
                #     distances_array.append(distance)
                distances_array.append(distance)
            # print('- feature_vector_length:  ', len(distances_array))
            # print('- distance array: ', distances_array)
            # print('minpost: %i' % distances_array.index(min(distances_array)))
            min_index = distances_array.index(min(distances_array))
            histogram[min_index] += 1
            # print('histogram: ', histogram)
        # print('feature vector under histogram form: ', histogram)
        return histogram

    # MULTICORE PROCESSING
    def extract_feature(self, img_paths_list, centroids, start_point, end_point, core_index, queue):
        if start_point <= len(img_paths_list):
            if end_point <= len(img_paths_list):
                path_list = img_paths_list[start_point:end_point]
            else:
                path_list = img_paths_list[start_point:len(img_paths_list)]
        else:
            path_list = []

        i = start_point
        all_descriptions = []
        for image_path in path_list:
            print(str(i), ', Processing on: ', image_path)
            feature = self.compute_feature_histogram(centroids, image_path)
            all_descriptions.append(feature)

            # store feature to storage
            path_split = re.split('[. \\\]', image_path)
            file_path = "features/" + 'BoW_hist' + '/' + path_split[1] + "/"
            file_name = path_split[2] + '.pkl'

            # # create directory if not exist
            # if not os.path.exists(file_path):
            #     os.makedirs(file_path)
            # # using pickle to save variable to file
            # with open(file_path + file_name, mode='wb') as feature_file:
            #     pickle.dump(feature, feature_file, protocol=pickle.HIGHEST_PROTOCOL)
            # Write to file
            write_to_file(storage_path=file_path, value_object=feature, file_name=file_name)
            # count += 1

            # add to return later
            all_descriptions.append(feature)

            # print("Completed! Extract %s features from %s files." % (count, len(img_paths_list)))
            i += 1
        print("Finish process in core No." + str(core_index) + " !")
        queue.put(all_descriptions)

    # not finished yet
    # # load file pickle
    def extract_feature_vectors_for_image_dataset(self, need_to_return=False):
        print('Loading codebook')
        with open('BoW_codebook/Codebook.pkl', 'rb') as handle:
            centroids = pickle.load(handle)

        # print(centroids)
        print('Number of centroid = ', len(centroids))
        all_descriptors = []

        # # read image paths - TRAINING SET
        # img = LoadData().getImagePath('db1')
        # img_paths_list = [img['trainImage'][0], ]  # read all image path in train images set

        # Read all image in dataset
        img_paths_list = glob.glob('images//*//*.jpg')
        img_paths_list.sort()
        print(len(img_paths_list))

        # count = 0 # use to check histogram dimentions
        # todo: main process for multicore extractor
        # multicores processing
        num_of_cores = 8
        if len(img_paths_list) % num_of_cores:
            distance = int(len(img_paths_list) / num_of_cores) + 1
        else:
            distance = int(len(img_paths_list) / num_of_cores)
        queue = Queue()
        # processes spliting
        similarities = [
            Process(target=self.extract_feature, args=(img_paths_list, centroids, x * distance, (x + 1) * distance, x, queue))
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
        all_descriptions = np.array([], dtype=np.float32).reshape(0, 500)
        for ele in queue_result:
            all_descriptions = np.concatenate((all_descriptions, ele), axis=0)

        self.write_to_file(storage_path='extracted_feature_descriptions/histogram/',
                           file_name='all_histogram_descriptions.pkl',
                           value_object=all_descriptions)

        if need_to_return:
            return all_descriptions

if __name__ == '__main__':
    histogram_feature_extractor = HistogramFeature()
    histogram_feature_extractor.extract_feature_vectors_for_image_dataset()
