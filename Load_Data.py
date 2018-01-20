import re
import pickle
import numpy as np


class LoadData:

    def loadFeatures(self, filename, model_name):

        # open file
        with open(filename, mode='r') as file:
            # get image path
            imagePaths = file.readlines()

            features = []
            # path: images\glass_cups\72.png
            # feature path: features\glass_cups\72.pkl
            count = 1
            for path in imagePaths:
                print("Completed: %s/%s" % (count, len(imagePaths)), end='\r')
                count += 1
                path = path.strip('\n')
                pathSplit = re.split('[. \\\]', path)

                featurePath = 'features\\' + model_name + '\\' + pathSplit[1] + '\\' + pathSplit[2] + '.pkl'
                with open(featurePath, mode='rb') as featureFile:
                    features.append(pickle.load(featureFile))
            features = np.array(features)
            print(features.shape)
            features = features.reshape(features.shape[0], features.shape[2])
            return features

    def loadLabels(self, filename):
        # open file
        with open(filename, mode='r') as file:
            # get image path
            labels = file.readlines()
            labels = [x.strip('\n') for x in labels]
            return np.array(labels)

    def loadData(self, data="db1", model='vgg16'):

        # load file train.txt

        # features
        trainImages = "db\\" + data + "\\train.txt"
        train_X = self.loadFeatures(trainImages, model)

        # labels
        trainLabels = "db\\" + data + "\\lbtrain.txt"
        train_Y = self.loadLabels(trainLabels)


        # load file test.txt
        testImages = "db\\" + data + "\\test.txt"
        test_X = self.loadFeatures(testImages, model)

        testLabels = "db\\" + data + "\\lbtest.txt"
        test_Y = self.loadLabels(testLabels)

        return train_X, test_X, train_Y, test_Y

    def getImagePath(self, data='db1'):

        # load file train.txt
        images = {}
        # features
        trainImages = "db\\" + data + "\\train.txt"
        with open(trainImages, mode='r') as file:
            # get image path
            imagePaths = file.readlines()
            imagePaths = [x.strip('\n') for x in imagePaths]
            images['trainImage'] = imagePaths

        # load file test.txt
        testImages = "db\\" + data + "\\test.txt"
        with open(testImages, mode='r') as file:
            # get image path
            imagePaths = file.readlines()
            imagePaths = [x.strip('\n') for x in imagePaths]
            images['testImage'] = imagePaths

        return images
