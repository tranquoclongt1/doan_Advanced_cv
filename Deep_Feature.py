import glob
import pickle
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image
import re
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications import vgg16, vgg19, resnet50, inception_v3
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from scipy import misc


class ExtractDeepFeatures:

    def extractFeatures(self, image_path, model, base_model):
        """
        Extract deep features from image by model
        """
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = base_model.preprocess_input(x)
        preds = model.predict(x)

        return preds

    def extractListFeatures(self, list_image_path, model_name, return_features=False):
        """
        Extract feature of each image in list_image_path by model_name
        model_name in vgg16, vgg19, inceptionv3, resnet50
        return_features: True if want to return list features, otherwise False.
        """
        b_model = None
        if model_name == 'vgg16' or model_name=='caltech5-vgg16':
            # get layer 'fc2' - 2th full connected layer as output
            b_model = VGG16(weights='imagenet')
            base_model = vgg16
        elif model_name == 'vgg19':
            # get layer
            b_model = VGG19(weights='imagenet')
            base_model = vgg19
        elif model_name == 'resnet50':
            b_model = ResNet50(weights='imagenet')
            base_model = resnet50
        elif model_name == 'inceptionv3':
            b_model = InceptionV3(weights='imagenet')
            base_model = inception_v3

        if b_model is not None:
            model = Model(inputs=b_model.input, outputs=b_model.layers[-2].output)

        count = 1
        features = []
        # Extract feature of each image in List image then store it to features/model_name/...(class folder)
        for image_path in list_image_path:

            print("Completed: %s/%s"%(count, len(list_image_path)), end='\r')

            if model_name == 'raw_pixel':
                img = misc.imread(image_path, mode='RGB')
                img = misc.imresize(img, (64, 48))
                feature = img.reshape(img.shape[0]*img.shape[1]*img.shape[2], 1)
            else:
                feature = self.extractFeatures(image_path, model, base_model)

            # store feature to storage
            path_split = re.split('[. \\\]', image_path)
            file_path = "features\\" + model_name + '\\' + path_split[1] + "\\"
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

            if count == len(list_image_path):
                print("Feature shape: ", feature.shape)

        print("Completed! Extract %s features from %s files." % (count, len(list_image_path)))

        return features


if __name__ == '__main__':
    extract = ExtractDeepFeatures()

    images = glob.glob('caltech5\*\*.jpg')

    extract.extractListFeatures(images, 'caltech5-vgg16')








"""

b_model = InceptionV3(weights = None)
#b_model = VGG16(weights='imagenet')
b_model.load_weights('E:\Dataset-20171030T085442Z-001\inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

b_model.summary()
model = Model(inputs= b_model.input, outputs=b_model.layers[-1].output)


#model = InceptionV3(weights=None, include_top=True)
#model.load_weights('E:\Dataset-20171030T085442Z-001\inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
#model.summary()

images = glob.glob('images\*\*.png')
images.sort()
count = 0
features = []
for image_path in images:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    count = count + 1
    print(count)
    #targets.append(image.split("\\.")

    # save feature to file
    path_split = re.split('[. \\\]', image_path)
    file_path = "features\\" + 'inceptionV3\\' + path_split[1] + "\\"
    file_name = path_split[2] + '.pkl'

    # create directory if not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_path + file_name, mode='wb') as feature_file:
        pickle.dump(preds, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

    features.append(preds)

features = np.array(features)

with open('features.pkl', mode='wb') as feature_file:
    pickle.dump(features, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

print('Predicted:', decode_predictions(preds, top=3)[0])
print("feature type: ", type(preds))
print("feature shape: ", preds.shape)
print(preds)


fc2_features = vgg16_model.predict(X)
fc2_features = vgg16_model.reshape((4096,1))

"""