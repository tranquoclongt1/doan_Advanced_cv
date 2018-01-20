import glob, os
from sklearn.model_selection import train_test_split

"""
# load feature
with open('features.pkl', 'rb') as feature_file:
    features = pickle.load(feature_file)

features = features.reshape((features.shape[0], features.shape[2]))
"""

# load all image name
images = glob.glob('images//*//*.jpg')
images.sort()
print(len(images))

# set labels for each image
# note: need to change '\\' to '/' if running on linux
# diffenrce between file path in windows and linux
targets = []
for image in images:
    # print (image.split("\\")[1])
    targets.append(image.split("\\")[1])


# split to train/test set


num_db = 3
for i in range(num_db):
    X_train, X_test, Y_train, Y_test = train_test_split(images, targets, test_size=0.2)
    db_path = 'db/db%s'%(i+1)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    with open(db_path+'/train.txt', mode='w') as train_file:
        for file in X_train:
            train_file.write(file)
            train_file.write('\n')

    with open(db_path+'/lbtrain.txt', mode='w') as label_train_file:
        for label in Y_train:
            label_train_file.write(label)
            label_train_file.write('\n')

    with open(db_path+'/test.txt', mode='w') as test_file:
        for file in X_test:
            test_file.write(file)
            test_file.write('\n')

    with open(db_path+'/lbtest.txt', mode='w') as label_test_file:
        for label in Y_test:
            label_test_file.write(label)
            label_test_file.write('\n')

print("Done")

