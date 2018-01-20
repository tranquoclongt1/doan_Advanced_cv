import cv2

sift = cv2.xfeatures2d.SIFT_create(nfeatures=15, nOctaveLayers=6,
                                   contrastThreshold=0.08, edgeThreshold=15,
                                   sigma=1.6)
# sift = cv2.xfeatures2d.SIFT_create(nfeatures=15, nOctaveLayers=3,
#                                    contrastThreshold=0.04, edgeThreshold=10,
#                                    sigma=1.6)

image_path = 'images/faces/image_0001.jpg'
image = cv2.imread(image_path)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
key_points, descriptors = sift.detectAndCompute(gray_img, None)
print(len(key_points))
