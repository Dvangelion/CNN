import numpy as np
#import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from sklearn import decomposition
from skimage.transform import rotate
import augmenters as iaa
import imgaug as ia



def trainfileName(k):
	k_str = '' + `k`
	while len(k_str) < 5:
		k_str = '0' + k_str
	return './train_scale/' + k_str + '.jpg'

def testfileName(k):
	k_str = '' + `k`
	while len(k_str) < 5:
		k_str = '0' + k_str
	return './val/' + k_str + '.jpg';

def transformTrainData():
	train_data = np.zeros((7000, 227, 227, 3))
	for i in range(7000):
		img = plt.imread(trainfileName(i+1)).astype(np.float32) / 255


		#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#gray = rgb2gray(img)
		train_data[i,...] = img
	np.save("train_data.npy", train_data)

def transformTrainLabel():
	train_label = np.genfromtxt('train.csv', delimiter=',')[1:,1]
	np.save("train_label.npy", train_label)

def transformTestData():
	test_data = np.zeros((970, 128, 128))
	for i in range(970):
		img = cv2.imread(testfileName(i+1)).astype(np.float32) / 255
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		test_data[i,...] = gray
	np.save("test_data.npy", test_data)

# def transformTrainLabel():
# 	test_label = np.genfromtxt('test.csv', delimiter=',')[1:,1]
# 	np.save("test_label.npy", test_label)

def transformTrainData_SIFT():
	train_label_sift = []
	train_id_sift = []

	sift = cv2.xfeatures2d.SIFT_create()

	train_label = np.genfromtxt('train.csv', delimiter=',')[1:,1]

	img = cv2.imread(trainfileName(1))
	kp, des = sift.detectAndCompute(img, None)
	train_data_sift = des
	train_label_sift += [train_label[0]] * des.shape[0]
	train_id_sift += [0] * des.shape[0]

	for i in range(1,7000):
		img = cv2.imread(trainfileName(i + 1))

		kp, des = sift.detectAndCompute(img,None)

		if (des is not None):

			train_data_sift = np.vstack((train_data_sift, des))
			train_label_sift += [train_label[i]] * des.shape[0]
			train_id_sift += [i] * des.shape[0]

	np.save("train_data_sift.npy", train_data_sift)
	np.save("train_label_sift.npy", train_label_sift)
	np.save("train_id_sift.npy", train_id_sift)

def transformTestData_SIFT():
	test_id_sift = []

	sift = cv2.xfeatures2d.SIFT_create()

	img = cv2.imread(testfileName(1))
	kp, des = sift.detectAndCompute(img,None)

	test_data_sift = des
	test_id_sift += [0] * des.shape[0]

	for i in range(1,970):
		img = cv2.imread(testfileName(i+1))
		kp, des = sift.detectAndCompute(img,None)

		if (des is not None):
			test_data_sift = np.vstack((test_data_sift, des))
			test_id_sift += [i] * des.shape[0]

	np.save("test_data_sift.npy", test_data_sift)
	np.save("test_id_sift.npy", test_id_sift)

def pca():
	#pca = PCA()
	pca = decomposition.RandomizedPCA(n_components=150, whiten=True)


	# Input
	train_data = np.load('train_data.npy')
	train_data = train_data.reshape(7000,128*128)
	#test_data = np.load('test_data.npy')

	#train_data = train_data.reshape(train_data.shape[0], -1)
	#test_data = test_data.reshape(test_data.shape[0], -1)


	pca.fit(train_data)
	X_train_pca = pca.transform(train_data)
	#train_data = pca.fit_transform(train_data)
	#test_data = pca.transform(test_data)

	np.save("train_data_pca", X_train_pca)
	#np.save("test_data_pca", test_data)


def random_rotation():
	data = np.load('train_data.npy')[:1000]
	rotation = np.zeros((1000,227,227,3))
	for i in range(1000):
		angle = 90*np.random.randint(1,4)
		image = data[i]
		rotation[i,...] = rotate(image,angle)

	np.save('rotate.npy',rotation)

#transformTrainData()
#pca()

def augment_data():
	aug_data = np.zeros((4000, 227, 227, 3))

	st = lambda aug: iaa.Sometimes(0.5, aug)


	seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
        st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            #order=ia.ALL, # use any of scikit-image's interpolation methods
            cval=(0, 1.0), # if mode is constant, use a cval between 0 and 1.0
            #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
    ],
    random_order=True # do all of the above in random order
)



	for i in range(4000):
		img = plt.imread(trainfileName(i+1))
		image_aug = seq.augment_image(img)




		#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#gray = rgb2gray(img)
		aug_data[i,...] = image_aug.astype(float)/255

	np.save("aug_data_1.npy", aug_data)

#augment_data()