import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib
import time
import os


def train(X_train, y_train):
    print("Training svm")
    if not os.path.exists("saved_svm.pkl"):
        start = time.time()

        clf = svm.SVC()
        clf.fit(X_train, y_train)
        joblib.dump(clf, "saved_svm.pkl")

        print("Train time: " + str(time.time() - start))

    clf = joblib.load("saved_svm.pkl")
    return clf


def accuracy(X_test, y_test, clf):
    print("Calculating accuracy")
    prediction = clf.predict(X_test)
    score = prediction == y_test
    return score.sum() / y_test.shape[0]

def hog(img, pix_per_cell = 8, cells_per_block = 2, amount_bins = 9):

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Choose  channel with max magnitude
    h = mag.shape[0]
    w = mag.shape[1]
    max_mag = np.zeros((h, w), dtype=np.float32)
    resp_angle = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            ind = 0
            for k in range(3):
                if mag[i][j][k] > mag[i][j][ind]:
                    ind = k
            max_mag[i][j] = mag[i][j][ind]
            resp_angle[i][j] = angle[i][j][ind]

    bins_ind = np.int32(amount_bins * resp_angle / 360)

    shape_hists = (max_mag.shape[0] // pix_per_cell,
                  max_mag.shape[1] // pix_per_cell,
                  amount_bins)
    hists = np.zeros(shape_hists, dtype=np.float32)

    # Calculate histograms
    for i in range(max_mag.shape[0]):
        for j in range(max_mag.shape[1]):
            hists[i // pix_per_cell][j // pix_per_cell][bins_ind[i][j]] += max_mag[i][j]

    # Vector of features
    features = np.array([], dtype=np.float32)

    #Normolize historgrams and calculate features
    for i in range(hists.shape[0] - (cells_per_block - 1)):
        for j in range(hists.shape[1] - (cells_per_block - 1)):
            norm = (np.sum(np.square(hists[i][j])) + np.sum(np.square(hists[i][j + 1])) +
                    np.sum(np.square(hists[i + 1][j])) + np.sum(np.square(hists[i + 1][j + 1])))

            norm = np.sqrt(norm) + 0.1

            features = np.hstack((features, hists[i][j] / norm))
            features = np.hstack((features, hists[i][j + 1] / norm))
            features = np.hstack((features, hists[i + 1][j] / norm))
            features = np.hstack((features, hists[i + 1][j + 1] / norm))


    return features


def convert_img_to_hog(list_img):
    list_hog = []
    for i, img in enumerate(list_img):
        list_hog.append(hog(img))
    return list_hog

def read_imgs(path):
    list_imgs = []
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        list_imgs.append(img)
    return list_imgs

def get_hogs(path):
    print("Loading vector of features")
    raw_features_file = path + "_features.pkl"
    if not os.path.exists(raw_features_file):
        start = time.time()

        imgs = read_imgs(path)
        data = convert_img_to_hog(imgs)
        data = np.array(data, dtype=np.float32)
        joblib.dump(data, raw_features_file)

        print(path + " features calc time: ", time.time() - start)
    data = joblib.load(raw_features_file)
    return data

def run():
    pos_data = get_hogs("positive")
    pos_target = np.ones(pos_data.shape[0], dtype=np.int32)

    neg_data = get_hogs("negative")
    neg_target = np.zeros(neg_data.shape[0], dtype=np.int32)

    data = np.vstack((pos_data, neg_data))
    target = np.hstack((pos_target, neg_target))


    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    clf = train(X_train, y_train)
    acr = accuracy(X_test, y_test, clf)
    print("accuracy: " + str(acr))


if __name__ == "__main__":
    run()
