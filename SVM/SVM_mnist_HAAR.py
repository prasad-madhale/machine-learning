import tensorflow as tf
from HAAR_feature import generate_rectangles, each_item, verify_rectangle, reduce_train
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np


mnist = tf.keras.datasets.mnist

# get training and testing mnist set
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# GENERATE AND STORE HAAR FEATURES

# # generate 100 rectangles
# rectangles = generate_rectangles()
#
# # verify areas between points
# correct = verify_rectangle(rectangles)
#
# # assert if all the generated rectangles have required areas
# assert correct, True
#
# # only pick 20% of each class data points from training set
# new_train, new_labels = reduce_train(x_train, y_train, data_percent=0.25)
#
# # save HAAR features to files
# np.savez('./data/train_labels', train_labels=new_labels)
#
# # get haar features
# haar = each_item(new_train, rectangles)
#
# np.savez('./data/train_haar_features', train_haar=haar)
#
# # get haar features for test data as well
# test_haar = each_item(x_test, rectangles)
#
# np.savez('./data/test_haar_features', test_haar=test_haar)

# LOAD HAAR FEATURES from npz files

train_haar = np.load(file='./data/train_haar_features.npz')['train_haar']
test_haar = np.load(file='./data/test_haar_features.npz')['test_haar']
train_labels = np.load(file='./data/train_labels.npz')['train_labels']
test_labels = y_test


with open('./logs/out_svm_haar', 'w') as file_op:
    print('HAAR features loaded!', file=file_op)
    print('SHAPES:', file=file_op)
    print(train_haar.shape, test_haar.shape, train_labels.shape, test_labels.shape, file=file_op)

    # svm for multi-class classification
    # clf = LinearSVC(multi_class='ovr', max_iter=10000, C=0.1, tol=0.01)

    clf = svm.SVC(kernel='linear')

    # fit train data with labels
    model = clf.fit(train_haar, train_labels)

    # predict for training data
    train_predictions = model.predict(train_haar)

    train_acc = accuracy_score(train_labels, train_predictions)

    print('Train Accuracy: {}'.format(train_acc), file=file_op)

    # predict for test data
    test_predictions = model.predict(test_haar)

    test_acc = accuracy_score(test_labels, test_predictions)

    print('Test Accuracy: {}'.format(test_acc), file=file_op)

print('Done!')
