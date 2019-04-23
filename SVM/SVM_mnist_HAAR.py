import tensorflow as tf
from HAAR_feature import generate_rectangles, each_item, verify_rectangle, reduce_train
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


mnist = tf.keras.datasets.mnist

# get training and testing mnist set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# generate 100 rectangles
rectangles = generate_rectangles()

# verify areas between points
correct = verify_rectangle(rectangles)

# assert if all the generated rectangles have required areas
assert correct, True

# only pick 20% of each class data points from training set
new_train, new_labels = reduce_train(x_train, y_train, data_percent=0.25)

# get haar features
haar = each_item(new_train, rectangles)

# get haar features for test data as well
test_haar = each_item(x_test, rectangles)

with open('./logs/out', 'w') as file_op:
    print('HAAR features extracted!', file=file_op)

    # svm for multi-class classification
    clf = LinearSVC(multi_class='ovr', max_iter=10000, C=0.1, tol=0.01)

    # fit train data with labels
    model = clf.fit(haar, new_labels)

    # predict for training data
    train_predictions = model.predict(haar)

    train_acc = accuracy_score(new_labels, train_predictions)

    print('Train Accuracy: {}'.format(train_acc), file=file_op)

    # predict for test data
    test_predictions = model.predict(test_haar)

    test_acc = accuracy_score(y_test, test_predictions)

    print('Test Accuracy: {}'.format(test_acc), file=file_op)

