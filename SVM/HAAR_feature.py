import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

# get training and testing mnist set
(x_train, y_train),(x_test, y_test) = mnist.load_data()


def generate_rectangles(min_area=130, max_area=170, num_points=100):

    points = []

    for first in range(28):
        for second_hor in range(28):
            if len(points) >= (num_points // 2):
                break

            lent = abs(second_hor - first)
            area = lent * 28

            if min_area <= area <= max_area:
                points.append([(first, 0), (second_hor, 27)])

        for second_ver in range(28):
            if len(points) >= num_points:
                break

            lent = abs(second_ver - first)
            area = lent * 28

            if min_area <= area <= max_area:
                points.append([(0, second_ver), (27, first)])

    return np.array(points)


def verify_rectangle(rectangles, min_area=130, max_area=170):

    for r in rectangles:
        first = r[0]
        second = r[1]

        x_diff = abs(first[0] - second[0])
        y_diff = abs(first[1] - second[1])

        area = x_diff * y_diff

        if min_area > area or area > max_area:
            return False

    return True


def compute_black_vals(image):

    black = np.array([[0 for _ in range(28)] for _ in range(28)])

    for r in range(28):
        for c in range(28):

            if r == 0 and c == 0:
                black[r][c] = image[r][c]
            elif r == 0:
                black[r][c] = black[r][c - 1] + image[r][c]
            elif c == 0:
                black[r][c] = black[r - 1][c] + image[r][c]
            else:
                black[r][c] = black[r][c-1] + black[r-1][c] - black[r-1][c-1] + image[r][c]

    return black


def compute_black_rectangle(rec, black):
    first = rec[0]
    second = rec[1]

    top = black[second[0]][first[1]]
    left = black[first[0]][second[1]]
    entire_img = black[27][27]

    rec_black = entire_img - top - left + black[first[0]][first[1]]

    return rec_black


def each_item(train, recs):

    haar_features = []

    for i, image in enumerate(train):
        # black values
        black = compute_black_vals(image)

        haar_rec = []

        for j, r in enumerate(recs):
            hor = get_hor_half(r)
            ver = get_ver_half(r)

            hor_feature = split_black_val(hor, black)
            ver_feature = split_black_val(ver, black)

            haar_rec.append(hor_feature)
            haar_rec.append(ver_feature)

        haar_features.append(haar_rec)

    return np.array(haar_features)


def split_black_val(splitted, black):
    first = splitted[0]
    second = splitted[1]

    first_val = compute_black_rectangle(first, black)
    second_val = compute_black_rectangle(second, black)

    return first_val - second_val


def get_ver_half(rec):

    first = rec[0]
    second = rec[1]

    x_len = abs(first[0] - second[0]) // 2

    left = [(first[0], first[1]), (second[0] - x_len, second[1])]
    right = [(min(first[0] + x_len, 27), first[1]), (second[0], second[1])]

    return [left, right]


def get_hor_half(rec):

    first = rec[0]
    second = rec[1]

    y_len = abs(first[1] - second[1]) // 2

    # top half
    top = [(first[0], first[1]), (second[0], second[1] - y_len)]
    bottom = [(first[0], min(first[1] + y_len, 27)), (second[0], second[1])]

    return [top, bottom]


def reduce_train(x_train, y_train, data_percent=0.20):
    new_train = np.empty(shape=(1, 28, 28))
    new_labels = np.array([])

    for label in range(10):
        indices = np.where(y_train == label)[0]

        # number of values to pick
        new_size = int(data_percent * len(indices))

        # 20% values
        picks = np.random.choice(indices, new_size)

        label_reduced = x_train[picks]

        new_train = np.concatenate([new_train, label_reduced])

        r_labels = y_train[picks]

        new_labels = np.concatenate([new_labels, r_labels])

    return np.array(new_train[1:]), np.array(new_labels)


def generate_code_matrix(num_features=50):
    rand_mat = np.random.randint(2, size=num_features)
    coding_matrix = []
    coding_matrix.append(rand_mat)
    coding_matrix.append(np.zeros((num_features,)))

    for i in range(1, 9):
        rand_mat = np.roll(rand_mat, i)
        coding_matrix.append(rand_mat)

    return np.array(coding_matrix)



