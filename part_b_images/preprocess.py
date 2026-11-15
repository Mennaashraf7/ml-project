import os

import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io, transform

IMG_SIZE = (64, 64, 3)


def load_data(path):
    images = []
    labels = []

    if len(os.listdir(path)) == 1:
        path = os.path.join(path, os.listdir(path)[0])

    for class_name in sorted(os.listdir(path)):
        class_folder = os.path.join(path, class_name)

        if not os.path.isdir(class_folder):
            continue

        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)

            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image = io.imread(file_path)

            image = transform.resize(image, IMG_SIZE)

            image = image.flatten()

            images.append(image)
            labels.append(class_name)

    evidence = np.array(images)
    labels = np.array(labels)

    return evidence, labels


def split_data(evidence, labels):
    x_train, x_temp, y_train, y_temp = train_test_split(
        evidence, labels, test_size=0.30, stratify=labels, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
