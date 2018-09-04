import os
import cv2
import time

def crop_out(img, stnd_height,  stnd_weight):
    height, weight, channels = img.shape

    top_indent_h = (height - stnd_height) // 2
    bottom_indent_h = (height - stnd_height - top_indent_h)

    left_indent_w = (weight - stnd_weight) // 2
    right_indent_w = (weight - stnd_weight - left_indent_w)

    cropped_img = img[top_indent_h : height - bottom_indent_h, left_indent_w : weight - right_indent_w]
    return cropped_img


def prepare_positive_dataset():
    SIZE = 240

    path_to_read = "../lfw"
    dir_list = os.listdir(path_to_read)

    path_to_write = "positive"
    if not os.path.exists(path_to_write):
        os.mkdir(path_to_write)

    cnt = 0
    for dir in dir_list:
        path = os.path.join(path_to_read, dir)
        img_name = os.listdir(path)[0]
        img = cv2.imread(os.path.join(path,img_name))
        img = crop_out(img, SIZE, SIZE)

        cv2.imwrite(os.path.join(path_to_write, "pos_" + str(cnt) + ".jpg"), img)
        cnt += 1

def cutting_image(img, stnd_height, stnd_weight):
    h, w, ch = img.shape
    cnt_h = h // stnd_height
    cnt_w = w // stnd_weight

    cutting_images = []

    for i in range(cnt_h):
        for j in range(cnt_w):
            x = i * stnd_height
            y = j * stnd_weight
            cur_img = img[x : x + stnd_height, y : y + stnd_weight]
            cutting_images.append(cur_img)

    return cutting_images

def prepare_negative_dataset():
    path_to_read = "images_from_google"
    list_dir = os.listdir(path_to_read)
    SIZE = 240

    list_img = []
    for dir in list_dir:
        n_path = os.path.join(path_to_read, dir)
        for img_name in os.listdir(n_path):
            img_path = os.path.join(n_path, img_name)
            img = cv2.imread(img_path)

            if  img is None or img.shape[0] < SIZE or img.shape[1] < SIZE:
                os.remove(img_path)
            else:
                list_img.append(img)

    path_to_write = "negative"
    if not os.path.exists(path_to_write):
        os.mkdir(path_to_write)

    cnt = 0
    for img in list_img:
        cutting_images = cutting_image(img, SIZE, SIZE)
        for img in cutting_images:
            cv2.imwrite(os.path.join(path_to_write, "neg_" + str(cnt)) + ".jpg", img)
            cnt += 1

def resize_img(img, size=64):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def resize_dataset(path):
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = resize_img(img)
        cv2.imwrite(img_path, img)


def main():
    prepare_positive_dataset()

    prepare_negative_dataset()

    start = time.time()
    resize_dataset("positive")
    print(time.time() - start)

    start = time.time()
    resize_dataset("negative")
    print(time.time() - start)

if __name__ == "__main__":
    main()

