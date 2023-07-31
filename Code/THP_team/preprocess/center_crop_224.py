import argparse
import os
import cv2
import numpy as np


def crop(img_gray, img_color, img_size, padding=5):
    padding = 2 * padding  # 5 + 5
    output_img = np.zeros((img_size, img_size, 3), dtype=np.uint8) + 255
    x_min = y_min = x_max = y_max = 0
    h, w = img_gray.shape[:2]

    h_sum = np.sum(img_gray, axis=1)
    w_sum = np.sum(img_gray, axis=0)

    for i in range(0, h):
        if h_sum[i] > 0:
            y_min = i
            break

    for i in range(h - 1, -1, -1):
        if h_sum[i] > 0:
            y_max = i
            break

    for i in range(0, w):
        if w_sum[i] > 0:
            x_min = i
            break

    for i in range(w - 1, -1, -1):
        if w_sum[i] > 0:
            x_max = i
            break

    crop_img = img_color[y_min:y_max, x_min:x_max]
    new_size = img_size - padding

    crop_h, crop_w = crop_img.shape[:2]
    h_w_scale = 1. * crop_h / crop_w

    if h_w_scale > 1:
        new_h = new_size
        new_w = int(new_h / h_w_scale)
    else:
        new_w = new_size
        new_h = int(new_w * h_w_scale)

    resize_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    y_min_map = int(img_size / 2. - new_h / 2.)
    x_min_map = int(img_size / 2. - new_w / 2.)
    output_img[y_min_map:y_min_map + new_h, x_min_map:x_min_map + new_w] = resize_img

    return output_img


def crop_224_model(root, root_output):
    all_folder = os.listdir(root)
    for folder_name in all_folder:
        new_folder = os.path.join(root_output, folder_name)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        folder_path = os.path.join(root, folder_name)
        all_img = os.listdir(folder_path)

        for img_name in all_img:
            img_path = os.path.join(root, folder_name, img_name)
            img_color = cv2.imread(img_path)
            img_gray = 255 - cv2.imread(img_path, 0)
            new_arr = crop(img_gray, img_color, 224)

            new_path = os.path.join(new_folder, img_name)
            cv2.imwrite(new_path, new_arr)
            print("Converted :", new_path)


def crop_224_sketch(root, root_output):
    all_img = os.listdir(root)

    for img_name in all_img:
        img_path = os.path.join(root, img_name)
        img_color = cv2.imread(img_path)
        img_gray = 255 - cv2.imread(img_path, 0)
        new_arr = crop(img_gray, img_color, 224)

        new_path = os.path.join(root_output, img_name)
        cv2.imwrite(new_path, new_arr)
        print("Converted :", new_path)


def main(args):
    if args["is-sketch"]:
        crop_224_sketch(args["root"], args["output-dir"])
    else:
        crop_224_model(args["root"], args["output-dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='./input_edge/view_1',
                        help='path to input directory')
    parser.add_argument('--output-dir', type=str,
                        default='./output_edge/view_1',
                        help='output directory')
    parser.add_argument('--is-sketch', action='store_true', help='process for sketch or model')
    args = parser.parse_args()
    main(args)
