import argparse
import os
import cv2


def canny(root, root_output):
    t_lower = 150  # Lower Threshold
    t_upper = 200  # Upper threshold

    all_folder = os.listdir(root)
    for folder_name in all_folder:
        new_folder = os.path.join(root_output, folder_name)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        folder_path = os.path.join(root, folder_name)
        all_img = os.listdir(folder_path)

        for img_name in all_img:
            img_path = os.path.join(root, folder_name, img_name)
            img = cv2.imread(img_path, 0)

            edges = cv2.Canny(img, t_lower, t_upper)
            edges = 255 - edges

            # cv2.imshow("edges", edges)
            # cv2.waitKey(0)
            new_path = os.path.join(new_folder, img_name)
            cv2.imwrite(new_path, edges)
            print("Converted :", new_path)


def dilate_canny(root, root_output):
    all_folder = os.listdir(root)
    for folder_name in all_folder:
        new_folder = os.path.join(root_output, folder_name)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        folder_path = os.path.join(root, folder_name)
        all_img = os.listdir(folder_path)

        for img_name in all_img:
            img_path = os.path.join(root, folder_name, img_name)
            img = 255 - cv2.imread(img_path, 0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.dilate(img, kernel)

            # cv2.imshow("edges", edges)
            # cv2.waitKey(0)
            new_path = os.path.join(new_folder, img_name)
            cv2.imwrite(new_path, 255 - result)
            print("Converted :", new_path)


def dilate_sketch(root, root_output):
    all_img = os.listdir(root)

    for img_name in all_img:
        img_path = os.path.join(root, img_name)
        img = 255 - cv2.imread(img_path, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(img, kernel)

        new_path = os.path.join(root_output, img_name)
        cv2.imwrite(new_path, 255 - result)
        print("Converted :", new_path)


def main(args):
    if args["is-sketch"]:
        dilate_sketch(args["root"], args["output-canny-dilate"])
    else:
        canny(args["root"], args["output-canny"])
        dilate_canny(args["output-canny"], args["output-canny-dilate"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='./input_edge/view_1',
                        help='path to input directory of 2D projected images')
    parser.add_argument('--output-canny', type=str,
                        default='./output_canny/view_1',
                        help='output directory for canny edge image')
    parser.add_argument('--output-canny-dilate', type=str,
                        default='./output_canny_dilate/view_1',
                        help='output directory for canny edge image applied dilation')
    parser.add_argument('--is-sketch', action='store_true', help='process for sketch or model')
    args = parser.parse_args()
    main(args)
