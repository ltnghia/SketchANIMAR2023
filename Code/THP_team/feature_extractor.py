import argparse
from skimage.io import imread
from skimage import feature
import glob
import clip
import os
from PIL import Image
import numpy as np
import torch


def get_sketch_feature(sketch_folder_path, sketch_output_dir):
    sketch_paths = glob.glob(sketch_folder_path + "/*")
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    for i, sketch_path in enumerate(sketch_paths):
        sketches = []
        image = Image.open(sketch_path).convert("RGB")
        sketches.append(preprocess(image))
        sketch_input = torch.tensor(np.stack(sketches)).cuda()
        with torch.no_grad():
            features = model.encode_image(sketch_input).float()
            features /= features.norm(dim=-1, keepdim=True)
            # save vision feature
            sketch_id = os.path.basename(sketch_path)[:-4]
            output_path = os.path.join(sketch_output_dir, sketch_id + ".pt")
            torch.save(features, output_path)

            print("Sketch feature (1, 512) saved at: ", output_path)


def get_model_feature(model_folder_path, model_output_dir):
    all_views = ["view_1", "view_2", "view_3", "view_4"]
    all_model_id = os.listdir(os.path.join(model_folder_path, all_views[0]))
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    for model_id in all_model_id:
        all_view_path = [os.path.join(model_folder_path, view, model_id) for view in all_views]
        images = []  # batch size = 48
        for view_path in all_view_path:
            img_paths = glob.glob(view_path + "/*")
            for img_path in img_paths:
                image = Image.open(img_path).convert("RGB")
                images.append(preprocess(image))
        images_input = torch.tensor(np.stack(images)).cuda()
        with torch.no_grad():
            features = model.encode_image(images_input).float()
            features /= features.norm(dim=-1, keepdim=True)

            # save vision feature
            output_path = os.path.join(model_output_dir, model_id + ".pt")
            torch.save(features, output_path)
            print("Model feature (48, 512) saved at: ", output_path)


def sketch_hog(input_folder, feature_folder):
    all_sketch = os.listdir(input_folder)
    for img_name in all_sketch:
        img_path = os.path.join(input_folder, img_name)
        img = imread(img_path, as_gray=True)
        H, _ = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                           visualize=True)

        out_path = os.path.join(feature_folder, img_name[:-4] + ".npy")
        np.save(out_path, H)


def get_model_hog(model_folder_path, model_output_dir):
    all_views = ["view_1", "view_2", "view_3", "view_4"]
    all_model_id = os.listdir(os.path.join(model_folder_path, all_views[0]))
    for m, model_id in enumerate(all_model_id):
        all_view_path = [os.path.join(model_folder_path, view, model_id) for view in all_views]
        all_hog = np.zeros(shape=(48, 26244), dtype=np.float64)  # batch size = 48
        for i, view_path in enumerate(all_view_path):
            img_paths = glob.glob(view_path + "/*")
            for j, img_path in enumerate(img_paths):
                img = imread(img_path, as_gray=True)

                H, _ = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                   visualize=True)
                all_hog[12 * i + j] = H
        out_path = os.path.join(model_output_dir, model_id + ".npy")
        np.save(out_path, all_hog)
        if m % 20 == 0:
            print("Done {} / {}".format(m, len(all_model_id)))


def main(args):
    """ Get sketch CLIP feature """
    get_sketch_feature(args["model-root"], args["model-clip-save-dir"])

    """ Get model CLIP feature """
    get_model_feature(args["sketch-root"], args["sketch-clip-save-dir"])

    """ Get sketch HOG feature """
    sketch_hog(args["model-root"], args["model-hog-save-dir"])

    """ Get model HOG feature """
    get_model_hog(args["sketch-root"], args["sketch-hog-save-dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=str,
                        default='./input_edge_dilate',
                        help='directory store canny dilated image of 3D object model')
    parser.add_argument('--model-clip-save-dir', type=str,
                        default='./output_model_feature',
                        help='directory save 3D model CLIP feature')
    parser.add_argument('--model-hog-save-dir', type=str,
                        default='./output_model_hog_feature',
                        help='directory save 3D model HOG feature')

    parser.add_argument('--sketch-root', type=str,
                        default='./input_sketch_dilate',
                        help='directory store dilated image of sketch image')
    parser.add_argument('--sketch-clip-save-dir', type=str,
                        default='./output_sketch_feature',
                        help='directory save sketch CLIP feature')
    parser.add_argument('--sketch-hog-save-dir', type=str,
                        default='./output_sketch_hog_feature',
                        help='directory save sketch HOG feature')

    args = parser.parse_args()
    main(args)
