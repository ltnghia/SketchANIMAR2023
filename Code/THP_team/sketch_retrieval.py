import argparse
import json
import shutil
import glob
import clip
import os
from PIL import Image
import numpy as np
import torch


def sort_func(e):
    return e[1]


@torch.no_grad()
def sketch_image_retrieval(model_path, sketch_path, output_vis_dir,
                           sketch_original_dir, model_original_dir, result_phase_1_path):
    output_phase_1 = {}
    num_side = 48
    num_view = 4
    num_max = 4
    top_k = 711  # number of candidates
    all_model_path = glob.glob(model_path + "/*")
    all_sketch_path = glob.glob(sketch_path + "/*")

    # load sketch feature
    num_sketch_feature = len(all_sketch_path)
    all_sketch_feature = torch.zeros((num_sketch_feature, 512))
    for i, sketch_path in enumerate(all_sketch_path):
        sketch_feature = torch.load(sketch_path)
        all_sketch_feature[i] = sketch_feature

    # load model feature
    num_model_feature = len(all_model_path)
    all_model_feature = torch.zeros((num_model_feature * num_side, 512))
    for i, model_path in enumerate(all_model_path):
        model_feature = torch.load(model_path)
        all_model_feature[i * num_side: (i + 1) * num_side] = model_feature

    # get similarity for each sketch
    all_model_feature = all_model_feature.cpu().numpy()
    for i, sketch_feature in enumerate(all_sketch_feature):
        final_score = np.zeros(shape=(len(all_model_path),), dtype=np.float64)
        sketch_feature = np.expand_dims(sketch_feature.cpu().numpy(), 0)
        similarity = sketch_feature @ all_model_feature.T
        similarity = similarity.reshape(num_model_feature, num_view, -1)  # shape = (711, 4, 12)

        # get top 4
        for j, multi_views in enumerate(similarity):
            multi_view_score = np.zeros(shape=(4,), dtype=np.float64)
            for k, view in enumerate(multi_views):
                top_4_index = np.argpartition(view, -num_max)[-num_max:]
                top_4_max = view[top_4_index]
                view_score = 1. * np.sum(top_4_max) / num_max
                multi_view_score[k] = view_score
            model_score = np.max(multi_view_score)
            final_score[j] = model_score  # shape = (711,)

        # get top k candidates
        top_k_score_index = np.argpartition(final_score, -top_k)[-top_k:]
        top_k_score = final_score[top_k_score_index]

        index_value = list(np.stack((top_k_score_index, top_k_score), axis=1))
        index_value.sort(key=sort_func, reverse=True)
        index_value = np.array(index_value)  # shape=(100,)

        # save candidate with sketch in the same folder for visualize
        top_k_score_index = index_value[:, 0].astype(np.int32)
        top_k_score = index_value[:, 1]

        sketch_feature_path = all_sketch_path[i]
        sketch_name = os.path.basename(sketch_feature_path)
        sketch_id = sketch_name.split(".")[0]
        output_phase_1[sketch_id] = []

        sketch_path = os.path.join(sketch_original_dir, sketch_id + ".jpg")
        output_folder = os.path.join(output_vis_dir, sketch_id)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        new_sketch_path = os.path.join(output_folder, "999.jpg")
        shutil.copyfile(sketch_path, new_sketch_path)
        for j, ik in enumerate(top_k_score_index):
            model_id = os.path.basename(all_model_path[ik]).split(".")[0]
            output_phase_1[sketch_id].append(model_id)

            # visualize 50 model
            if j < 50:
                model_folder = os.path.join(model_original_dir, "view_1", model_id)
                img_path = glob.glob(model_folder + "/*")[0]
                new_img_path = os.path.join(output_folder, str(round(top_k_score[j], 8)) + ".png")
                # new_img_path = os.path.join(output_folder, model_id + ".png")
                shutil.copyfile(img_path, new_img_path)

        if i % 20 == 0:
            print("Done {} / {}".format(i, num_sketch_feature))

    with open(result_phase_1_path, 'w') as jsonfile:
        json.dump(output_phase_1, jsonfile)


def load_model_hog(feature_folder, candidate_models, num_views=48):
    output_feature = np.zeros(shape=(len(candidate_models) * num_views, 26244))
    for i, model_id in enumerate(candidate_models):
        npy_path = os.path.join(feature_folder, model_id + ".npy")
        npy_feature = np.load(npy_path, allow_pickle=True)
        output_feature[num_views * i: num_views * (i + 1)] = npy_feature
    return output_feature


def load_sketch_hog(feature_folder, sketch_id):
    npy_path = os.path.join(feature_folder, sketch_id + ".npy")
    npy_feature = np.load(npy_path, allow_pickle=True)
    return npy_feature


def cal_l2_loss(point1, point2):
    return np.linalg.norm(point1 - point2, axis=-1)


def get_min(arr, num_item, sub_len):
    result = np.zeros(shape=(num_item,))
    for i in range(num_item):
        sub_array = arr[sub_len * i: sub_len * (i + 1)]
        result[i] = np.min(sub_array)
    return result


def hog_retrieval_re_ranking(result_phase_1_json, model_folder, sketch_folder,
                             sketch_original_dir, model_original_dir, output_vis):
    num_views = 48

    with open(result_phase_1_json, "r") as f:
        candidates = json.load(f)

    k = 50
    cnt = 0
    clip_factor = 0.7
    write_str = ""
    for sketch_id in candidates:
        sketch_feature = load_sketch_hog(sketch_folder, sketch_id)  # shape = (1, 26244)
        query = np.expand_dims(sketch_feature, axis=0)
        candidate_models = candidates[sketch_id][:k]
        candidate_rank = np.arange(1, k + 1, 1)[::-1].tolist()
        candidate_score = dict(zip(candidate_models, candidate_rank))
        candidate_feature = load_model_hog(model_folder, candidate_models, num_views)  # shape=(k * 48, 26244)

        loss = cal_l2_loss(query, candidate_feature)
        loss = get_min(loss, num_item=len(candidate_models), sub_len=num_views)

        # get top k min
        index_gen = np.arange(len(candidate_models))
        index_value = list(np.stack((index_gen, loss), axis=1))
        index_value.sort(key=sort_func, reverse=False)
        index_value = np.array(index_value[:k])

        top_k_score_index = index_value[:, 0].astype(np.int32)
        top_k_score = index_value[:, 1]

        # visualize sketch
        sketch_path = os.path.join(sketch_original_dir, sketch_id + ".jpg")
        output_folder = os.path.join(output_vis, sketch_id)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        new_sketch_path = os.path.join(output_folder, "99999.jpg")
        shutil.copyfile(sketch_path, new_sketch_path)

        # visualize model
        hog_score = k
        final_result = []
        write_str = write_str + sketch_id
        for j, ik in enumerate(top_k_score_index):
            model_id = candidate_models[ik]
            clip_score = candidate_score[model_id]
            candidate_score[model_id] = clip_factor * clip_score + (1 - clip_factor) * hog_score
            hog_score -= 1
            final_result.append([model_id, candidate_score[model_id]])

        for m, model_id in enumerate(candidate_score):
            model_path = os.path.join(model_original_dir, "view_1", model_id)
            img_path = glob.glob(model_path + "/*")[0]

            new_img_path = os.path.join(output_folder,
                                        str(int(candidate_score[model_id])).zfill(5) + "_" + str(m) + ".png")
            shutil.copyfile(img_path, new_img_path)

        final_result.sort(key=sort_func, reverse=True)
        for rs in final_result:
            write_str = write_str + "," + rs[0]

        candidate_remain = candidates[sketch_id][k:]
        for candidate_id in candidate_remain:
            write_str = write_str + "," + candidate_id
        write_str = write_str + "\n"

        if cnt % 20 == 0:
            print("Done {} / {}".format(cnt, 66))
        cnt += 1

    output_folder = r"./output_analyse"
    csv_output_path = os.path.join(output_folder, "THP_SketchANIMAR2023_Test.csv")
    with open(csv_output_path, "w") as wf:
        wf.write(write_str)
    print("Result saved at: ", csv_output_path)


def main(args):
    """ phase 1 retrieval base on CLIP feature"""
    sketch_image_retrieval(args["model-clip-save-dir"],
                           args["sketch-clip-save-dir"],
                           args["output-vis-dir"],
                           args["sketch-original-dir"],
                           args["model-original-dir"],
                           args["result-phase-1-json"])

    """ phase 2 base on HOG feature + re ranking """
    hog_retrieval_re_ranking(args["result-phase-1-json"],
                             args["model-hog-save-dir"],
                             args["sketch-hog-save-dir"],
                             args["sketch-original-dir"],
                             args["model-original-dir"],
                             args["output-vis-dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-clip-save-dir', type=str,
                        default='./output_model_feature',
                        help='directory save 3D model CLIP feature')
    parser.add_argument('--model-hog-save-dir', type=str,
                        default='./output_model_hog_feature',
                        help='directory save 3D model HOG feature')
    parser.add_argument('--model-original-dir', type=str,
                        default='./model_original_dir',
                        help='directory store canny dilated image of 3D object model')

    parser.add_argument('--sketch-clip-save-dir', type=str,
                        default='./output_sketch_feature',
                        help='directory save sketch CLIP feature')
    parser.add_argument('--sketch-hog-save-dir', type=str,
                        default='./output_sketch_hog_feature',
                        help='directory save sketch HOG feature')
    parser.add_argument('--sketch-original-dir', type=str,
                        default='./input_sketch_dilate',
                        help='directory save dilate sketch image')

    parser.add_argument('--result-phase-1-json', type=str,
                        default='./result_retrival/phase_1.json',
                        help='json file save output retrieval based on CLIP feature')

    parser.add_argument('--output-vis-dir', type=str,
                        default='./output_vis_dir',
                        help='directory save visualize image')

    args = parser.parse_args()
    main(args)
