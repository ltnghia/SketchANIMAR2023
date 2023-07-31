import os


def cal_top1_acc():
    ground_truth_csv = r"/media/phongquang/SSD2T_3/workspace/Phong/project/VisionLanguage/TextANIMAR/data/SketchANIMAR2023/Train/SketchQuery_GT_Train.csv"
    predict_csv = "/media/phongquang/SSD2T_3/workspace/Phong/project/VisionLanguage/TextANIMAR/source/SketchANIMAR/output_analyse/train/THP_SketchANIMAR2023_Train.csv"

    gt_dict = {}
    with open(ground_truth_csv, "r") as rf:
        lines = rf.readlines()
    for line in lines[1:]:
        line = line.strip()
        model_id, result_id_1 = line.split(";")[:2]
        gt_dict[model_id] = result_id_1

    pd_dict = {}
    with open(predict_csv, "r") as rf:
        lines = rf.readlines()

    for line in lines:
        line = line.strip()
        model_id, result_id_1, result_id_2, result_id_3 = line.split(",")[:4]

        # pd_dict[model_id] = [result_id_1, result_id_2, result_id_3]
        pd_dict[model_id] = [result_id_1, result_id_2]

    total = 0
    total_true = 0
    for model_id in gt_dict:
        gt = gt_dict[model_id]
        pd = pd_dict[model_id]

        if gt in pd:
            print(model_id, pd)
            total_true += 1
        total += 1
    print("Num true: ", total_true)
    print("Total: ", total)
    print("Top 1 accuracy: ", round(1. * total_true / total, 4))


if __name__ == "__main__":
    cal_top1_acc()
