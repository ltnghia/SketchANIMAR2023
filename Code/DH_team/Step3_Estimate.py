import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from pathlib import Path
from tqdm import tqdm


# Measure a sketch image against a set of captured 3D model images and retrieve the image with the highest matching score.
def predict(modelf, sketchf):
    l = [mean_squared_error(i, sketchf) for i in modelf]
    return np.min(l)


# Measure the distance between each sketch image and the set of captured 3D model images.
# The result is a list containing the distances between each sketch and each 3D model.
def main(
    Model_feature_path=r"data/features_generation_models",
    Sketch_feature_path=r"data/features_generation_sketch",
):
    testSketch_files = np.loadtxt(
        r"data/SketchQuery_Test.csv", delimiter=",", dtype=str
    )[1:]
    testSketch_files = [i + ".csv" for i in testSketch_files]

    modelFeature_files = os.listdir(Model_feature_path)

    result = []
    with tqdm(total=len(testSketch_files)) as pbar:
        for s in testSketch_files:
            sketchf = np.loadtxt(
                Sketch_feature_path + "/" + s, delimiter=",", dtype=float
            )
            sketchf = sketchf.reshape((16, 1))

            # Reorder the sorting sequence of vectors instead of flipping the Sketch image vertically.
            sketchf_flip = sketchf[
                [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
            ].reshape((16,))
            sketchf = sketchf.reshape((16,))
            vector = []
            for m in modelFeature_files:
                modelf = np.loadtxt(
                    Model_feature_path + "/" + m, delimiter=",", dtype=float
                )

                # Matching each Sketch image with sets of 3D model capture images.
                # Perform measurements twice on the Sketch image, and then flip the image and measure once more
                # since there are cases where the 3D image is flipped compared to the Sketch image.
                k = min(predict(modelf, sketchf), predict(modelf, sketchf_flip))
                vector.append([Path(m).stem, k])
            result.append([Path(s).stem, vector])
            pbar.update(1)

    # Exprort result into submission.csv file
    import pandas as pd

    report = []
    for i in result:
        # Processing and Sorting Matched Results.
        sket = i[0]
        df = i[1]
        df = pd.DataFrame(df)
        df = df.sort_values(by=[1])
        df = list(df[0])
        df.insert(0, sket)
        report.append(df)

    a = pd.DataFrame(report)
    # result
    a.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
