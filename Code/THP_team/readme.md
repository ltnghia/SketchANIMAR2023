# THP_team

## Environment
- **Linux:** 18.04.5
- **GPU:** NVIDIA GeForce GTX 1080 Ti 11GB
- **CUDA Version:** 10.2
- **Torch Version:** 1.10.0

## Instructions

1. Generate 2D Images from 3D Model
Follow the instructions at [https://github.com/WeiTang114/BlenderPhong](https://github.com/WeiTang114/BlenderPhong) to generate 2D images. Make modifications to the `phong.py` file as necessary. Use Blender version `blender-2.79-linux-glibc219-i686`, available for download from [https://www.blender.org/](https://www.blender.org/). The generated data can be found at [Google Drive](https://drive.google.com/drive/folders/1njdHzfOZxfTJoWndiAM1aQqSdxUz2Io4?usp=sharing).

2. Preprocessing
- Crop and Resize:
    ```bash
    python center_crop_224.py \
    --root path/to/2d_image_folder \
    --output-dir path/to/output \
    --is-sketch
    ```
- Get Canny Edges and Apply Dilation Morphology:
    ```bash
    python get_canny_dilate.py \
    --root path/to/2d_image_folder \
    --output-canny path/to/canny_output \
    --output-canny-dilate path/to/canny_dilate_output \
    --is-sketch
    ```
3. Feature Extractor
Extract and Save CLIP, HOG Features of 3D Model and Sketch Images:
    ```bash
    python feature_extractor.py \
    --model-root path/to/model_dilate_dilate \
    --model-clip-save-dir path/to/output_model_CLIP_feature \
    --model-hog-save-dir path/to/output_model_HOG_feature \
    --sketch-root path/to/sketch_dilate \
    --sketch-clip-save-dir path/to/output_sketch_CLIP_feature \
    --sketch-hog-save-dir path/to/output_sketch_HOG_feature
    ```

4. Retrieval
- Retrieval by CLIP Feature, HOG Feature, and Ranking:
    ```bash
    python sketch_retrieval.py \
    --model-clip-save-dir path/to/model_CLIP_feature \
    --model-hog-save-dir path/to/model_HOG_feature \
    --model-original-dir path/to/original_model \
    --sketch-clip-save-dir path/to/sketch_CLIP_feature \
    --sketch-hog-save-dir path/to/sketch_HOG_feature \
    --sketch-original-dir path/to/original_sketch \
    --result-phase-1-json path/to/output_retrieval_by_CLIP_json \
    --output-vis-dir path/to/visualize_folder
    ```