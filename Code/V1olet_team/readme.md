# V1olet_team

## Training

1. Go to the train folder and create a conda environment with name animar and activate it.

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script, make sure to change the path to the root folder in the script:
    ```bash
    bash run.sh
    ```
- The trained models is saved in the same folder with .h5 extension.
- Note: the original models were trained with batch size 128, but for proof of concept, we trained with batch size 1. You can change the batch size in setting.yaml.

## Inference

1. Go to the inference folder and create/activate a conda environment with name "animar".

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script to download the model weights and the dataset:
    ```bash
    bash run.sh
    ```
    - The trained models are saved in the download/ folder.
    - The original test set is saved in unzip/SketchQuery_Test folder.
    - The preprocessed dataset is saved in the clean_dataset/ folder.

4. Run predict script to generate the predictions which is saved as submit.csv:
    ```bash
    python predict.py
    ```