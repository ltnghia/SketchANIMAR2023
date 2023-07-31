# V1olet's inference script for SHREC23 Sketch-Based 3D Shape Retrieval ANIMAR Challenge.

1. Create a conda environment with name animar and activate it.

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the script to download the model weights and the dataset:
```
bash run.sh
```
- The trained models are saved in the download/ folder.
- The original test set is saved in unzip/SketchQuery_Test folder.
- The preprocessed dataset is saved in the clean_dataset/ folder.

4. Run predict script to generate the predictions which is saved as submit.csv:
```
python predict.py
```