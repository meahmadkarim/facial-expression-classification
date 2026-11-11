# Facial Expression Classification

An educational image-classification project with a historical Keras CNN notebook and a maintained CPU baseline. The baseline validates face boxes, crops grayscale images and trains a scaled logistic-regression classifier. It was added with AI assistance during portfolio maintenance.

## Run the maintained baseline

Use Python 3.12 and an isolated virtual environment:

```sh
python -m pip install -r requirements.txt
python train.py --images data/images --labels data/label.lst --output artifacts
```

The eight whitespace-separated annotation fields are `image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_confidence expression_label`. Labels must be integers 0..6. Invalid paths, missing images, invalid boxes and label ranges produce an error. The output contains `model.joblib` and `metrics.json`. Load only trusted joblib files.

## Dataset source and status

The annotation format appears to match **Expression in-the-Wild (ExpW)** from [Zhang, Luo, Loy and Tang's official project page](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html). That page links the original archive. The exact provenance of the notebook's missing archive is unconfirmed; obtain the dataset from its authors and check its terms and class mapping before use. Real face data is not fabricated or redistributed in this repository.

The script splits by source image before fitting preprocessing; faces from one image cannot appear in both partitions. Person identities are unavailable, so identity-disjoint evaluation is not guaranteed. There is no oversampling. A single split is only a baseline; inspect class coverage and use a suitable group-stratified evaluation before reporting results.

## Historical notebook

The original notebook uses personal Drive paths and applies SMOTE before splitting, which contaminates evaluation. It is retained as historical work, with a warning and cleared outputs; use `train.py` for the maintained baseline. The CPU baseline does not claim to reproduce the CNN experiment. Real-dataset training and a trained CNN checkpoint remain outstanding.

The system classifies dataset expression labels, not a person's inner emotional state. It is not an identity-recognition or psychological-assessment tool.

## Verification

The maintained baseline completed loading, grouped splitting, fitting, prediction and artifact saving on a generated noise-image fixture. This validates software execution only. No facial-expression accuracy is claimed from synthetic fixtures.
