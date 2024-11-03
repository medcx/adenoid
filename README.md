# Adenoid
This is the codebase for the article "A fully automatic artificial intelligence system for the diagnosis of pathological adenoid hypertrophy based on face images". 

This codebase uses a pre-trained model for face segmentation [Retinaface](https://github.com/serengil/retinaface).

# Adenoid hypertrophy recognition
At the beginning, we need to build an environment.
```
conda create -n adenoid python=3.10
conda activate adenoid
pip install -r requirements.txt
```
**Data preparation**

Since the collected images contain a lot of background information, we need to process the data using retinaface model.

Here are the pre-trained model download link: [Google Drive](https://drive.google.com/file/d/1LLZ2BcPWgeScCjJblN6WIPruyG3vsnfa/view?usp=sharing). Save it to ```retinaface/weights/```.
```
cd retinaface
python extract_frame.py --input_dir [folder of input images] --output_dir [folder of output images]
```
To facilitate model training, we recommend building a folder structure like this:
```
/data
├── normal
│   ├── train
|       ├──sample_1
|          └── 0_degree.jpg
|          └── 1_degree.jpg
|          └── 2_degree.jpg
|       ├──sample_2
|          └── ...
|       ├──...
|   
│   ├── test
|       ├──sample_1
|          └── ...
|       ├──...
|
├── disease
│   ├── train
|       ├──sample_1
|          └── 0_degree.jpg
|          └── 1_degree.jpg
|          └── 2_degree.jpg
|       ├──sample_2
|          └── ...
|       ├──...
|   
│   ├── test
|       ├──sample_1
|          └── ...
|       ├──...
```
**Single-degree model training**

We pre-trained a face recognition model on the FaceForensics++ dataset and used the model weights as initialization to fine-tune our single-degree model.

Here are the pre-trained model download link: [Google Drive](https://drive.google.com/file/d/1Vs-H6Z3wcFGuxQv1eqnOJGl020gdQ4ub/view?usp=sharing). Save it to ```effnet/pretrain_weight/```.

Now, all preparations are ready, we can run the following codes to fine-tuning the single-degree model:
```
cd effnet
python train_single.py --data_path ../data --epoch 200 --batch_size 8 --degree [chosen from (0, 1, 2)]
```
Finally, we get the single degree model with five-fold cross validation.

**Multi-degree model training**

We can run the following codes to train the multi-degree model:
```
python train_multi.py --data_path ../data --epoch 200 --batch_size 8
```

**Multi-degree model validation**

Validation is divided into internal validation and external validation.

For the internal validation, run:
```
python internal_eval.py --data_path ../data --fold all
```
For the external validation, run:
```
python external_eval.py --data_path ../data --fold all
```
We will provide a demo dataset soon to facilitate you to better use this repository.



