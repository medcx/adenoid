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
Since the collected images contain a lot of background information, we need to process the data using retinaface model.
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
|       ├──...
|   
│   ├── test
|       ├──sample_1
|          └── 0_degree.jpg
|          └── 1_degree.jpg
|          └── 2_degree.jpg
|       ├──sample_2
|       ├──...
|
├── disease
│   ├── train
|       ├──sample_1
|          └── 0_degree.jpg
|          └── 1_degree.jpg
|          └── 2_degree.jpg
|       ├──sample_2
|       ├──...
|   
│   ├── test
|       ├──sample_1
|          └── 0_degree.jpg
|          └── 1_degree.jpg
|          └── 2_degree.jpg
|       ├──sample_2
|       ├──...
```
