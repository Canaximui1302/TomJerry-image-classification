# Image classifier for Tom and Jerry frames

## Content

This project implements a toy model using AlexNet architecture for image classification. The model is defined and trained from scratch for learning purpose.
The file for testing model's prediction is `src/test.py` and the training loop is defined in `src/train.py`.

The repository includes:

`TomJerry-image-classification/`
- `src/` — model architecture, training and evaluation scripts  
- `data/` — dataset and annotation files  
- `test_img/` — custom images for inference testing  
- `notebook.ipynb` — experimentation and visualization  
- `environment.yml` — Conda environment configuration  
- `README.md` — project documentation  
- `LICENSE` — license information  


## Data

The dataset is a folder containing frames cut from the animated series Tom and Jerry. Training dataset can be retrieved from Kaggle:

https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

In order to train the model, you need to download the dataset and organize the folders as below:
```
TomJerry-image-classification/
│
├── README.md
├── LICENSE
├── environment.yml
├── notebook.ipynb
│
├── src/
│
├── test_img/
│
└── data/
    ├── tom_jerry_dataset/
    │   └── tom_and_jerry/
    │       ├── jerry/
    │       ├── tom/
    │       ├── tom_jerry_0/
    │       └── tom_jerry_1/
    │
    ├── challenges.csv
    └── ground_truth.csv

```


## Environment Setup

This project uses a Conda environment defined in `environment.yml`.

### Create & activate environment

```bash
conda env create -f environment.yml
conda activate tomjerry
```

### Update environment (if needed)
```bash
conda env update -f environment.yml --prune
```

## Train the toy model

To start training, simply execute the training script and wait.
```bash
cd src
python train.py
```

After the model is trained, you can try making inferences using `test.py`, this file will print the predicted label of the test data
as well as the probability of the output.
```bash
python test.py
```



