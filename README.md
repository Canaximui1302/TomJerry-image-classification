# TomJerry_classification

## Files

This project implements a toy model using AlexNet architecture for image classification. The model is defined and trained from scratch in `src/model.py`.
The file for testing model's prediction is `src/test.py` and the training loop is defined in `src/train.py`.

## Data

The dataset is a folder containing frames cut from the animated series Tom and Jerry. Training dataset can be retrieved from Kaggle:

https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

In order to train the model, you need to download the dataset and organize the folders as below:
```
project
│   README.md
│   file001.txt    
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
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




