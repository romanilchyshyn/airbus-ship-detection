# Task: Airbus Ship Detection

### Prerequisites

- Working environment expeccts that dataset is downloaded and unarchived in the same directory as the notebooks into `data` directory.
- Run `pip install -r requirements.txt` to install all the necessary packages.

### Notebooks and strcuture

Project contains bunch of notebooks:

[explore.ipynb](explore.ipynb) - contains exploratory data analysis of the dataset, sanity checks. 
Also, it samples the dataset to create a smaller datasets of sizes `s` (20), `m` (200), `l` (1000). Preprocessed csv files are saved in `data_p` directory. All the next notebooks use these preprocessed files.

[preprocess.ipynb](preprocess.ipynb) - contains csv preprocessing. It maps `EncodedPixels` to corresponding mask images. Images put into `data_p` subdirectories directory.

[dataset.ipynb](dataset.ipynb) - contains dataloader for the dataset. It also contains some visualization of the dataset.

[train.ipynb](train.ipynb) - contains training of the model. It uses `data_p` directory to load the dataset. It saves model in with snapshot mechanism. Tensorboard logs are saved in `logs` directory.

[inference.ipynb](inference.ipynb) - use trained model to make predictions on the test dataset and visualize the results.

[submit.ipynb](submit.ipynb) - creates submission file for the competition.

### Model

`segmentation_models` library is used to train the model. It uses `efficientnet-b2` as a backbone. The model is trained with `bce_dice_loss` loss function and `adam` optimizer. `models` directory contains trained models for some combination of datasets.
