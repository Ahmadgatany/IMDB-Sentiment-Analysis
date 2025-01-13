# IMDB Sentiment Analysis with RNN and MLflow

## Project Description
This project involves building a sentiment analysis model using the IMDB movie reviews dataset. The model is built using a SimpleRNN layer and tracks experiments using MLflow.

## Dataset
The dataset used is the IMDB Dataset of 50K Movie Reviews, which is available on Kaggle. The dataset contains 50,000 movie reviews labeled as positive or negative.

## Requirements
- Python 3.x
- pandas
- numpy
- tensorflow
- sklearn
- mlflow

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Running the Project
1. Clone the repository.
2. Navigate to the project directory.
3. Run the script using:
   ```bash
   python script_name.py
   ```

## Experiment Tracking with MLflow
MLflow is used to track experiments, including parameters, metrics, and model artifacts. The experiments are logged in an MLflow tracking server.

### Viewing the MLflow UI
To explore the tracked experiments, you can start the MLflow UI by running the following command:

```bash
mlflow ui
```

Then, open your browser and navigate to [http://localhost:5000](http://localhost:5000) to view the experiments.

## Model Results
The model achieved an accuracy of approximately **X.XXX** on the test set.

## Project Structure
```
project/
|
├── script_name.py       # The main script for training the model
├── requirements.txt     # File listing the dependencies
├── README.md            # This README file
└── ...
```

## Acknowledgements
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) provided by Kaggle.

