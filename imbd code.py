import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras

# Load dataset
path = 'IMDB Dataset.csv'
df = pd.read_csv(path)

# Extract sentences and labels
sentences = df['review'].values
labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

maxlen = 200
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Start MLflow experiment tracking
with mlflow.start_run():
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128))
    model.add(SimpleRNN(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.7))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

    # Log parameters
    mlflow.log_param("num_words", 5000)
    mlflow.log_param("maxlen", maxlen)
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("rnn_units", 64)
    mlflow.log_param("dropout_rate", 0.7)
    mlflow.log_param("dense_units", 32)
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("batch_size", 128)
    mlflow.log_param("epochs", 6)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=6, batch_size=128,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Log metrics
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)

    # Log the model
    mlflow.keras.log_model(model, "model")

    print(f'Test Accuracy: {accuracy:.4f}')
    
    
    
