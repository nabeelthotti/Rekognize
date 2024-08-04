# app/model/load_emnist.py

import pandas as pd
import numpy as np

def load_emnist(csv_path):
    # Read the CSV file
    data = pd.read_csv(csv_path, header=None)
    
    # The first column is the label, adjust labels to be zero-indexed
    y = data.iloc[:, 0].values - 1  # Ensure labels start from 0
    # The rest of the columns are the pixels
    x = data.iloc[:, 1:].values
    
    # Reshape the data to the correct dimensions for an image
    x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return x, y
