# __init__.py in the model directory

# Import key functions or classes from the module to make them accessible at the package level
# app/model/__init__.py
from .train_alphabet_model import train_and_save_alphabet_model

from .load_emnist import load_emnist

# You can also include any initial setup code here
print("Model package initialized.")


