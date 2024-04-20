# MNIST Digit Classifier

## Overview
This project includes a convolutional neural network (CNN) designed to classify handwritten digits from the MNIST dataset. Additionally, it features a graphical user interface (GUI) that allows users to draw a digit on a canvas and use the trained model to predict the digit.

## Features
- **Data Loading:** Load and preprocess the MNIST dataset.
- **Model Training:** Train a CNN model to classify handwritten digits.
- **Model Evaluation:** Evaluate and save the trained model.
- **GUI Application:** Digit prediction using the trained model through a GUI.

## Prerequisites
Ensure you have the following software installed:
- Python 3.8 or higher
- NumPy
- Pandas
- TensorFlow
- Scikit-Learn
- PIL (Pillow)
- tkinter

Install the required packages using pip:
```bash
pip install numpy pandas tensorflow scikit-learn pillow
```
## Dataset
Place the MNIST dataset files in the following directory structure:
- Ensure `mnist_train.csv` and `mnist_test.csv` are located under `C:\Users\YourUsername\Downloads\mnist\`.
- Substitute `YourUsername` with your actual username on your computer.

## Usage

### Training the Model
Execute the script parts that involve data loading, preprocessing, model training, and evaluation. Adjust the file paths to match the location where you've stored the MNIST dataset.

### Running the GUI Application
To launch the GUI application for digit prediction, run the tkinter script portion. This will allow you to draw a digit on a canvas and use the "Predict" button to see the model's prediction.

## Project Structure
- `mnist_train.csv`: The training data file.
- `mnist_test.csv`: The test data file.
- `mnist_cnn_model.h5`: The file where the trained model is saved.

## Model Architecture
- **Convolutional Layers:** Three layers with ReLU activations (first layer with 32 filters of size 3x3, followed by two layers with 64 filters of the same size).
- **Max-Pooling Layers:** Two layers following the first and second convolutional layers, both with a pool size of 2x2.
- **Flattening Layer:** Converts the convolutional layer outputs to a flat array for the dense layer.
- **Dense Layers:** Two layers (one with 64 units and ReLU activation, another with 10 units and softmax activation for digit classification).

## License
Please specify the license under which the project is released.
