# MachineLearning101

A simple machine learning project demonstrating linear regression to predict unit sales based on marketing spend.

## Project Overview

This project implements a basic linear regression model to analyze the relationship between marketing spend and units sold. It includes:

- Training a linear regression model
- Saving and loading the trained model
- Visualizing the results with a plot

## Files

- `TrainModel.py` - Trains the linear regression model on the retail sales data
- `PlotModel.py` - Creates a visualization of the model's predictions
- `Retail_sales.csv` - Input data containing marketing spend and units sold
- `trained_model.pkl` - Saved trained model (tracked with DVC)
- `eda_outputs/` - Contains exploratory data analysis results

## Requirements

- Python 3.x
- Required packages:
  ```
  scikit-learn
  pandas
  numpy
  matplotlib
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Haripdurai/machineLearning101.git
   cd machineLearning101
   ```

2. Install required packages:
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```

## Usage

1. Train the model:
   ```bash
   python TrainModel.py
   ```
   This will create `trained_model.pkl` with the trained model and training data.

2. Generate the prediction plot:
   ```bash
   python PlotModel.py
   ```
   This will create `plot.png` showing the actual data points and regression line.

## Data

The project uses retail sales data with two main columns:
- Marketing Spend (USD)
- Units Sold

The data is tracked using DVC (Data Version Control) for better version management of large files.

## Model Details

- Type: Linear Regression
- Input: Marketing Spend (USD)
- Output: Predicted Units Sold
- Files: Model is saved in `trained_model.pkl` using Python's pickle module

## Notes

- The trained model is saved with both the model object and the training data for reproducibility
- Data files are version controlled using DVC
- Exploratory data analysis results are stored in the `eda_outputs` directory