# Stock-Price-Prediction-using-Multiple-Machine-Learning-Models


---

# **Stock Price Prediction using Multiple Machine Learning Models**

## **Overview**

This project aims to predict stock prices using various machine learning models, including **Support Vector Regression (SVR)**, **Random Forest Regressor**, **XGBoost**, and **LSTM**. The models are trained using historical stock data along with **technical indicators** and **sentiment analysis** as features. The goal is to evaluate and compare the performance of different models in predicting the next day's stock price.

---

## **Key Features**

- **Stock Price Prediction** using multiple regression models (SVR, Random Forest, XGBoost, LSTM).
- **Data Preprocessing** including the calculation of technical indicators like RSI, MACD, and SMA.
- **Sentiment Analysis** based on stock-related news headlines (using a dummy sentiment score for this project).
- **Model Evaluation** using Mean Squared Error (MSE), Residual Plots, and Heatmaps to compare model performances.
- **Visualization** of model performance and stock price prediction results.

---

## **Getting Started**

### **Prerequisites**

To run this project, you need Python 3.6+ and the following libraries:

- **yfinance**: To download stock data.
- **ta**: For calculating technical indicators like RSI, MACD, and SMA.
- **nltk, vaderSentiment**: For sentiment analysis.
- **scikit-learn**: For machine learning models (SVR, Random Forest).
- **XGBoost**: For the XGBoost model.
- **TensorFlow/Keras**: For LSTM model.
- **matplotlib**: For plotting visualizations.

### **Installation**

You can install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following:

```txt
yfinance
ta
nltk
vaderSentiment
scikit-learn
xgboost
tensorflow
matplotlib
```

---

## **How to Run the Project**

1. Clone the repository:

```bash
git clone https://github.com/Geek-MJ/stock-price-prediction.git
cd stock-price-prediction
```

2. Download the stock data (the code automatically fetches it for you):

```python
python download_data.py
```

3. Run the model training and evaluation:

```python
python train_and_evaluate.py
```

This will train the models on the stock data and display performance metrics and comparison.

4. Visualize the predictions:

```python
python plot_results.py
```

This will plot the actual vs predicted stock prices for each model.

---

## **Project Structure**

- `download_data.py`: Downloads historical stock data using the **yfinance** library.
- `data_preprocessing.py`: Handles the preprocessing steps such as adding technical indicators and performing sentiment analysis.
- `train_and_evaluate.py`: Trains the models (SVR, Random Forest, XGBoost, LSTM) and evaluates them using MSE and residuals.
- `plot_results.py`: Plots model performance, actual vs predicted stock prices, and residuals.
- `requirements.txt`: Lists all dependencies for the project.
- `README.md`: This file.

---

## **Model Evaluation**

- **Mean Squared Error (MSE)**: We use MSE to evaluate the performance of the models. A lower MSE value indicates a better fit.
- **Residuals Plot**: A plot showing the difference between predicted and actual values for each model.
- **Correlation Heatmap**: A heatmap comparing the predictions of each model against the actual values.

---


## **Conclusion**

This project provides a baseline for stock price prediction using machine learning models. The comparison of multiple models gives valuable insights into which approaches work best for predicting stock prices based on historical data, technical indicators, and sentiment analysis. 

---
