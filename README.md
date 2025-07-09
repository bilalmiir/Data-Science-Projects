# Data-Science-Projects

# Project 1: # 🩺 Common Disease Diagnosis

A machine learning project designed to predict **common diseases** based on symptoms using supervised learning models. The goal is to assist in early diagnosis by leveraging ML algorithms on real-world health data.

---

## 📌 Project Overview

This project uses machine learning techniques to diagnose common diseases by analyzing symptoms data.  
It involves:

- Cleaning and preprocessing a medical dataset
- Training multiple classification models
- Comparing model performance
- Deploying the best-performing model for predictions

---

## 📊 Dataset Details

- **Source**: [Kaggle - Disease Prediction Dataset](https://www.kaggle.com/)
- **Content**: The dataset contains records of symptoms and corresponding diagnosed diseases.
- **Features**:
  - Multiple binary symptom indicators (e.g., `fever`, `headache`, `nausea`, etc.)
  - Target label indicating the disease

---

## 🧠 Models Used

Two classification models were trained and evaluated:

- ✅ **Logistic Regression**
- ✅ **Decision Tree Classifier**

After evaluation, the model with the **best accuracy** was selected for the final prediction pipeline.

---

# Project 2: # 📉 Customer Churn Prediction

Customer churn is a critical issue for businesses, especially in subscription-based industries. This project aims to predict whether a customer is likely to churn (leave) using an **Artificial Neural Network (ANN)**. By identifying customers at risk, businesses can take proactive steps to retain them and improve customer satisfaction.

---

## 📌 Project Overview

Customer churn refers to when a customer stops doing business or ends their relationship with a company. Predicting churn helps companies retain valuable customers and reduce revenue loss.

This machine learning project uses an **Artificial Neural Network (ANN)** to analyze various customer-related features and predict the likelihood of churn. The model learns patterns from historical data and helps in making data-driven business decisions.

---

## 📊 Dataset Description

The dataset used in this project was obtained from **[Kaggle](https://www.kaggle.com/)** and includes information about customers of a telecom company. It contains the following types of features:

- **Demographic Information**: Gender, Age, Geography
- **Account Information**: Tenure, Balance, Number of Products
- **Service Usage**: Has Credit Card, Is Active Member, Estimated Salary
- **Target Variable**: `Exited` (1 = churned, 0 = retained)

---

## 🧠 Model & Training

The core of this project is an **Artificial Neural Network (ANN)** built using **TensorFlow/Keras**.

### 🔧 Preprocessing Steps:
- Handled missing values
- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Feature scaling using StandardScaler

### 🏗️ ANN Architecture:
- **Input Layer**: 11 input features
- **Hidden Layer 1**: 11 neurons, ReLU activation
- **Hidden Layer 2**: 11 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

### 🛠️ Training:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 100
- Batch Size: 32

---

## 📈 Evaluation Metrics

The model was evaluated using the following metrics:

- **Accuracy**

These metrics help assess the model’s ability to correctly identify churned vs. retained customers.

---

## 🚀 How to Run

Follow the steps below to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

```

# Project 3: 📈 Stock Price Prediction using LSTM

## 📝 Overview

This project aims to predict the future stock prices of **Nestlé Pakistan (NESTLE.PK)** using **Long Short-Term Memory (LSTM)** neural networks. By analyzing historical stock data obtained from the **Pakistan Stock Exchange (PSX)**, the model attempts to capture temporal patterns in the data to forecast future trends with improved accuracy.

---

## 📊 Dataset

- **Source:** [Pakistan Stock Exchange (PSX)](https://www.psx.com.pk/)
- **Stock Selected:** Nestlé Pakistan Limited (NESTLE.PK)
- **Data Collected:** Historical daily stock prices (Open, High, Low, Close, Volume)
- **Time Period:** Based on data available at the time of scraping

---

## 🧰 Technology Stack

The following tools and libraries were used to build the project:

- **Python 3.x** – Programming language
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and preprocessing
- **Matplotlib / Seaborn** – Data visualization
- **Scikit-learn** – Scaling and evaluation utilities
- **TensorFlow / Keras** – Building and training the LSTM neural network

---

## 🧠 Model

### Why LSTM?

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) that is effective for sequence prediction problems like time-series forecasting. It is especially suitable for capturing long-term dependencies and patterns in sequential data, such as stock prices.

### Architecture Summary

- **Input:** Sequences of normalized stock prices
- **Layers:**
  - LSTM layer with 50 units
  - Dropout for regularization
  - Dense output layer for regression
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predicted stock price
