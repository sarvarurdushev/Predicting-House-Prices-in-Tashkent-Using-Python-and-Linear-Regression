# Predicting-House-Prices-in-Tashkent-Using-Python-and-Linear-Regression
Platform: Anaconda Online
# 📘 Introduction
Ever wondered how much a home in your city might cost based on its size? Or how machine learning lets us make such predictions?

In this project, we’ll use Python to explore linear regression—a key method in predictive modeling. With just a few essential libraries, you’ll learn how to predict housing prices in Tashkent based on size using real data.

Libraries we’ll use:

Scikit-learn for machine learning

Pandas for data handling

NumPy for numerical operations

Matplotlib for visualization

# 📈 What is Linear Regression?
Linear regression helps us understand the relationship between one or more input features (like house size) and an output (like house price). Think of it as drawing the best-fit line through data to spot trends and make predictions. It’s a popular method for estimating things like pricing, trends, and value shifts.

In our case, we’ll use it to estimate house prices in Tashkent based on the square footage of each home.

# ⚙️ Getting Started with Anaconda
We’ll run everything on Anaconda Online—no local installs needed. Just sign up for a free account, head to the Anaconda homepage, and click “Launch Notebook” under Code Online.

Create a new Jupyter Notebook project to begin coding.

# 🧰 Tools You’ll Use
We’ll use these core libraries:

NumPy: for working with arrays and numeric data

Pandas: for reading and cleaning the dataset

Matplotlib: to visualize data

Scikit-learn: to apply the linear regression model

# 🏡 Dataset: Tashkent House Sizes and Prices
We’re using a dataset containing property sizes and prices from Tashkent. After uploading your Excel file (uybor.xlsx) to your notebook, let’s load and prepare the data:

<pre> 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
</pre>

# Load data from XLSX file
<pre>data = pd.read_excel('uybor.xlsx')</pre>

# Extract features and target variable
<pre>datahouse_sizes = data['size'].values
house_prices = data['price'].values</pre>

# 🖼️ Visualizing the Data
Let’s plot a scatter graph to see how house size correlates with price:

<pre>
plt.scatter(house_sizes, house_prices, marker='o', color='blue')
plt.title('House Prices vs. Size in Tashkent')
plt.xlabel('Size (sq.m)')
plt.ylabel('Price ($)')
plt.show()</pre>

# 🧪 Train-Test Split
We’ll divide the dataset so 80% is used for training the model and 20% for testing how well it performs:

<pre>x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)</pre>

Before feeding the data into our model, let’s reshape it for NumPy:

<pre>
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)</pre>
🤖 Building and Training the Model
Now let’s create our model and train it:

<pre>
model = LinearRegression()
model.fit(x_train, y_train)</pre>
# 📊 Making Predictions
We’ll now use the model to predict prices from the test set and visualize the results:

<pre> predictions = model.predict(x_test)</pre>

<pre>
plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual Prices')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Prices')
plt.title('Tashkent Housing Price Prediction')
plt.xlabel('Size (sq.m)')
plt.ylabel('Price ($)')
plt.legend()
plt.show()</pre>
# ✅ Conclusion
You’ve successfully built and visualized a simple predictive model using linear regression on real housing data from Tashkent!

This is just the beginning. Try experimenting with more variables or exploring other datasets. Whether it’s predicting sales, performance, or prices, linear regression is a great tool to have in your data science toolkit.

# 📚 Resources
Kaggle Datasets: Browse more datasets

Scikit-learn Docs: Explore ML features

Pandas Docs: Learn data handling

Matplotlib Docs: Master data visualization
