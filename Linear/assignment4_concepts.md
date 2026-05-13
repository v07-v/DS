# Assignment 4 Notebook Analysis (assign4.ipynb)

## Overview
This notebook demonstrates the idea of **linear regression** using two parts:
1. A small manual example with `numpy.polyfit()` and `numpy.poly1d()`
2. A full regression model using the California Housing dataset with **train-test split**, **LinearRegression**, **prediction**, **MSE**, and **R² score**

The main goal is to predict a numeric value from input features.

---

## Part 1: Simple Linear Regression Example

### 1) Data creation
```python
x = np.array([95,85,80,70,60])
y = np.array([85,95,70,65,70])
```

Meaning:
- `x` is the input feature
- `y` is the target/output

This is a small demo dataset used to show how linear regression works.

### 2) Fitting a line with `np.polyfit()`
```python
model = np.polyfit(x, y, 1)
```

What it does:
- Fits a straight line to the data.
- The `1` means degree 1, so the model is a line.

Line equation:
$$
y = mx + c
$$
Where:
- $m$ = slope
- $c$ = intercept

`np.polyfit(x, y, 1)` returns the line coefficients `[m, c]`.

Why it matters:
- It gives the best straight line that fits the data.

### 3) Creating a prediction function with `np.poly1d()`
```python
predict = np.poly1d(model)
```

What it does:
- Converts the coefficients into a usable polynomial function.
- After this, you can directly call `predict(65)`.

Example:
- If you do `predict(65)`, it estimates the `y` value when `x = 65`.

### 4) Predicting values for the same x data
```python
y_pred = predict(x)
```

What it does:
- Calculates predicted `y` values for each `x`.

Why it matters:
- Lets you compare actual values and predicted values.

### 5) R² score
```python
r2_score(y, y_pred)
```

Meaning:
- Measures how well the line fits the data.
- R² close to 1 means a good fit.
- R² close to 0 means weak fit.

Formula:
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $\bar{y}$ = mean of actual values

### 6) Plotting the line and points
```python
y_line = model[1] + model[0] * x
plt.plot(x, y_line, c='r')
plt.scatter(x, y_pred)
plt.scatter(x, y, c='r')
```

What it does:
- Draws the regression line.
- Shows actual points and predicted points.

Why it matters:
- Helps visually check whether the line matches the data.

---

## Part 2: Real Regression Model on Housing Data

### 1) Load dataset
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data = pd.DataFrame(housing.data)
```

Meaning:
- Loads a built-in housing dataset.
- The notebook comment mentions Boston Housing, but the code actually uses **California Housing**.

### 2) Add column names
```python
data.columns = housing.feature_names
```

What it does:
- Assigns meaningful names to features like income, rooms, population, etc.

### 3) Add target column
```python
data['PRICE'] = housing.target
```

Meaning:
- Adds the value to be predicted.
- Here, `PRICE` is the target variable.

### 4) Check missing values
```python
data.isnull().sum()
```

What it does:
- Counts missing values in each column.

Why it matters:
- A regression model usually needs clean data.

### 5) Separate features and target
```python
x = data.drop(['PRICE'], axis=1)
y = data['PRICE']
```

Meaning:
- `x` = input features
- `y` = output/target

This is a standard machine learning setup.

---

## Train-Test Split Theory

### 6) Split data into training and testing sets
```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
```

What it does:
- Splits the dataset into two parts:
  - training set
  - testing set

Why this is important:
- The model should learn from one part of the data and be tested on unseen data.
- This checks whether the model generalizes well.

Train-test split idea:
- Training data: used to fit the model
- Testing data: used only to evaluate the model

Why not use all data for training?
- If you test on the same data you trained on, the result may look too good.
- That gives an overfitted and unrealistic performance estimate.

`test_size=0.2` means:
- 80% training data
- 20% testing data

`random_state=0` means:
- The split is reproducible, so you get the same split every time.

---

## Linear Regression Theory

### 7) Create linear regression model
```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
```

What it does:
- Creates a linear regression model object.

Regression equation:
$$
\hat{y} = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
$$

Where:
- $\hat{y}$ = predicted output
- $b_0$ = intercept
- $b_1, b_2, ..., b_n$ = coefficients
- $x_1, x_2, ..., x_n$ = input features

If there is only one feature, it becomes:
$$
\hat{y} = mx + c
$$

### 8) Fit the model
```python
model = lm.fit(xtrain, ytrain)
```

What it does:
- Learns the relationship between features and target using training data.

In simple words:
- The model tries to find the best line/plane that predicts house price.

---

## Prediction and Evaluation

### 9) Predict on train and test data
```python
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)
```

What it does:
- Produces predicted values for both training and test inputs.

Why it matters:
- Training predictions show how well the model learned the training data.
- Test predictions show how well it performs on unseen data.

### 10) Mean Squared Error (MSE)
```python
mse = mean_squared_error(ytest, ytest_pred)
```

Formula:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Meaning:
- Measures average squared difference between actual and predicted values.
- Lower MSE means better predictions.

Why square the error?
- Large mistakes are penalized more.

The notebook also computes MSE for training predictions:
```python
mse = mean_squared_error(ytrain_pred, ytrain)
```

### 11) Compare true vs predicted values
```python
plt.scatter(ytrain, ytrain_pred, ...)
plt.scatter(ytest, ytest_pred, ...)
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title('True value vs Predicted value')
plt.legend(loc='upper left')
plt.show()
```

What it does:
- Plots actual values against predicted values.
- Blue points = training data
- Green points = test data

How to read it:
- If points are close to a diagonal line, predictions are good.
- More spread means more prediction error.

---

## Important Concepts in Simple Words

### Train vs Test
- Training data = data the model studies
- Testing data = data used to check if the model really learned

### Overfitting
- When the model memorizes training data too well but performs badly on new data

### Underfitting
- When the model is too simple and does not learn the real pattern

### Prediction
- Using the trained model to estimate a target value

### Error
- Difference between actual and predicted values

---

## Overall Significance of the Notebook
This notebook teaches the basic workflow of regression in machine learning:
1. Prepare data
2. Split data into train and test
3. Train a linear regression model
4. Predict values
5. Evaluate with MSE and R²
6. Visualize results

This is the standard process used in many supervised learning problems.

---

## Notes
- The first part is a small manual example to explain regression conceptually.
- The second part is the real machine learning workflow.
- The notebook comment mentions Boston Housing, but the code uses California Housing data.
