# Week 1: Introduction & Fundamentals

## 1. What is Data Science?
Data Science is an interdisciplinary field that uses statistics, machine learning, and domain knowledge to extract insights from structured and unstructured data. It involves data collection, cleaning, analysis, visualization, and predictive modeling.

### **Key Areas of Data Science:**
- **Data Analysis**: Understanding and summarizing data.
- **Machine Learning**: Creating models to make predictions.
- **Big Data**: Handling large datasets efficiently.
- **Data Visualization**: Representing data insights visually.
- **AI & Deep Learning**: Advanced analytics using neural networks.

---

## 2. Data Science Lifecycle & Roles

### **Lifecycle Stages:**
1. **Problem Definition** – Understanding the business need.
2. **Data Collection** – Gathering relevant data.
3. **Data Cleaning** – Handling missing values and outliers.
4. **Exploratory Data Analysis (EDA)** – Summarizing data patterns.
5. **Feature Engineering** – Creating useful features for modeling.
6. **Model Building** – Training machine learning models.
7. **Model Evaluation** – Checking model performance.
8. **Deployment** – Integrating models into applications.
9. **Monitoring & Maintenance** – Ensuring ongoing accuracy.

### **Roles in Data Science:**
- **Data Scientist**: Builds models and extracts insights.
- **Data Engineer**: Develops data pipelines.
- **Data Analyst**: Analyzes and visualizes data for business decisions.
- **ML Engineer**: Deploys and maintains ML models.

---

## 3. Essential Math

### **Linear Algebra:**
- **Matrix Operations**: Addition, subtraction, multiplication.
- **Eigenvalues & Eigenvectors**: Used in dimensionality reduction (PCA).
- **Dot Product & Cross Product**: Essential in ML algorithms.

### **Statistics:**
- **Mean (Average)**: \( \mu = \frac{\sum X}{n} \)
- **Median**: Middle value of sorted data.
- **Mode**: Most frequently occurring value.
- **Standard Deviation**: Measures data spread.
- **Correlation**: Relationship between variables.

### **Probability:**
- **Basic Probability Formula**: \( P(A) = \frac{\text{Favorable Outcomes}}{\text{Total Outcomes}} \)
- **Bayes’ Theorem**: Used for conditional probability.
- **Random Variables & Distributions**: Normal, Binomial, Poisson distributions.

---

## 4. Environment Setup

### **Tools Needed:**
1. **Python**: Install from [Python.org](https://www.python.org/)
2. **Jupyter Notebook**: Install via pip:
   ```bash
   pip install notebook
   ```
3. **Essential Libraries**:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```
4. **Verifying Installation**:
   ```python
   import numpy, pandas, matplotlib, seaborn
   print("Setup Successful!")
   ```

---

## 5. Mini Project: Math & Setup

### **Perform Basic Matrix Operations Using NumPy**
```python
import numpy as np

# Define Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Addition
add_result = A + B

# Matrix Multiplication
mult_result = np.dot(A, B)

# Determinant of A
det_A = np.linalg.det(A)

print("Matrix Addition:\n", add_result)
print("Matrix Multiplication:\n", mult_result)
print("Determinant of A:", det_A)
```

### **Calculate Simple Probabilities**
```python
# Probability of rolling a 6 on a fair die
prob_six = 1 / 6

# Probability of getting heads in a fair coin toss
prob_heads = 1 / 2

print("Probability of rolling a 6:", prob_six)
print("Probability of getting heads:", prob_heads)
```

### **Visualize Results in Jupyter Notebook**
```python
import matplotlib.pyplot as plt

# Sample Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot
plt.plot(x, y, label='Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Visualization')
plt.legend()
plt.show()
```

---

## **Conclusion**
In **Week 1**, you learned:
✅ What Data Science is  
✅ The Data Science lifecycle & roles  
✅ Basic math concepts in Data Science  
✅ Setting up your Python environment  
✅ Hands-on practice with NumPy, probability, and visualization  

# **Week 2: Python Essentials for Data Science**

## **Topics Covered**
1. **Python Basics** – Core programming concepts in Python.
2. **Python Data Libraries** – Introduction to NumPy, Pandas, and Matplotlib.
3. **Data Structures** – Understanding lists, dictionaries, and sets.
4. **Sample Project** – Reading and cleaning a dataset using Pandas.

---

## **1. Python Basics**

### **Data Types**
Python has several built-in data types:

- **Integers (`int`)**: Whole numbers (e.g., `10`, `-5`, `1000`).
- **Floats (`float`)**: Numbers with decimals (e.g., `3.14`, `-0.99`, `2.0`).
- **Strings (`str`)**: Text data enclosed in quotes (`'Hello'`, `"Python"`).
- **Booleans (`bool`)**: Represents `True` or `False`, used in conditions.

**Example:**
```python
x = 10       # Integer
y = 3.14     # Float
name = "John"  # String
is_valid = True  # Boolean
```

### **Loops**
Loops help in automating repetitive tasks.

- **For loop**: Iterates over a sequence (list, tuple, dictionary, string, etc.).
- **While loop**: Repeats a block of code while a condition is true.

**For Loop Example:**
```python
for i in range(5):  # Loops from 0 to 4
    print("Iteration:", i)
```

**While Loop Example:**
```python
count = 0
while count < 3:
    print("Count is:", count)
    count += 1
```

### **Functions**
Functions help in organizing code into reusable blocks.

- **Defining a function**
- **Calling a function**
- **Lambda (anonymous) functions**

**Example:**
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Output: Hello, Alice!
```

**Lambda Example:**
```python
square = lambda x: x ** 2
print(square(4))  # Output: 16
```

### **Modules**
Modules are used to organize reusable code.

- **Built-in modules** (e.g., `math`, `random`).
- **Third-party modules** (e.g., `pandas`, `numpy`).

**Importing a module:**
```python
import math
print(math.sqrt(16))  # Output: 4.0
```

---

## **2. Python Data Libraries**

### **NumPy (Numerical Computing)**
NumPy is used for fast numerical operations.

- Efficient array operations.
- Matrix manipulations.
- Math functions like `sin()`, `cos()`, `log()`, etc.

**Example:**
```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr * 2)  # Output: [2 4 6 8]
```

### **Pandas (Data Manipulation)**
Pandas is used to manipulate structured data using DataFrames.

- **DataFrames**: 2D table-like data structure.
- **Series**: 1D array with labeled index.

**Example:**
```python
import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)
```

### **Matplotlib (Data Visualization)**
Matplotlib is used for plotting graphs and charts.

**Example:**
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.show()
```

---

## **3. Data Structures**

### **Lists (Ordered, Mutable)**
A list is an ordered collection that can hold different data types.

**Example:**
```python
fruits = ["Apple", "Banana", "Cherry"]
fruits.append("Mango")
print(fruits)  # Output: ['Apple', 'Banana', 'Cherry', 'Mango']
```

### **Dictionaries (Key-Value Pairs)**
Dictionaries store data in key-value format.

**Example:**
```python
person = {"name": "Alice", "age": 25}
print(person["name"])  # Output: Alice
```

### **Sets (Unordered, Unique Elements)**
A set stores unique values without any order.

**Example:**
```python
unique_numbers = {1, 2, 3, 3, 4}
print(unique_numbers)  # Output: {1, 2, 3, 4}
```

---

## **4. Sample Project: Data Manipulation Using Pandas**

### **Read Data from a CSV File & Clean it**
```python
import pandas as pd

# Load CSV File
df = pd.read_csv("data.csv")

# Display first 5 rows
print(df.head())

# Drop missing values
df_cleaned = df.dropna()

# Calculate key statistics
print("Summary Statistics:\n", df_cleaned.describe())
```

---

## **Conclusion**
- Learned Python basics: data types, loops, functions, and modules.
- Explored data libraries: NumPy, Pandas, Matplotlib.
- Understood data structures: lists, dictionaries, sets.
- Implemented a simple project using Pandas.
