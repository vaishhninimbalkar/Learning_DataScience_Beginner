
# 6-Month Data Science Learning Course (Extended with Python Code Samples)

Below is a **24-week** schedule (6 months) with weekly topics, brief descriptions, sample projects, **Python code examples**, and **useful documentation links**. Each week includes **four** hands-on mini-projects or program snippets in Python to reinforce learning. Where relevant, links to documentation and popular tools/frameworks are provided.

---

## Month 1: Foundations of Data Science

---

### Week 1: Introduction & Fundamentals

**Topics:**
- What is Data Science?
- Data Science lifecycle and roles
- Essential math (basic linear algebra, statistics, probability)
- Environment setup (Python, Jupyter, libraries)

**Description:**
- Introduction to the data science field, its roles (Data Scientist, Data Engineer, Analyst), and fundamental math concepts.
- Setting up your Python environment, including Jupyter notebooks and key libraries (NumPy, Pandas).

**Sample Projects/Programs (Code Examples)**

1. **Check Environment & Print Versions**  
   
A simple script to check your Python environment and print library versions.
```python
   import sys
   import numpy as np
   import pandas as pd

   print("Python version:", sys.version)
   print("NumPy version:", np.__version__)
   print("Pandas version:", pd.__version__)
```

Matrix Multiplication with NumPy
Demonstrates basic linear algebra operations.

import numpy as np

# Create two matrices

```
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
```

# Matrix multiplication

```
C = np.dot(A, B)
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C = A x B:\n", C)

```

Basic Probability Calculation
Calculates simple probabilities for coin toss outcomes.

```
import random

def coin_toss(num_tosses=1000):
    heads_count = 0
    for _ in range(num_tosses):
        # 0 = tails, 1 = heads
        outcome = random.randint(0, 1)
        heads_count += outcome
    return heads_count / num_tosses

probability_heads = coin_toss()
print(f"Estimated probability of heads: {probability_heads}")

```

Data Distribution Plot
Show how to visualize a random distribution (e.g., normal distribution) using Matplotlib.

import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data, bins=30, alpha=0.7, color='blue')
plt.title("Normal Distribution (Mean=0, Std=1)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
References & Documentation

Python Official Documentation
NumPy Documentation
Pandas Documentation
Jupyter Documentation
Week 2: Python Essentials for Data Science
Topics:

Python basics (data types, loops, functions, modules)
Python data libraries (NumPy, Pandas, Matplotlib)
Data structures (lists, dictionaries, sets)
Description:

Build a strong foundation in Python programming.
Learn to use libraries for data manipulation (NumPy, Pandas) and visualization (Matplotlib).
Sample Projects/Programs (Code Examples)

Python Basics: Data Types & Loops

# Simple list and loop example
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(f"{i} -> {fruit}")
Defining and Importing a Custom Module
Create a simple module, my_math.py, and import it.

# my_math.py
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

# main.py
import my_math

print(my_math.add_numbers(3, 5))
print(my_math.multiply_numbers(4, 7))
Pandas DataFrame Creation

import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)
print(df)
Matplotlib Bar Chart

import matplotlib.pyplot as plt

cities = ["New York", "Los Angeles", "Chicago"]
population = [8.4, 4.0, 2.7]  # in millions

plt.bar(cities, population, color='green')
plt.title("Population by City")
plt.xlabel("City")
plt.ylabel("Population (millions)")
plt.show()
References & Documentation

Python Official Documentation
NumPy Documentation
Pandas Documentation
Matplotlib Documentation
Week 3: Statistics & Probability
Topics:

Descriptive statistics (mean, median, mode, variance, standard deviation)
Probability distributions (Normal, Binomial, Poisson)
Inferential statistics (confidence intervals, hypothesis testing)
Description:

Delve into basic and advanced statistical techniques.
Learn to compute descriptive statistics and understand distributions.
Sample Projects/Programs (Code Examples)

Descriptive Statistics with Python

import numpy as np

data = np.random.randint(1, 100, 50)
print("Data:", data)
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Std Dev:", np.std(data))
Binomial Distribution Simulation

import numpy as np
import matplotlib.pyplot as plt

# n = number of trials, p = probability of success
n, p = 10, 0.5
binom_data = np.random.binomial(n=n, p=p, size=1000)

plt.hist(binom_data, bins=range(n+2), alpha=0.7, color='purple', align='left')
plt.title("Binomial Distribution")
plt.xlabel("Number of successes")
plt.ylabel("Frequency")
plt.show()
Confidence Interval Calculation

import numpy as np
from scipy import stats

sample_data = np.random.normal(loc=10, scale=2, size=50)
mean_est = np.mean(sample_data)
sem = stats.sem(sample_data)  # standard error of the mean
confidence = 0.95
interval = stats.t.interval(confidence, len(sample_data)-1, loc=mean_est, scale=sem)

print(f"Mean estimate: {mean_est:.2f}")
print(f"{int(confidence*100)}% Confidence Interval: {interval}")
Hypothesis Testing (t-test)

import numpy as np
from scipy.stats import ttest_ind

group1 = np.random.normal(10, 1.5, 30)
group2 = np.random.normal(11, 1.5, 30)

t_stat, p_value = ttest_ind(group1, group2)
print("T-statistic:", t_stat)
print("P-value:", p_value)
if p_value < 0.05:
    print("Significant difference between the two groups (p < 0.05).")
else:
    print("No significant difference (p >= 0.05).")
References & Documentation

SciPy Stats Documentation
NumPy Documentation
Matplotlib Documentation
Week 4: Exploratory Data Analysis (EDA)
Topics:

Data cleaning (handling missing data, outliers)
Data visualization (histograms, box plots, scatter plots)
Correlation analysis
Description:

Explore and visualize datasets to detect anomalies, relationships, and gain initial insights.
Sample Projects/Programs (Code Examples)

Handling Missing Data

import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 7, 8, 9],
    'C': [10, 11, 12, 13, 14]
})
print("Original DataFrame:\n", df)
df_filled = df.fillna(df.mean())
print("\nAfter Filling NaNs with Mean:\n", df_filled)
Outlier Detection with Box Plot

import matplotlib.pyplot as plt

data = [1,2,2,3,4,5,20,2,3,4,5,2,100,3,4]
plt.boxplot(data)
plt.title("Outlier Detection via Box Plot")
plt.show()
Scatter Plot & Correlation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(50)
y = 2.5 * x + np.random.randn(50) * 0.2
df = pd.DataFrame({'x': x, 'y': y})

plt.scatter(df['x'], df['y'])
plt.title('Scatter Plot of x vs. y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

corr_value = df.corr().loc['x','y']
print("Correlation between x and y:", corr_value)
Pairwise Plot (Using Seaborn)

import seaborn as sns
import pandas as pd

df_iris = sns.load_dataset("iris")
sns.pairplot(df_iris, hue="species")
plt.show()
References & Documentation

Pandas Documentation
Matplotlib Documentation
Seaborn Documentation
Month 2: Data Wrangling & Python for Analytics
Week 5: Advanced Pandas & Data Wrangling
Topics:

Merging, grouping, pivoting data
Advanced filtering and transformations
Optimizing performance for large datasets
Description:

Learn sophisticated methods in Pandas to combine, reshape, and manipulate large datasets efficiently.
Sample Projects/Programs (Code Examples)

Merging DataFrames

import pandas as pd

df_left = pd.DataFrame({
    'key': [1, 2, 3],
    'A': ['A1', 'A2', 'A3']
})
df_right = pd.DataFrame({
    'key': [1, 2, 4],
    'B': ['B1', 'B2', 'B4']
})
merged = pd.merge(df_left, df_right, on='key', how='outer')
print(merged)
Grouping and Aggregation

import pandas as pd

data = {
    'Team': ['A','A','B','B','C','C'],
    'Score': [10, 20, 15, 25, 30, 45]
}
df = pd.DataFrame(data)
group_result = df.groupby('Team').agg({'Score': ['mean','sum']})
print(group_result)
Pivot Tables

import pandas as pd

df = pd.DataFrame({
    'City': ['New York','New York','Chicago','Chicago','LA','LA'],
    'Month': ['Jan','Feb','Jan','Feb','Jan','Feb'],
    'Sales': [100,150,80,120,90,110]
})
pivot_df = df.pivot_table(values='Sales', index='City', columns='Month', aggfunc='sum')
print(pivot_df)
Optimizing Performance (Chunking)

import pandas as pd

# For large CSV, read in chunks to avoid memory issues
chunk_size = 50000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process each chunk
    filtered_chunk = chunk[chunk['value'] > 100]
    # ... further processing ...
    print("Processed a chunk of size:", len(filtered_chunk))
References & Documentation

Pandas User Guide
Python Data Science Handbook
Dask Documentation (for larger-than-memory data)
Week 6: Working with Databases & SQL
Topics:

SQL essentials (SELECT, JOIN, GROUP BY, subqueries)
Integrating Python with SQL databases
Best practices for data retrieval and storage
Description:

Perform queries using SQL and seamlessly integrate results into Python for further analysis.
Sample Projects/Programs (Code Examples)

Basic SELECT Statement (SQLite)

import sqlite3

# Create in-memory database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        salary REAL
    )
''')
# Insert data
cursor.execute("INSERT INTO employees (name, salary) VALUES ('Alice', 70000)")
cursor.execute("INSERT INTO employees (name, salary) VALUES ('Bob', 80000)")
conn.commit()

# SELECT query
cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
SQL JOIN Example

import sqlite3

conn = sqlite3.connect(':memory:')
c = conn.cursor()

c.execute("CREATE TABLE department (dept_id INTEGER, dept_name TEXT)")
c.execute("CREATE TABLE employee (emp_id INTEGER, emp_name TEXT, dept_id INTEGER)")

# Insert into department
c.execute("INSERT INTO department VALUES (1, 'Sales')")
c.execute("INSERT INTO department VALUES (2, 'Engineering')")
# Insert into employee
c.execute("INSERT INTO employee VALUES (101, 'Alice', 1)")
c.execute("INSERT INTO employee VALUES (102, 'Bob', 2)")
conn.commit()

c.execute("""
    SELECT e.emp_name, d.dept_name
    FROM employee e
    JOIN department d ON e.dept_id = d.dept_id
""")
print(c.fetchall())
conn.close()
Using Pandas to Read SQL

import pandas as pd
import sqlite3

conn = sqlite3.connect('example.db')
query = "SELECT * FROM employees;"
df = pd.read_sql_query(query, conn)
print(df.head())
conn.close()
Best Practices: Parameterized Queries

import sqlite3

conn = sqlite3.connect('secure.db')
cursor = conn.cursor()

# Parameterized query to prevent SQL injection
name = "Alice"
salary = 90000
cursor.execute("INSERT INTO employees (name, salary) VALUES (?, ?)", (name, salary))

conn.commit()
conn.close()
References & Documentation

SQLite Documentation
PostgreSQL Documentation
Pandas SQL Interface
Week 7: Data Cleaning & Feature Engineering
Topics:

Handling missing values, categorical encoding, transformations (log, scaling)
Feature selection and feature extraction
Dealing with imbalanced datasets
Description:

Preprocess raw data by fixing anomalies, transforming variables, and engineering new features.
Sample Projects/Programs (Code Examples)

Categorical Encoding

import pandas as pd

data = {
    'color': ['red', 'blue', 'green', 'blue', 'red', 'green']
}
df = pd.DataFrame(data)
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['color'])
print(df_encoded)
Log Transformation & Scaling

import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([100, 200, 1000, 5000, 10000], dtype=float).reshape(-1,1)
log_data = np.log(data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(log_data)

print("Original:", data.ravel())
print("Log Transformed:", log_data.ravel())
print("Scaled:", scaled_data.ravel())
Feature Selection (Using SelectKBest)

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))
Imbalanced Dataset Handling (SMOTE)

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
print("Original class distribution:", pd.Series(y).value_counts())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Resampled class distribution:", pd.Series(y_resampled).value_counts())
References & Documentation

Scikit-learn Feature Selection
Imbalanced-learn Documentation
Scikit-learn Preprocessing
Week 8: Data Visualization & Storytelling
Topics:

Advanced visualization libraries (Seaborn, Plotly)
Best practices for storytelling with data
Dashboard creation (Tableau, Power BI, or Python-based dashboards)
Description:

Master visualization techniques with advanced libraries and create effective dashboards to communicate insights.
Sample Projects/Programs (Code Examples)

Seaborn Heatmap

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_iris = sns.load_dataset("iris")
corr = df_iris.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()
Plotly Interactive Plot

import plotly.express as px
import pandas as pd

df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x='year', y='lifeExp', title='Life Expectancy in Canada')
fig.show()
Dashboard in Jupyter (Using Voila + Widgets)

# Install Voila: pip install voila
import ipywidgets as widgets
from IPython.display import display

slider = widgets.IntSlider(min=0, max=10, step=1, value=5)
text = widgets.Label()

def on_value_change(change):
    text.value = f"Slider value: {change['new']}"

slider.observe(on_value_change, names='value')
display(slider, text)
# Run: voila notebook.ipynb to see a dashboard-like UI
Matplotlib Subplots (Storytelling)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(x, y1, label='sin(x)', color='blue')
axes[0].legend()
axes[0].set_title("Sine Curve")

axes[1].plot(x, y2, label='cos(x)', color='red')
axes[1].legend()
axes[1].set_title("Cosine Curve")
plt.tight_layout()
plt.show()
References & Documentation

Seaborn Documentation
Plotly Documentation
Voila Documentation
Tableau / Power BI
Month 3: Machine Learning — Fundamentals
Week 9: Introduction to Machine Learning
Topics:

Supervised vs. Unsupervised learning
ML workflow (data splitting, training, validation, testing)
Overfitting, underfitting, bias-variance tradeoff
Description:

Overview of machine learning landscape, types of learning, and common pitfalls.
Sample Projects/Programs (Code Examples)

Train/Test Split Example

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
Overfitting vs. Underfitting (Decision Tree Depth)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=42)

depths = range(1, 15)
train_scores = []
test_scores = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.plot(depths, train_scores, label='Train Score')
plt.plot(depths, test_scores, label='Test Score')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Simple Unsupervised Learning Example (KMeans)

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
print("Cluster centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
Bias-Variance Demonstration (Synthetic Data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 20)

X = X[:, np.newaxis]

# Linear model
lin_model = LinearRegression()
lin_model.fit(X, y)

# Polynomial model
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=4)),
    ('lin_reg', LinearRegression())
])
poly_model.fit(X, y)

# Plot
X_plot = np.linspace(-5, 5, 100)[:, np.newaxis]
plt.scatter(X, y, color='black')
plt.plot(X_plot, lin_model.predict(X_plot), label='Linear Model', color='red')
plt.plot(X_plot, poly_model.predict(X_plot), label='Polynomial (deg=4)', color='blue')
plt.legend()
plt.show()
References & Documentation

Scikit-learn Documentation
Machine Learning Crash Course by Google
K-Means Algorithm Explanation
Week 10: Regression Models
Topics:

Linear Regression (OLS, Ridge, Lasso)
Polynomial Regression
Evaluation metrics (MAE, MSE, R²)
Description:

Learn methods for predicting continuous variables and how to measure their accuracy.
Sample Projects/Programs (Code Examples)

Simple Linear Regression (OLS)

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
Ridge Regression

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X = np.random.rand(100, 3)
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(100)*0.1

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
y_pred = ridge_model.predict(X)
print("Ridge MSE:", mean_squared_error(y, y_pred))
Lasso Regression

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

X = np.random.rand(100, 3)
y = 5*X[:,0] - 2*X[:,1] + 0.5*X[:,2] + np.random.randn(100)*0.2

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)
y_pred = lasso_model.predict(X)
print("Lasso MAE:", mean_absolute_error(y, y_pred))
Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.random.rand(30, 1)*5
y = 2 + 1.5 * X**2 + np.random.randn(30,1)*2

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

X_plot = np.linspace(0, 5, 100).reshape(100,1)
X_plot_poly = poly.transform(X_plot)
y_plot = poly_reg.predict(X_plot_poly)

plt.scatter(X, y)
plt.plot(X_plot, y_plot, color='red')
plt.show()
References & Documentation

Scikit-learn Linear Models
Metrics and Scoring in Sklearn
Regularization Techniques Explained
Week 11: Classification Models
Topics:

Logistic Regression
Decision Trees, Random Forests
Performance metrics (accuracy, precision, recall, F1-score, ROC)
Description:

Predict discrete outcomes. Measure performance using relevant metrics.
Sample Projects/Programs (Code Examples)

Logistic Regression on Iris

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X = [[0,0],[1,0],[1,1],[0,1]]
y = [0, 1, 1, 0]  # XOR pattern
clf = DecisionTreeClassifier()
clf.fit(X, y)

plot_tree(clf, filled=True)
plt.show()
print("Predictions:", clf.predict(X))
Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
ROC Curve & AUC

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_scores = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
References & Documentation

Scikit-learn Classification Models
ROC and AUC Explained
Week 12: Model Evaluation & Optimization
Topics:

Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
Cross-validation strategies
Handling class imbalance (SMOTE, undersampling, oversampling)
Description:

Systematic methods to find optimal hyperparameters and ensure robust evaluation.
Sample Projects/Programs (Code Examples)

GridSearchCV Example

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X, y)
print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
RandomizedSearchCV Example

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'max_features': ['auto', 'sqrt', 'log2']
}
rand_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=5, cv=3, random_state=42)
rand_search.fit(X, y)
print("Best Params:", rand_search.best_params_)
print("Best Score:", rand_search.best_score_)
Cross-Validation (K-Fold)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
model = LogisticRegression(max_iter=500)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
Oversampling with SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

lr = LogisticRegression(max_iter=200)
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
References & Documentation

Scikit-learn Model Selection
Imbalanced-learn Over-sampling
Cross-Validation Explained
Month 4: Machine Learning — Advanced Topics
Week 13: Ensemble Methods
Topics:

Bagging (Random Forest)
Boosting (XGBoost, LightGBM, CatBoost)
Stacking
Description:

Combine multiple models for better predictive performance. Explore bagging, boosting, and stacking.
Sample Projects/Programs (Code Examples)

Bagging Classifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred))
Random Forest Review

from sklearn.ensemble import RandomForestClassifier
# Similar to Week 11 example, just reaffirming concept of bagging
XGBoost

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
Stacking Classifier

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]
stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stack_clf.fit(X_train, y_train)
y_pred = stack_clf.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred))
References & Documentation

XGBoost Documentation
LightGBM Documentation
CatBoost Documentation
Scikit-learn Ensemble Methods
Week 14: Unsupervised Learning
Topics:

Clustering (K-means, Hierarchical, DBSCAN)
Dimensionality reduction (PCA, t-SNE)
Anomaly detection
Description:

Discover hidden patterns in unlabeled data through clustering and dimensionality reduction.
Sample Projects/Programs (Code Examples)

K-Means Clustering

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=200, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
labels = kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='x')
plt.show()
Hierarchical Clustering (Agglomerative)

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

X = np.random.rand(20, 2)
Z = linkage(X, 'ward')
dendrogram(Z)
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(X)
print("Cluster labels:", labels)
DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.05)
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='plasma')
plt.show()
PCA for Dimensionality Reduction

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='Paired')
plt.colorbar()
plt.title("PCA of Digits Dataset")
plt.show()
References & Documentation

Scikit-learn Clustering
PCA Explained
DBSCAN vs. K-Means Discussion
Week 15: Time Series Analysis
Topics:

Time series components (trend, seasonality)
ARIMA, SARIMA, Prophet
Forecasting techniques and error metrics (MAE, MAPE)
Description:

Analyze and forecast data that changes over time using specialized models.
Sample Projects/Programs (Code Examples)

ARIMA with statsmodels

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Example time series
data = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
model = ARIMA(data['Passengers'], order=(1,1,1))
results = model.fit()
print(results.summary())
Seasonal Decomposition

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
decomposition = seasonal_decompose(data['Passengers'], model='additive')
decomposition.plot()
plt.show()
SARIMA

from pmdarima import auto_arima
import pandas as pd

data = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
model = auto_arima(data['Passengers'], seasonal=True, m=12)
print("Best Model:", model.summary())
Facebook Prophet

# pip install prophet (formerly fbprophet)
import pandas as pd
from prophet import Prophet

df = pd.read_csv('airline_passengers.csv')
df.columns = ['ds', 'y']  # Prophet expects ds (date) and y (value)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)
model.plot(forecast)
References & Documentation

statsmodels Time Series
pmdarima (Auto-ARIMA)
Facebook Prophet Documentation
Week 16: Recommendation Systems
Topics:

Collaborative Filtering (user-based, item-based)
Content-based recommendations
Hybrid methods
Description:

Develop personalized recommendation algorithms that power modern e-commerce and media platforms.
Sample Projects/Programs (Code Examples)

User-Based Collaborative Filtering

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example user-item matrix
data = {
    'User1': [5,0,3,0],
    'User2': [4,0,0,2],
    'User3': [0,1,5,3]
}
df = pd.DataFrame(data, index=['ItemA','ItemB','ItemC','ItemD']).T
user_sim = pd.DataFrame(cosine_similarity(df), index=df.index, columns=df.index)
print("User-User Similarity:\n", user_sim)
Item-Based Collaborative Filtering

item_df = df.T  # Transpose for item-based
item_sim = pd.DataFrame(cosine_similarity(item_df), index=item_df.index, columns=item_df.index)
print("Item-Item Similarity:\n", item_sim)
Content-Based Recommender

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

documents = [
    "Action adventure film with epic battles",
    "Romantic drama focusing on relationships",
    "Comedy movie with hilarious scenes",
    "Action-packed thriller with suspense"
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("Content Similarity Matrix:\n", cosine_sim)
Hybrid Approach Sketch

# Pseudocode combining user-based and content-based
user_sim_weight = 0.5
content_sim_weight = 0.5

final_score = user_sim.iloc[user_idx, :] * user_sim_weight + content_sim[item_idx, :] * content_sim_weight
recommended_item = final_score.idxmax()
print("Recommended Item:", recommended_item)
References & Documentation

Surprise Library (Recommendation)
Microsoft Recommenders (GitHub)
Introduction to Recommender Systems
Month 5: Deep Learning & Specialized Topics
Week 17: Introduction to Deep Learning
Topics:

Neural network basics (neurons, activation functions)
Forward and backward propagation
Frameworks overview (TensorFlow, Keras, PyTorch)
Description:

Understand the fundamental concepts of neural networks and the major deep learning frameworks.
Sample Projects/Programs (Code Examples)

Simple Feedforward Neural Network in NumPy

import numpy as np

# X: (n_samples, n_features)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR

# Initialize weights randomly
np.random.seed(42)
W1 = np.random.randn(2,2)
b1 = np.random.randn(2)
W2 = np.random.randn(2,1)
b2 = np.random.randn(1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

# Train (very simple approach)
lr = 0.1
for _ in range(10000):
    # Forward
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    # Backprop
    dz2 = (a2 - y) * a2*(1-a2)
    dW2 = a1.T.dot(dz2)
    db2 = dz2.sum(axis=0)

    dz1 = dz2.dot(W2.T)*a1*(1-a1)
    dW1 = X.T.dot(dz1)
    db1 = dz1.sum(axis=0)

    # Gradient descent
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("Final predictions:", a2)
Hello, TensorFlow

import tensorflow as tf

# Simple constant
a = tf.constant(5)
b = tf.constant(3)
c = a + b
print("Sum:", c.numpy())
Hello, PyTorch

import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2
print("y:", y)

# Sum of y
s = y.sum()
s.backward()  # compute gradients
print("x.grad:", x.grad)
Keras Sequential Model

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X, y, epochs=10)  # X, y from some dataset
References & Documentation

TensorFlow
PyTorch
Keras
Week 18: Convolutional Neural Networks (CNNs)
Topics:

Convolution layers, pooling, padding, strides
Popular architectures (LeNet, AlexNet, VGG, ResNet)
Image classification tasks
Description:

CNNs are specialized for image-like data, learning to detect spatial patterns through convolution filters.
Sample Projects/Programs (Code Examples)

Basic CNN in Keras

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, validation_split=0.1)
Loading MNIST Dataset

import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
Data Augmentation Example

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
# datagen.fit(train_images)  # For example
Transfer Learning with ResNet50

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
References & Documentation

Keras CNN Guide
PyTorch Vision
Transfer Learning in Keras
Week 19: Natural Language Processing (NLP)
Topics:

Text preprocessing (tokenization, stemming, lemmatization)
Word embeddings (Word2Vec, GloVe)
Modern NLP (Transformers, BERT, GPT - high-level overview)
Description:

Process and analyze text data. Learn traditional NLP approaches (stemming, embeddings) and modern transformer-based methods.
Sample Projects/Programs (Code Examples)

Text Preprocessing (NLTK)

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "This is a sample sentence, showing off the stop words filtration."
tokens = word_tokenize(text.lower())
sw = set(stopwords.words('english'))
ps = PorterStemmer()

processed = [ps.stem(w) for w in tokens if w.isalpha() and w not in sw]
print(processed)
Word2Vec with Gensim

import gensim
from gensim.models import Word2Vec

sentences = [["dog", "bites", "man"], ["man", "bites", "dog"], ["dog", "eats", "food"], ["man", "eats", "food"]]
model = Word2Vec(sentences, vector_size=50, min_count=1, window=2)
vector = model.wv['dog']
print("Word Vector for 'dog':", vector)
print("Most similar to 'dog':", model.wv.most_similar('dog'))
Simple Sentiment Analysis (Naive Bayes)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this movie", "I hate this movie", "This movie is great", "This movie is terrible"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

test_text = ["I love it", "terrible choice"]
test_vec = vectorizer.transform(test_text)
preds = model.predict(test_vec)
print(preds)
Transformers (Hugging Face)

# pip install transformers
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely love this product!")
print(result)
References & Documentation

NLTK Documentation
Gensim Documentation
Hugging Face Transformers
Week 20: Advanced Deep Learning Topics
Topics:

Recurrent Neural Networks (RNN, LSTM, GRU) for sequences
Transfer learning
Generative models (GANs) overview
Description:

Explore specialized architectures for sequence data and generative models. Reuse pre-trained networks for performance gains.
Sample Projects/Programs (Code Examples)

Simple LSTM for Sequence Prediction

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Dummy sequential data
X = np.array([[[i], [i+1], [i+2]] for i in range(100)])
y = np.array([i+3 for i in range(100)])

model = Sequential()
model.add(LSTM(10, input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)
prediction = model.predict(np.array([[[100],[101],[102]]]))
print("Next value:", prediction)
GRU Example

from tensorflow.keras.layers import GRU

# Similar structure to LSTM, just replace LSTM with GRU
# model.add(GRU(10, input_shape=(3,1)))
Transfer Learning Example (Fine-Tuning)

from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
# unfreeze the last block for fine-tuning
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
GAN (Conceptual Sketch)

import tensorflow as tf

# Pseudocode for building generator & discriminator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(784, activation='sigmoid')  # For MNIST
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Then define the training loop to alternate between generator and discriminator
References & Documentation

Keras RNN Guide
Transfer Learning & Fine Tuning
GAN Tutorial (TensorFlow)
Month 6: Deployment, Big Data, and Final Projects
Week 21: Model Deployment & MLOps
Topics:

Saving and loading models (Pickle, joblib)
Deployment strategies (Flask, FastAPI, Docker)
CI/CD for ML, version control, monitoring
Description:

Learn how to serve ML models in production, containerize them, and establish continuous integration/delivery.
Sample Projects/Programs (Code Examples)

Pickle Model Save/Load

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("Loaded model accuracy:", loaded_model.score(X, y))
Flask API for Inference

# app.py
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
FastAPI Example

# main.py
from fastapi import FastAPI
import pickle

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])[0]
    return {"prediction": int(prediction)}
Dockerfile

# Use a lightweight Python image
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "app.py"]
References & Documentation

Pickle Python Docs
Flask Docs
FastAPI Docs
Docker Documentation
Week 22: Big Data Ecosystem
Topics:

Hadoop & MapReduce fundamentals
Spark for data processing and MLlib
Distributed storage and computing
Description:

Introduction to large-scale data processing using Hadoop and Spark. Learn to handle big data efficiently.
Sample Projects/Programs (Code Examples)

PySpark Setup

# Terminal: start PySpark
pyspark
Spark DataFrame Basics

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkDF").getOrCreate()
data = [("Alice", 29), ("Bob", 35), ("Cathy", 30)]
columns = ["Name", "Age"]
df = spark.createDataFrame(data, columns)
df.show()
Spark MLlib Example

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

spark_data = [
    Row(features=Vectors.dense([0.0, 1.1, 0.1]), label=0.0),
    Row(features=Vectors.dense([2.0, 1.0, -1.0]), label=1.0),
    # More rows...
]
training_df = spark.createDataFrame(spark_data)
lr = LogisticRegression(maxIter=10, regParam=0.001)
model = lr.fit(training_df)
RDD MapReduce-like Operation

rdd = spark.sparkContext.parallelize([("cat",1),("dog",1),("cat",1)])
rdd_reduced = rdd.reduceByKey(lambda a,b: a+b)
print(rdd_reduced.collect())  # [('cat', 2), ('dog', 1)]
References & Documentation

Apache Hadoop
Apache Spark
Spark MLlib
Week 23: Cloud Platforms & Data Engineering
Topics:

Cloud services overview (AWS, Azure, GCP)
Data engineering pipelines (ETL, Airflow)
Data security and governance
Description:

Learn how to build and schedule data workflows in the cloud, focusing on secure and scalable data pipelines.
Sample Projects/Programs (Code Examples)

AWS S3 Upload (boto3)

import boto3

s3 = boto3.client('s3')
s3.upload_file('local_file.csv', 'my-bucket', 'data/local_file.csv')
GCP BigQuery Ingestion

from google.cloud import bigquery

client = bigquery.Client()
table_id = "project_id.dataset.table"
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
)
with open("data.csv", "rb") as source_file:
    job = client.load_table_from_file(source_file, table_id, job_config=job_config)
job.result()
Airflow DAG Definition

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1)
}

def extract_data(**kwargs):
    print("Extract data from source")

with DAG('my_etl_dag', default_args=default_args, schedule_interval='@daily') as dag:
    extract_task = PythonOperator(
        task_id='extract_task',
        python_callable=extract_data
    )
    extract_task
Data Governance Example (Pseudocode)

# This is conceptual, showcasing how you might enforce access policies
# or encryption in code for compliance (GDPR, HIPAA, etc.)
def encrypt_data(data):
    # Use a library like cryptography
    pass

def check_access(user_role):
    # Check user_role from IAM or Access Control
    pass
References & Documentation

AWS Documentation
Azure Documentation
Google Cloud Documentation
Apache Airflow
Week 24: Capstone Project & Review
Topics:

Capstone project development
Final review (ML, deep learning, data wrangling, deployment)
Presentation and job preparation (portfolio, resume, interviews)
Description:

Integrate all skills into a final real-world project. Polish presentation skills and prepare for job searches.
Sample Projects/Programs (Code Examples)

Note: In Week 24, the focus is on developing a comprehensive Capstone Project. Below are ideas/snippets that might appear in a final project, but the actual code depends on your chosen dataset and approach.

Data Ingestion & Wrangling

# Pseudocode for a final pipeline
import pandas as pd
df = pd.read_csv('raw_data.csv')
# Clean, wrangle, feature engineer
# ...
df.to_csv('clean_data.csv', index=False)
Model Training

# Train multiple models and compare performance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print("F1 Score:", f1_score(y_test, preds))
Deployment Script

# E.g., a simple FastAPI script that loads the best model and serves predictions
Presentation Notebook

# Jupyter Notebook focusing on:
# 1. Problem Statement
# 2. EDA
# 3. Modeling
# 4. Evaluation
# 5. Conclusion & Next Steps
References & Documentation

Review all previous weeks’ links and documentation.
Kaggle Datasets
MLflow for tracking experiments
Final Thoughts
This extended curriculum includes:

Weekly topics covering the breadth of data science skills.
Code samples (4 per week) to reinforce hands-on learning.
References to official documentation and popular open-source libraries.
By the end of 6 months, you should have the foundational and advanced knowledge to tackle real-world data challenges, build and deploy machine learning models, and confidently present your work to potential employers or clients.
