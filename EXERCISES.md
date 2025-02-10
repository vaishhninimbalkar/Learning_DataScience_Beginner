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
```python
import numpy as np

# Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C = A x B:\n", C)
```

Basic Probability Calculation
Calculates simple probabilities for coin toss outcomes.
```python
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
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data, bins=30, alpha=0.7, color='blue')
plt.title("Normal Distribution (Mean=0, Std=1)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

References & Documentation

Python Official Documentation
NumPy Documentation
Pandas Documentation
Jupyter Documentation

### Week 2: Python Essentials for Data Science

**Topics:**

- Python basics (data types, loops, functions, modules)
- Python data libraries (NumPy, Pandas, Matplotlib)
- Data structures (lists, dictionaries, sets)

**Description:**

- Build a strong foundation in Python programming.
- Learn to use libraries for data manipulation (NumPy, Pandas) and visualization (Matplotlib).

**Sample Projects/Programs (Code Examples)**

Python Basics: Data Types & Loops
```python
# Simple list and loop example
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(f"{i} -> {fruit}")
```

Defining and Importing a Custom Module
Create a simple module, my_math.py, and import it.
```python
# my_math.py
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b
```
```python
# main.py
import my_math

print(my_math.add_numbers(3, 5))
print(my_math.multiply_numbers(4, 7))
```

```python
# Pandas DataFrame Creation
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)
print(df)
```

```python
# Matplotlib Bar Chart
import matplotlib.pyplot as plt

cities = ["New York", "Los Angeles", "Chicago"]
population = [8.4, 4.0, 2.7]  # in millions

plt.bar(cities, population, color='green')
plt.title("Population by City")
plt.xlabel("City")
plt.ylabel("Population (millions)")
plt.show()
```

References & Documentation

Python Official Documentation
NumPy Documentation
Pandas Documentation
Matplotlib Documentation

### Week 3: Statistics & Probability

**Topics:**

- Descriptive statistics (mean, median, mode, variance, standard deviation)
- Probability distributions (Normal, Binomial, Poisson)
- Inferential statistics (confidence intervals, hypothesis testing)

**Description:**

- Delve into basic and advanced statistical techniques.
- Learn to compute descriptive statistics and understand distributions.

**Sample Projects/Programs (Code Examples)**

Descriptive Statistics with Python
```python
import numpy as np

data = np.random.randint(1, 100, 50)
print("Data:", data)
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Std Dev:", np.std(data))
```

Binomial Distribution Simulation
```python
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
```

Confidence Interval Calculation
```python
import numpy as np
from scipy import stats

sample_data = np.random.normal(loc=10, scale=2, size=50)
mean_est = np.mean(sample_data)
sem = stats.sem(sample_data)  # standard error of the mean
confidence = 0.95
interval = stats.t.interval(confidence, len(sample_data)-1, loc=mean_est, scale=sem)

print(f"Mean estimate: {mean_est:.2f}")
print(f"{int(confidence*100)}% Confidence Interval: {interval}")
```

Hypothesis Testing (t-test)
```python
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
```

References & Documentation

SciPy Stats Documentation
NumPy Documentation
Matplotlib Documentation

### Week 4: Exploratory Data Analysis (EDA)

**Topics:**

- Data cleaning (handling missing data, outliers)
- Data visualization (histograms, box plots, scatter plots)
- Correlation analysis

**Description:**

- Explore and visualize datasets to detect anomalies, relationships, and gain initial insights.

**Sample Projects/Programs (Code Examples)**

Handling Missing Data
```python
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
```

Outlier Detection with Box Plot
```python
import matplotlib.pyplot as plt

data = [1,2,2,3,4,5,20,2,3,4,5,2,100,3,4]
plt.boxplot(data)
plt.title("Outlier Detection via Box Plot")
plt.show()
```

Scatter Plot & Correlation
```python
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
```

Pairwise Plot (Using Seaborn)
```python
import seaborn as sns
import pandas as pd

df_iris = sns.load_dataset("iris")
sns.pairplot(df_iris, hue="species")
plt.show()
```

References & Documentation

Pandas Documentation
Matplotlib Documentation
Seaborn Documentation

### Week 18: Convolutional Neural Networks (CNNs)

**Topics:**

- Convolution layers, pooling, padding, strides
- Popular architectures (LeNet, AlexNet, VGG, ResNet)
- Image classification tasks

**Description:**

- CNNs are specialized for image-like data, learning to detect spatial patterns through convolution filters.

**Sample Projects/Programs (Code Examples)**

Basic CNN in Keras
```python
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
```

Loading MNIST Dataset
```python
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
```

Data Augmentation Example
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
# datagen.fit(train_images)  # For example
```

Transfer Learning with ResNet50
```python
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
```

References & Documentation

Keras CNN Guide
PyTorch Vision
Transfer Learning in Keras

### Week 19: Natural Language Processing (NLP)

**Topics:**

- Text preprocessing (tokenization, stemming, lemmatization)
- Word embeddings (Word2Vec, GloVe)
- Modern NLP (Transformers, BERT, GPT - high-level overview)

**Description:**

- Process and analyze text data. Learn traditional NLP approaches (stemming, embeddings) and modern transformer-based methods.

**Sample Projects/Programs (Code Examples)**

Text Preprocessing (NLTK)
```python
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
```

Word2Vec with Gensim
```python
import gensim
from gensim.models import Word2Vec

sentences = [["dog", "bites", "man"], ["man", "bites", "dog"], ["dog", "eats", "food"], ["man", "eats", "food"]]
model = Word2Vec(sentences, vector_size=50, min_count=1, window=2)
vector = model.wv['dog']
print("Word Vector for 'dog':", vector)
print("Most similar to 'dog':", model.wv.most_similar('dog'))
```

Simple Sentiment Analysis (Naive Bayes)
```python
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
```

Transformers (Hugging Face)
```python
# pip install transformers
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely love this product!")
print(result)
```

References & Documentation

NLTK Documentation
Gensim Documentation
Hugging Face Transformers

### Week 20: Advanced Deep Learning Topics

**Topics:**

- Recurrent Neural Networks (RNN, LSTM, GRU) for sequences
- Transfer learning
- Generative models (GANs) overview

**Description:**

- Explore specialized architectures for sequence data and generative models. Reuse pre-trained networks for performance gains.

**Sample Projects/Programs (Code Examples)**

Simple LSTM for Sequence Prediction
```python
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
```

GRU Example
```python
from tensorflow.keras.layers import GRU

# Similar structure to LSTM, just replace LSTM with GRU
# model.add(GRU(10, input_shape=(3,1)))
```

Transfer Learning Example (Fine-Tuning)
```python
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
```

GAN (Conceptual Sketch)
```python
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
```

References & Documentation

Keras RNN Guide
Transfer Learning & Fine Tuning
GAN Tutorial (TensorFlow)

### Week 21: Model Deployment & MLOps

**Topics:**

- Saving and loading models (Pickle, joblib)
- Deployment strategies (Flask, FastAPI, Docker)
- CI/CD for ML, version control, monitoring

**Description:**

- Learn how to serve ML models in production, containerize them, and establish continuous integration/delivery.

**Sample Projects/Programs (Code Examples)**

Pickle Model Save/Load
```python
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
```

Flask API for Inference
```python
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
```

FastAPI Example
```python
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
```

Dockerfile
```dockerfile
# Use a lightweight Python image
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "app.py"]
```

References & Documentation

Pickle Python Docs
Flask Docs
FastAPI Docs
Docker Documentation

### Week 22: Big Data Ecosystem

**Topics:**

- Hadoop & MapReduce fundamentals
- Spark for data processing and MLlib
- Distributed storage and computing

**Description:**

- Introduction to large-scale data processing using Hadoop and Spark. Learn to handle big data efficiently.

**Sample Projects/Programs (Code Examples)**

PySpark Setup
```python
# Terminal: start PySpark
pyspark
```

Spark DataFrame Basics
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkDF").getOrCreate()
data = [("Alice", 29), ("Bob", 35), ("Cathy", 30)]
columns = ["Name", "Age"]
df = spark.createDataFrame(data, columns)
df.show()
```

Spark MLlib Example
```python
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
```

RDD MapReduce-like Operation
```python
rdd = spark.sparkContext.parallelize([("cat",1),("dog",1),("cat",1)])
rdd_reduced = rdd.reduceByKey(lambda a,b: a+b)
print(rdd_reduced.collect())  # [('cat', 2), ('dog', 1)]
```

References & Documentation

Apache Hadoop
Apache Spark
Spark MLlib

### Week 23: Cloud Platforms & Data Engineering

**Topics:**

- Cloud services overview (AWS, Azure, GCP)
- Data engineering pipelines (ETL, Airflow)
- Data security and governance

**Description:**

- Learn how to build and schedule data workflows in the cloud, focusing on secure and scalable data pipelines.

**Sample Projects/Programs (Code Examples)**

AWS S3 Upload (boto3)
```python
import boto3

s3 = boto3.client('s3')
s3.upload_file('local_file.csv', 'my-bucket', 'data/local_file.csv')
```

GCP BigQuery Ingestion
```python
from google.cloud import bigquery

client = bigquery.Client()
table_id = "project_id.dataset.table"
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
)
with open("data.csv", "rb") as source_file:
    job = client.load_table_from_file(source_file, table_id, job_config=job_config)
job.result()
```

Airflow DAG Definition
```python
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
```

Data Governance Example (Pseudocode)
```python
# This is conceptual, showcasing how you might enforce access policies
# or encryption in code for compliance (GDPR, HIPAA, etc.)
def encrypt_data(data):
    # Use a library like cryptography
    pass

def check_access(user_role):
    # Check user_role from IAM or Access Control
    pass
```

References & Documentation

AWS Documentation
Azure Documentation
Google Cloud Documentation
Apache Airflow

### Week 24: Capstone Project & Review

**Topics:**

- Capstone project development
- Final review (ML, deep learning, data wrangling, deployment)
- Presentation and job preparation (portfolio, resume, interviews)

**Description:**

- Integrate all skills into a final real-world project. Polish presentation skills and prepare for job searches.

**Sample Projects/Programs (Code Examples)**

Note: In Week 24, the focus is on developing a comprehensive Capstone Project. Below are ideas/snippets that might appear in a final project, but the actual code depends on your chosen dataset and approach.

Data Ingestion & Wrangling
```python
# Pseudocode for a final pipeline
import pandas as pd
df = pd.read_csv('raw_data.csv')
# Clean, wrangle, feature engineer
# ...
df.to_csv('clean_data.csv', index=False)
```

Model Training
```python
# Train multiple models and compare performance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print("F1 Score:", f1_score(y_test, preds))
```

Deployment Script

```python
# E.g., a simple FastAPI script that loads the best model and serves predictions
```

Presentation Notebook

```python
# Jupyter Notebook focusing on:
# 1. Problem Statement
# 2. EDA
# 3. Modeling
# 4. Evaluation
# 5. Conclusion & Next Steps
```

References & Documentation

Review all previous weeks’ links and documentation.
Kaggle Datasets
MLflow for tracking experiments

Final Thoughts

This extended curriculum includes:

- Weekly topics covering the breadth of data science skills.
- Code samples (4 per week) to reinforce hands-on learning.
- References to official documentation and popular open-source libraries.

By the end of 6 months, you should have the foundational and advanced knowledge to tackle real-world data challenges, build and deploy machine learning models, and confidently present your work to potential employers or clients.
