
# Iris Dataset Analysis with Pandas and Visualization with Matplotlib

# Task 1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset with error handling
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print("Failed to load dataset:", e)

# Display first few rows
print("\nFirst five rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis

# Basic statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping by species and calculating mean
grouped = df.groupby('species').mean()
print("\nAverage values per species:")
print(grouped)

# Observation
print("\nObservation:")
print("Iris-virginica generally has the largest petal length and width.")

# Task 3: Data Visualization

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line Chart: Sepal and Petal Length over Index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sepal length (cm)"], label='Sepal Length')
plt.plot(df.index, df["petal length (cm)"], label='Petal Length')
plt.title("Line Chart: Sepal vs Petal Length (Indexed as Time)")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average Petal Length per Species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped["petal length (cm)"], palette="Set2")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: Sepal Length Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["sepal length (cm)"], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
plt.title("Scatter Plot: Sepal vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 📝 Final Observations

print("\nFinal Observations:")
print("- Iris-virginica species typically shows the largest petal length and width.")
print("- Petal features are more effective in distinguishing species than sepal features.")
print("- Sepal length and petal length show a strong positive correlation.")
