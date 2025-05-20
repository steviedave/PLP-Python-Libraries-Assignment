# Iris Dataset Analysis and Visualization

## 📘 Project Overview

This project demonstrates how to load, analyze, and visualize data using Python libraries **Pandas**, **Matplotlib**, and **Seaborn**.I have used the **Iris dataset**, a well-known dataset in machine learning, for this purpose.

---

## 🎯 Objectives

- Load and explore a structured dataset.
- Perform basic data analysis using statistical methods.
- Visualize trends, distributions, and relationships using various plot types.
- Derive observations and patterns from the dataset.

---

## 📂 Dataset

The Iris dataset is built into the `sklearn.datasets` module and contains:
- 150 records
- 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
- 3 species of flowers: *setosa*, *versicolor*, *virginica*

---

## 🛠️ Tasks Breakdown

### Task 1: Data Loading and Exploration
- Load dataset using sklearn
- Inspect first few rows
- Check data types and missing values

### Task 2: Basic Data Analysis
- Compute descriptive statistics
- Group by species and calculate means
- Summarize insights

### Task 3: Data Visualization
- Line Chart: Trends over record index
- Bar Chart: Avg. petal length per species
- Histogram: Sepal length distribution
- Scatter Plot: Sepal vs petal length correlation

---

## 📊 Key Observations

- *Iris-virginica* species exhibits the largest petal dimensions.
- Petal-based features offer better separation among species.
- A strong positive correlation exists between sepal and petal lengths.

---

## 🧰 Libraries Used

- pandas
- matplotlib
- seaborn
- sklearn

---

## 📝 How to Run

```bash
python iris_analysis.py
```

Or open the `.ipynb` file in Jupyter Notebook for an interactive experience.

---

## 📎 Submission Includes

- `iris_analysis.py`: Main Python script
- `README.md`: Project documentation
