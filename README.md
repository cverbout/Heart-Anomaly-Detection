# Machine Learning for Heart Anomaly Detection

## Project Overview

This project aims to demonstrate the capablities of machine learning in medical fields through predicting the diagnosis of heart anomalies given SPECT radiography data. Leveraging a dataset made publicly available by Kurgan et al., the goal is to construct a machine learning model capable of achieving at least 70% accuracy in distinguishing between anomalous and normal instances.

## Features

- **Heart Anomaly Classification**: Utilizes machine learning to accurately classify heart anomalies in SPECT radiography data.
- **Decision Tree Classifier**: Employs a non-linear Decision Tree (ID3) model, known for its effectiveness in such classification tasks.
- **Robust Validation Method**: Given the limited dataset size (267 instances), the project employs 10-fold cross-validation for reliable accuracy estimation.
- **Data Preprocessing**: Involves careful selection and preparation of features and target columns from the dataset.

## Technical Stack

- **Python**: The primary programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: For implementing the Decision Tree Classifier and conducting cross-validation.

## Data Source

The dataset, `heart-anomalies.csv`, is available in this GitHub repository. It contains preprocessed SPECT radiography data suitable for machine learning applications provided by Bart Massey.

## How to Run

To run this project:
1. Ensure Python and the required libraries (Pandas, Scikit-Learn) are installed.
2. Clone the repository.
3. Execute the script from your terminal: ```HeartAnomalies.py```

   
## Project Results

The project reaches 85% in classifying heart anomalies surpassing the target accuracy of 70%, and demonstrating the potential of machine learning in medical diagnostics. The output includes cross-validation scores and a test set score, providing a comprehensive evaluation of the model's performance.
