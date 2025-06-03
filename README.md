# Obesity Risk Prediction and Analysis System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Pandas](https://img.shields.io/badge/pandas-2.0%2B-blueviolet)

A machine learning-powered web application for analyzing obesity risk factors and predicting obesity levels based on lifestyle and demographic characteristics.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

## Features

📊 **Comprehensive Data Analysis**
- Exploratory data analysis with visualizations
- Correlation matrix of key obesity factors
- Distribution analysis of BMI and obesity classes

🤖 **Multiple ML Models**
- Random Forest (Classification & Regression)
- Gradient Boosting (Classification & Regression)
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Logistic Regression
- Decision Trees

🔍 **Model Evaluation**
- ROC curves and AUC scores
- Confusion matrices
- Feature importance analysis
- Cross-validation results

📝 **Interactive Prediction**
- User-friendly input form for personal data
- Real-time obesity risk prediction
- Probability distribution across classes
- Personalized health recommendations

## Dataset

The system uses the [Obesity Prediction Dataset](https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset) containing:

- **16 demographic and lifestyle features**:
  - Age, Gender, Height, Weight
  - Family history of overweight
  - Dietary habits (FAVC, FCVC, NCP)
  - Physical activity (FAF)
  - Technology usage (TUE)
  
- **Target variables**:
  - BMI (for regression)
  - Obesity class (for classification)

## Models

| Model Type       | Algorithms                          | Best Accuracy |
|------------------|-------------------------------------|---------------|
| Classification   | Random Forest, Gradient Boosting    | 92%           |
|                  | KNN, SVM, Logistic Regression       | 88%           |
| Regression       | Random Forest, Gradient Boosting    | R² 0.89       |
|                  | Decision Trees                      | R² 0.82       |

## Installation

1. Clone the repository:
```bash
git clone this-repo
cd nirs
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. d
```bash
pip install -r requirements.txt
```

## Usage 

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at http://localhost:8501

Navigate through the tabs:

Data Overview: Explore dataset statistics and visualizations

Model Analysis: Compare different ML models

Prediction Tool: Input your data for personalized obesity risk assessment

Recommendations: Get customized health advice

Project Structure
obesity-prediction/
├── data/                   # Dataset files
│   └── obesity_train.csv
├── app.py                  # Main Streamlit application
├── utils/                  # Utility functions
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   └── visualization.py    # Plotting functions
├── models/                 # Saved model files
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


## Results

Key Findings:
1. Family history is the strongest predictor of obesity (feature importance: 0.28)
2. Physical activity and water consumption show strong negative correlation with obesity
3. The Gradient Boosting classifier achieved the highest ROC AUC (0.96)

## License

MIT License

Copyright (c) 2025 Borizzler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.