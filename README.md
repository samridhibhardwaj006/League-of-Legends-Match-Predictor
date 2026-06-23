# 🎮 League of Legends Match Outcome Prediction

Predicting the winner of a League of Legends match using machine learning and in-game performance metrics.

---

## 📖 Introduction

League of Legends is one of the world's most popular esports titles, generating millions of matches and vast amounts of gameplay data. Understanding which factors contribute to victory can provide valuable insights for players, analysts, and teams.

This project develops a machine learning model capable of predicting match outcomes based on match statistics. Using Logistic Regression implemented in PyTorch, the model learns patterns associated with winning and losing games and evaluates its performance using standard classification metrics.

---

## 🎯 Objectives

The primary goals of this project are:

* Analyze League of Legends match data.
* Identify gameplay features that influence match outcomes.
* Build a binary classification model to predict winners.
* Evaluate model performance using multiple metrics.
* Interpret feature importance to understand key predictors.

---

## 📊 Dataset

The dataset contains match-level statistics collected from League of Legends games.

### Features Included

Examples of attributes used for prediction:

* Gold earned
* Kills
* Deaths
* Assists
* Damage dealt
* Objectives secured
* Team performance metrics

### Target Variable

* **Win = 1**
* **Loss = 0**

---

## ⚙️ Methodology

### 1. Data Preprocessing

Before training the model:

* Missing values were handled.
* Relevant features were selected.
* Data was standardized using feature scaling.
* The dataset was split into training and testing sets.

### 2. Model Development

A Logistic Regression classifier was implemented using PyTorch.

Key components include:

* Linear layer
* Sigmoid activation
* Binary Cross Entropy Loss
* Gradient Descent Optimization

### 3. Regularization

L2 Regularization was introduced to reduce overfitting and improve model generalization.

### 4. Hyperparameter Tuning

Different learning rates and training configurations were explored to improve predictive performance.

---

## 📈 Model Evaluation

The model was assessed using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve
* Area Under Curve (AUC)

These metrics provide a comprehensive understanding of model effectiveness and classification quality.

---

## 🔍 Feature Importance Analysis

After training, model coefficients were analyzed to determine which gameplay statistics had the strongest influence on match outcomes.

This analysis helps identify:

* Key winning indicators
* Important strategic factors
* High-impact gameplay metrics

---

## 🛠️ Technologies Used

| Tool             | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Programming Language       |
| PyTorch          | Model Development          |
| Pandas           | Data Manipulation          |
| Scikit-Learn     | Evaluation & Preprocessing |
| Matplotlib       | Visualization              |
| Jupyter Notebook | Development Environment    |

---

## 📁 Project Structure

```text
League-of-Legends-Match-Predictor/
│
├── league-of-legends-match-predictor.ipynb
├── README.md
├── requirements.txt
└── saved_model.pth
```

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

* Binary Classification Problems
* Machine Learning Workflows
* Data Preprocessing Techniques
* Model Evaluation Strategies
* Hyperparameter Optimization
* Interpreting Model Predictions

---

## 🚀 Future Improvements

Potential enhancements include:

* Testing advanced models such as Random Forests and XGBoost.
* Incorporating additional match statistics.
* Building a real-time prediction dashboard.
* Comparing multiple machine learning approaches.

---

## 👨‍💻 Author

Samridhi Bhardwaj


Developed as part of academic coursework focused on predictive analytics and machine learning applications in esports.
