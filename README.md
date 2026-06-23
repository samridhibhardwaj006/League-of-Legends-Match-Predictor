# League-of-Legends-Match-Predictor

🎮 League of Legends Match Predictor
A machine learning project that predicts the outcome of League of Legends matches using in-game statistics and a Logistic Regression model built with PyTorch.


📌 Project Overview
League of Legends generates a large amount of match data that can be used to identify patterns associated with winning and losing games. This project applies machine learning techniques to predict match outcomes based on gameplay features.
The model is trained using PyTorch and evaluated using standard classification metrics to assess its predictive performance.


🚀 Features
Data preprocessing and normalization
Logistic Regression implementation with PyTorch
Model training and optimization
L2 Regularization to reduce overfitting
Performance evaluation using:
Accuracy Score
Confusion Matrix
Classification Report
ROC Curve & AUC
Hyperparameter tuning
Feature importance analysis
Model persistence (Save & Load trained model)


🛠️ Technologies Used
Python
PyTorch
Pandas
Scikit-learn
Matplotlib
Jupyter Notebook


📂 Project Structure
├── Final Project League of Legends Match Predictor.ipynb
├── lol_model.pth
├── README.md
└── requirements.txt


📊 Machine Learning Workflow
1. Data Preparation
Load match dataset
Select relevant features
Normalize data using StandardScaler
Split into training and testing sets
2. Model Development
Build a Logistic Regression model in PyTorch
Define loss function and optimizer
Train using gradient descent
3. Model Evaluation
Measure prediction accuracy
Generate confusion matrix
Plot ROC curve
Analyze classification performance
4. Model Optimization
Apply L2 Regularization
Compare performance across learning rates
Select best hyperparameters
5. Feature Importance
Extract model weights
Identify the most influential game statistics


📈 Results
The trained model successfully learns relationships between in-game statistics and match outcomes, demonstrating how machine learning can be applied to esports analytics.
Key evaluation methods include:
Confusion Matrix
ROC-AUC Analysis
Classification Report
Feature Importance Ranking


🎯 Learning Outcomes
Through this project, I gained hands-on experience with:
Binary Classification
PyTorch Model Development
Model Optimization Techniques
Hyperparameter Tuning
Feature Engineering
Machine Learning Evaluation Metrics


📚 Course Project
This project was completed as part of a Machine Learning coursework assignment focused on applying predictive analytics to real-world datasets.
👨‍💻 Author
Samridhi Bhardwaj
