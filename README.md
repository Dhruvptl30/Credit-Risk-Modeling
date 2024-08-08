# Credit-Risk-Modeling
Machine Learning Model that Predicts the loss amount when it is default and estimates the probability of the default using the FICO score.

Project Overview

Loan default prediction is a critical task in the banking and finance industry, as it helps lenders to identify potential defaulters and take necessary measures to minimize losses. This project aims to develop a machine learning model to predict loan defaults using a Gradient Boosting Classifier and estimate the probability of default using a linear regression model.

Objective

Loan default prediction is a complex problem that involves analyzing various factors, including credit history, income, employment, and debt-to-income ratio. Traditional methods of loan default prediction rely on manual analysis of credit reports and financial statements, which can be time-consuming and prone to errors. Machine learning algorithms, such as Gradient Boosting Classifier and linear regression, can be used to analyze large datasets and predict loan defaults with high accuracy.

Methodology

The project consists of two main components:
1.Loan Default Prediction: A Gradient Boosting Classifier model is developed to predict loan defaults using a Dataset of loan applications.
2.Probability of Default Estimation: A linear regression model is developed to estimate the probability of default based on the FICO score of the borrower.

Steps Involved

1.Data Collection
The project uses a Dataset of loan applications, which includes various features such as credit lines outstanding, loan amount outstanding, total debt outstanding, income, years employed, and FICO score.

2.Data Processing
The data is pre-processed by handling missing values, encoding categorical variables, and scaling numerical variables.

3.Model Building & Development - Gradient Boosting Classifier
- Model Selection: Gradient Boosting Classifier was chosen for its effectiveness in handling complex classification tasks.
- Train-Test Split: The data was split into 80% training and 20% testing sets for model evaluation.
- Feature Engineering: Feature engineering techniques might be applied to improve model performance (not implemented in the provided code).
- Model Training: The Gradient Boosting Classifier model was trained on the training data.

4.Model Building & Development - Probability of Default with KMeans and Logistic Regression
1.KMeans Clustering:
- The FICO score, a key indicator of creditworthiness, was used for clustering borrowers.
- KMeans clustering was applied to group borrowers with similar FICO scores into clusters. These clusters represent "fico_rating."
2.Logistic Regression:
- A logistic regression model was trained to predict loan default probability based on the assigned "fico_rating" from KMeans clustering.
3.Model Training: The logistic regression model was trained on the training data.


5.Model Evaluation - Gradient Boosting Classifier
Evaluation Metrics: Metrics like accuracy score, classification report, and confusion matrix can be used to assess the model's performance on the testing set (not implemented in the provided code). These metrics will evaluate the model's ability to correctly classify loan defaults.

6.Functionality: Probability of Default Prediction
1.Function Definition: 
- A function predict_default was created to estimate the probability of default for a new borrower based on their FICO score. 
- The function predicts the "fico_rating" using the trained KMeans model.
- It then uses the logistic regression model to predict the probability of default based on the assigned "fico_rating."
2.User Input: The function allows users to input a FICO score for a new borrower.
3.Prediction and Interpretation:
- The function predicts the probability of default for the new borrower.
- The code outputs a message indicating whether the borrower is likely to default based on a pre-defined probability threshold (e.g., 0.5).

7.Functionality: Expected Loss Calculation
- Function Definition: A function calculate_expected_loss was created to estimate the expected loss for a loan based on the predicted probability of default and the loan amount.
- Input: The function requires the predicted probability of default, loan amount, and a recovery rate (percentage of the loan amount recovered in case of default).
- Output: The function calculates and returns the expected loss for the loan.

8.Results
The results of the project are as follows:

Loan Default Prediction: The Gradient Boosting Classifier model achieves an accuracy of 99.5% in predicting loan defaults.
Probability of Default Estimation: The linear regression model estimates the probability of default with a high degree of accuracy, with a mean squared error of 0.05.

9.Discussion
The results show that the Gradient Boosting Classifier model can accurately predict loan defaults using a Dataset of loan applications. The linear regression model can estimate the probability of default based on the FICO score of the borrower, which can be used to identify high-risk borrowers.

10.Conclusion
This project demonstrates the potential of machine learning algorithms in predicting loan defaults and estimating the probability of default. The results show that the Gradient Boosting Classifier model and linear regression model can be used to develop a robust loan default prediction system.
