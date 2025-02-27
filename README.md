# Project-4
Udacity Capstone
## Customer Churn Prediction
Project Overview
This project aims to develop a machine learning model to predict customer churn, enabling businesses to identify at-risk customers before they leave. By leveraging data from the Telco Customer Churn dataset, the project explores various machine learning techniques to build an effective churn prediction model.

**Table of Contents**
Project Overview
Data Collection and Preprocessing
Exploratory Data Analysis (EDA)
Modelling and Evaluation
Key Findings
Future Work
Installation
Usage
Contributing
License

## Data Collection and Preprocessing
The dataset used in this project is the Telco Customer Churn dataset, which includes information about customer demographics, account information, and services used. The preprocessing steps include:

Handling missing values
Encoding categorical variables
Scaling numerical features
Balancing the dataset using SMOTE

## Exploratory Data Analysis (EDA)
EDA is performed to understand the data distribution, identify patterns, and select relevant features. Key factors influencing churn, such as tenure, contract type, and payment method, are analyzed using visualizations and statistical summaries.

## Modeling and Evaluation
Multiple machine learning models are trained and evaluated, including Random Forest and XGBoost. The models are compared using metrics like accuracy, recall, precision, and ROC-AUC. Hyperparameter tuning is performed to optimize the models' performance.

## Key Findings
**XGBoost Performance:** XGBoost outperformed Random Forest, achieving an accuracy of 79.99% and a ROC-AUC score of 0.7387.
**Recall Challenges:** Both models struggled with recall, indicating that many actual churners were still missed (61% for XGBoost, 58% for Random Forest).
**Influential Factors:** Key factors influencing churn included tenure, contract type, and payment method, with month-to-month contracts and electronic check payments linked to higher churn rates.

## Future Work
**Improving Recall:** Further improvements could involve adjusting the decision threshold, enhancing feature engineering, or exploring advanced models to improve recall.
**Deployment:** Deploy the best-performing model and monitor its performance over time.
**Feature Engineering:** Experiment with creating new features or transforming existing ones to improve model performance.

##Installation
To run this project locally, follow these steps:

Clone the repository:
git clone https://github.com/Mazi2333/Project-4.git
Navigate to the project directory:
cd customer-churn-prediction
Install the required dependencies:
pip install -r requirements.txt
Usage
To use the churn prediction model, follow these steps:

Load the dataset and preprocess it using the provided scripts.
Train the model using the training script.
Evaluate the model using the evaluation script.
Use the trained model to predict churn for new customers.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Medium Blog Post 
For some insight on the project please visit the (https://medium.com/@mcmolebatsi/predicting-customer-churn-a-data-science-approach-8158cd7b8b45)
