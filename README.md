# Osteoporosis Risk Prediction - Optimizing Machine Learning Models for Early Diagnosis

## Overview
This project aims to significantly advance the field of osteoporosis risk assessment by employing sophisticated machine learning algorithms. By analyzing and interpreting complex datasets related to patient lifestyle, medical history, and other relevant factors, the project seeks to improve the accuracy and reliability of predicting an individual's risk of developing osteoporosis.

The ultimate goal is to empower healthcare providers with a robust predictive tool that enables earlier and more targeted interventions. This proactive approach not only aids in the prevention of osteoporosis but also contributes to better overall patient management and outcomes, reducing the long-term burden on both individuals and healthcare systems. Through the integration of these cutting-edge technologies, the project aspires to make a meaningful impact on public health by enhancing the precision of osteoporosis risk prediction and enabling timely, personalized care strategies.

## Problem Statement
Osteoporosis often goes undiagnosed until its advanced stages, leading to significant health complications and increased healthcare costs. Early detection is crucial for effective intervention and prevention. This project addresses this issue by using machine learning to improve the accuracy and reliability of osteoporosis risk prediction.

## Project Objectives
- Develop a robust predictive algorithm to identify individuals at high risk for osteoporosis.
- Complement efforts by physicians in early diagnosis of osteoporosis and its prevention.
- Strengthen existing screening protocols.
- Contribute to better patient management and reduced healthcare burden.

## Dataset
**Source:** [Osteoporosis Dataset on Kaggle](https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis?select=osteoporosis.csv)

**Description:**
- Number of entries: 1958
- Data columns: 16 (Age, Gender, Hormonal Changes, Family History, Race/Ethnicity, Body Weight, Calcium Intake, Vitamin D Intake, Physical Activity, Smoking, Alcohol Consumption, Medical Conditions, Medications, Prior Fractures, Osteoporosis)

## Tools and Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Plotly
- Bootstrap
- Spark

## Project Workflow

### 1. Data Acquisition and Exploratory analysis
- Load the dataset.
- Handle any missing values.
- Encode categorical variables.
- Normalize or standardize the data.
- Conduct exploratory data analysis to understand the dataset's structure and key features.
- Visualize data distributions and identify correlations between variables.

### 2. Model Development
- Select and implement various machine learning algorithms (logistic regression, random forests).
- Split the data into training and validation sets.

### 3. Model Evaluation
- Evaluate the models using performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
- Compare model results to identify the best-performing algorithm.

### 4. Model Optimization
- Fine-tune the selected model using hyperparameter tuning techniques to enhance predictive accuracy.
- Implement cross-validation to ensure model generalizability and robustness.

### 5. Documentation
- Document the entire project workflow, including data preprocessing, EDA, model development, evaluation, and optimization.
- Compile a comprehensive report detailing methodologies, findings, and insights gained throughout the project.

### 7. Presentation Preparation
- Create a detailed presentation to showcase the project, emphasizing the problem statement, methodology, results, and impact.
- Prepare to present the findings to stakeholders, highlighting the potential of the predictive model to improve early osteoporosis diagnosis and patient outcomes.

## Results
- Developed a machine learning model with high predictive accuracy for osteoporosis risk.
- Facilitated early diagnosis and preventive care through improved screening processes.
- Enhanced patient management and reduced healthcare burden.

- **Model Evaluation** 

The performance of two different machine learning models; logistic regression, and random forest classifier, were evaluated using various metrics. These evaluation metrics helped assess how well the models would perform in terms of classification accuracy, precision, recall, F1 score, and their ability to discriminate between classes. The results showed that Random Forest Classifier model generally outperforms the Logistic Regression model in terms of accuracy, precision, recall, F1 score, and AUC-ROC score based on the provided results. 

![Screenshot 2024-08-08 195249](https://github.com/user-attachments/assets/20da59b5-4af4-4164-9e4b-afe3f7e3c60b)



**Model Optimization** 

The first optimization utilizes GridSearchCV to fine-tune hyperparameters for the Random Forest Classifier model and improve its performance. The best parameters found were max_depth: 10, min_samples_leaf: 4, min_samples_split: 10, n_estimators: 100. 

![Screenshot 2024-08-08 201307](https://github.com/user-attachments/assets/3366348d-0f95-4cf3-8af9-fc28648f707f)

The second optimization utilizes RandomizedSearchCV to randomly sample a fixed number of hyperparameter combinations, making it more efficient. 

![Screenshot 2024-08-08 201336](https://github.com/user-attachments/assets/8be32dde-62ec-4042-8c04-e7542f0cfd5e)

The third optimization also utilizes RandomizedSearchCV but with a different function (randit, uniform) which allows for exploring a wider range of values for min_samples_leaf compared to using only integers. 

![Screenshot 2024-08-08 201351](https://github.com/user-attachments/assets/edcf9a86-05b4-4656-85c3-85b61b7d6016)

The fourth optimization again utilizes RandomizedSearchCV with the randint function. 

![Screenshot 2024-08-08 201404](https://github.com/user-attachments/assets/99ffcc18-e359-4e26-a955-82f648ed47e0)

The hyperparameter tuning with RandomizedSearchCV introduces a new hyperparameter (bootstrap) and explores a wider range for some existing ones. 

![Screenshot 2024-08-08 201415](https://github.com/user-attachments/assets/2a8e4ce1-afec-461b-a2fe-35f871388b56)


## Conclusion
The evaluation and optimization of the machine learning models demonstrated that the Random Forest Classifier significantly(85.4%) outperforms the Logistic Regression model(82.0%) across all key metrics, including accuracy, precision, recall, F1 score, and AUC-ROC score. These results highlight the Random Forest model's superior ability to classify and discriminate between classes in the given dataset.

Through hyperparameter tuning using GridSearchCV and RandomizedSearchCV, the performance of the Random Forest Classifier was further enhanced. The fine-tuning process identified optimal parameter settings, such as a max_depth of 10, min_samples_leaf of 4, min_samples_split of 10, and n_estimators of 100, which collectively improved the model's predictive power.

Overall, the thorough optimization process has resulted in a highly robust model, well-suited for improving the prediction of osteoporosis risk in asymptomatic patients. The strategic use of RandomizedSearchCV allowed for an efficient exploration of the hyperparameter space, ultimately leading to a model that balances performance and computational efficiency. The introduction of additional hyperparameters, such as bootstrap, provided further opportunities to refine the model, ensuring that it is finely tuned to deliver reliable and accurate predictions in practical applications.

To further improve the model optimization process, several additional techniques could be explored:


- K-Fold Cross-Validation:
  
Instead of relying on a single train-test split, K-Fold Cross-Validation divides the dataset into k subsets, trains the model on k-1 subsets, and tests it on the remaining subset. This process is repeated k times, with each subset used as the test set once. This technique provides a more robust evaluation by reducing the variance associated with the random sampling of training and test data.
Ensemble Methods

- Stacking:
  
Combine the predictions of multiple models (e.g., Random Forest, Gradient Boosting, and Logistic Regression) using another model (often a simple one like Logistic Regression) to make the final prediction. This method can improve predictive performance by leveraging the strengths of multiple models.
Blending: A variation of stacking, where the predictions of different models are combined using a weighted average or voting mechanism. This approach can be simpler and less prone to overfitting than full stacking.
Advanced Hyperparameter Tuning

- Hyperband:
  
A more efficient hyperparameter optimization method that uses adaptive resource allocation and early stopping to identify promising configurations quickly. It can significantly reduce computation time compared to traditional methods. 
 
- Feature Selection:

 Identify and retain only the most relevant features, possibly through techniques like Recursive Feature Elimination (RFE) or Lasso Regularization. Reducing the dimensionality of the data can enhance model performance and reduce overfitting.

- Feature Scaling and Normalization:

 Ensure that all features are on a similar scale to prevent the model from being biased toward features with larger magnitudes. Techniques like StandardScaler or MinMaxScaler can be employed.

- Polynomial Features: 

Generate interaction terms or polynomial terms of features to capture non-linear relationships within the data that the model might miss with the original features.
Algorithmic Alternatives

- Gradient Boosting Machines (GBM):
  
Models like XGBoost, LightGBM, or CatBoost, which are often more powerful than Random Forest, could be explored. These gradient boosting techniques build trees sequentially and correct errors made by the previous trees, often resulting in better performance on complex datasets.


- Synthetic Data Generation:
  
If the dataset is imbalanced, techniques like SMOTE (Synthetic Minority Over-sampling Technique) can be used to generate synthetic examples for the minority class, which can improve model performance.

By incorporating these techniques, we could potentially enhance the model's predictive accuracy, generalization ability, and computational efficiency, leading to even better performance in real-world applications.


## Contributors
- Andrea Nimako
- Lynda Sempele
- Kevin Ngala

## Acknowledgments
- George Washington University Data Analysis Bootcamp for providing the foundation and resources for this project.
- Kaggle for the dataset used in this project.
- Instructors and mentors for their guidance and support.

---

This comprehensive README file provides all the necessary information about the project, from problem statement and objectives to detailed workflow and installation instructions. It ensures that anyone interested in the project can understand its scope, methodology, and outcomes.

## Disclaimer:
This project report is for educational purposes only. The results and conclusions are based on the provided dataset and the methodologies applied during the course.
