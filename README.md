# Heart Attack Prediction: Classification Algorithms Comparison
## Project Overview
This project aims to predict the likelihood of heart attacks based on various medical parameters using a range of classification algorithms. The primary goal is to evaluate and compare the performance of multiple machine learning models to identify the best classifier for the task.

## Dataset
The dataset used for this project is the Heart Attack Analysis & Prediction Dataset, which includes various medical parameters of patients to predict the likelihood of heart attacks.

### *Source:* 
Dataset provided by the instructor.
### *Features:*
age: Age of the patient
sex: Gender (1 = male, 0 = female)
cp: Chest pain type
trtbps: Resting blood pressure
chol: Serum cholesterol
fbs: Fasting blood sugar
restecg: Resting electrocardiographic results
thalachh: Maximum heart rate achieved
exng: Exercise induced angina
oldpeak: Depression induced by exercise relative to rest
slp: Slope of the peak exercise ST segment
caa: Number of major vessels colored by fluoroscopy
thall: Thalassemia
output: Heart attack (1 = yes, 0 = no) – Target Variable
Size: The dataset consists of 303 rows and 14 columns .

## Algorithms Used
Several machine learning algorithms were tested to classify the dataset:

K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
AdaBoost
Random Forest
Decision Tree
Logistic Regression

These algorithms were evaluated after performing hyperparameter tuning using GridSearch and cross-validation to enhance accuracy and robustness.

## Data Preprocessing
To ensure the models perform optimally, the following steps were applied:

1.Data Balancing: Used RandomUndersampler and SMOTE to balance the target variable and mitigate class imbalance.

2.GridSearch with Cross-Validation: Hyperparameters for each model were optimized using GridSearch, which helps identify the best parameter combinations and ensures the models generalize well.

## Model Evaluation
Each model was evaluated using the following metrics:

-Accuracy: Overall prediction accuracy.

-Precision, Recall, F1-score: Metrics to evaluate model performance in detail.

-Cross-validation: Ensures that the model generalizes well to unseen data and reduces the risk of overfitting.

## Comparative Study
After evaluating the models, the following performance was observed:

Best Performers: Logistic Regression and KNN, with an accuracy of 87%.

Moderate Performers: SVM and AdaBoost, both achieving 84% accuracy.

Lower Performers: Decision Tree and Random Forest showed more modest performance compared to the other algorithms.

## Final Choice
The final model choice will be based on a combination of its accuracy and its ability to generalize effectively to unseen data.

Logistic Regression and KNN models have shown the highest accuracy, making them suitable choices, but care must be taken to avoid overfitting and ensure robust performance across both training and test datasets.

## Installation and Setup
To run this project, follow the instructions below:

1.Clone the repository
git clone https://github.com/Maryem-Jlassi/classification.git

2.Navigate to the project directory
cd classification

3.Install required libraries
pip install -r requirements.txt

## Usage
The project is implemented in the classif_homework.ipynb Jupyter notebook. To run the notebook, open it in Jupyter Notebook or JupyterLab and execute the cells in order.

1.Load the dataset.
2.Preprocess the data, balance the target variable.
3.Apply various classification algorithms.
4.Perform hyperparameter tuning using GridSearch.
5.Evaluate the models and compare their performance.
6.Choose the best model based on the comparative study.

## Results and Conclusion
The comparative study of classification algorithms revealed that Logistic Regression and KNN performed the best, with both achieving an accuracy of 87%. SVM and AdaBoost followed with 84% accuracy, while Decision Tree and Random Forest performed less effectively.

The choice of model will depend not only on accuracy but also on the ability to generalize and perform well with unseen data. Future work could include more hyperparameter tuning and testing with different models for further improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
