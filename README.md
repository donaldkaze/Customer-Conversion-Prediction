```markdown
# Customer Conversion Prediction with XGBoost on AWS SageMaker

## Project Overview
This project develops an AI-driven system to predict customer conversion, integrating Salesforce CRM data with AWS SageMaker and Lambda for scoring and automation. The goal is to identify high-potential leads early, enabling targeted interventions to improve conversion rates and sales efficiency.

## Problem Statement
Lead conversion in sales funnels is a critical business metric, but often suffers from low efficiency due to the difficulty of identifying truly interested leads from a large pool. Our objective is to build a machine learning model that accurately predicts which leads are most likely to convert, allowing sales teams to prioritize their efforts and optimize resource allocation.

## Dataset Information
The project utilizes a dataset named `customer_conversion_testing_dataset.csv`, which contains various attributes of leads, including demographic information, engagement metrics, and historical interaction data. The target variable is `Conversion (Target)`, a binary flag indicating whether a lead converted (1) or not (0).

**Key Characteristics:**
*   **Highly Imbalanced:** The dataset exhibits significant class imbalance, with approximately 98.42% non-conversions (0) and only 1.58% conversions (1). This challenge was a central focus of the preprocessing and modeling strategy.

## Methodology and Approach

### 1. Data Loading & Initial Exploration
The dataset was loaded into a Pandas DataFrame. Initial exploration involved viewing the first few rows (`df.head()`) and checking data types and non-null counts (`df.info()`). Missing values were found to be zero for all columns.

### 2. Data Preprocessing
To prepare the data for machine learning, the following steps were performed:

*   **Identification of Features:** Categorical and numerical features were identified.
*   **One-Hot Encoding:** Categorical features (e.g., `Gender`, `Location`, `LeadSource`, `DeviceType`, `ReferralSource`, `PaymentHistory`) were converted into numerical representations using One-Hot Encoding (`pd.get_dummies`).
*   **Feature Scaling:** Numerical features were standardized using `StandardScaler` to ensure no single feature dominated the learning process due to its scale.
*   **Train-Test Split:** The dataset was split into 80% for training and 20% for testing, with `stratify=y` to maintain the original class distribution in both sets.
*   **Class Imbalance Handling:** The `scale_pos_weight` was calculated based on the ratio of negative to positive samples in the training set to be used in the XGBoost model, giving more importance to the minority class during training.
*   **Addressing Data Leakage:** Through iterative analysis, two critical sources of data leakage were identified and removed:
    *   `LeadID`: A unique identifier that could allow the model to memorize outcomes.
    *   `LeadStatus` (and its encoded columns like `LeadStatus_Hot`, `LeadStatus_Warm`): This feature was found to be highly correlated with the target, potentially encoding the conversion event itself, leading to unrealistically perfect model performance. By removing it, the model's predictions became more robust and generalizable.

### 3. Model Selection & Training
An **XGBoost Classifier** was chosen due to its strong performance on tabular data and its native support for handling class imbalance through the `scale_pos_weight` parameter.

*   **Training:** The model was trained on the `X_train` and `y_train` datasets.

### 4. Model Evaluation
The trained model was evaluated on the `X_test` dataset using a suite of metrics suitable for imbalanced classification:

*   **Accuracy:** 0.9904
*   **Precision:** 0.6813
*   **Recall:** 0.7470
*   **F1-Score:** 0.7126
*   **ROC AUC Score:** 0.9954

**Confusion Matrix:**

|                   | Predicted Not Converted | Predicted Converted |
| :---------------- | :---------------------- | :------------------ |
| **Actual Not Converted** | 5117                    | 29                  |
| **Actual Converted**     | 21                      | 62                  |

**Interpretation:** The model demonstrates excellent performance, particularly evidenced by a high ROC AUC score, indicating strong discriminatory power. The F1-Score suggests a good balance between precision (correct positive predictions) and recall (identifying all actual positives), which is crucial for imbalanced datasets.

### 5. Feature Importance Analysis
An analysis of feature importance revealed the top drivers of conversion:

1.  `Location_Sialkot`
2.  `Location_Peshawar`
3.  `PagesViewed`
4.  `Location_Multan`
5.  `Location_Quetta`
6.  `Location_Rawalpindi`
7.  `Location_Gujranwala`
8.  `Location_Islamabad`
9.  `Location_Karachi`
10. `Age`

**Key Insight:** Geographical location plays a dominant role in predicting lead conversion, alongside `PagesViewed`, highlighting the importance of regional market dynamics and user engagement.

## Technical Stack
*   **Programming Language:** Python
*   **Libraries:** Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
*   **Cloud Platform:** AWS (SageMaker, S3, Lambda)

## Deployment to AWS SageMaker
The trained XGBoost model is prepared for deployment as a real-time inference endpoint on AWS SageMaker.

1.  **Model Saving:** The trained model is saved using `joblib` into a file named `xgboost-model.joblib` within a dedicated `model` directory.
2.  **Inference Script (`inference.py`):** A Python script (`inference.py`) is created, containing the `model_fn`, `input_fn`, `predict_fn`, and `output_fn` functions. These functions define how SageMaker loads the model, preprocesses incoming data, makes predictions, and formats the output.
3.  **Packaging Model Artifacts:** The `xgboost-model.joblib` and `inference.py` files are bundled into a `model.tar.gz` archive.
4.  **S3 Upload:** The `model.tar.gz` archive is uploaded to an Amazon S3 bucket.
5.  **SageMaker Endpoint Deployment:** The SageMaker Python SDK is used to create an `XGBoostModel` object pointing to the S3 location of the `model.tar.gz`. This model is then deployed to a SageMaker endpoint (e.g., `my-xgboost-conversion-predictor`) on a specified instance type (e.g., `ml.m5.xlarge`).
6.  **Integration (Optional):** AWS Lambda functions can be used to trigger the SageMaker endpoint for new leads (e.g., from Salesforce) and process the predictions for automated workflows.

## Future Enhancements
*   **Hyperparameter Tuning:** Conduct a more extensive search for optimal XGBoost hyperparameters.
*   **Cross-Validation:** Implement k-fold cross-validation for a more robust evaluation of model performance.
*   **Threshold Optimization:** Adjust the prediction probability threshold based on specific business costs of false positives vs. false negatives.
*   **Advanced Interpretability:** Utilize tools like SHAP or LIME to explain individual predictions.
*   **Model Monitoring:** Implement data drift and concept drift monitoring for the deployed model in production.
```
