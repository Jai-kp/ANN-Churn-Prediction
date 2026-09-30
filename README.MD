# Customer Churn Prediction with Artificial Neural Networks (ANN)

## Overview
This project utilizes Artificial Neural Networks (ANN) with TensorFlow/Keras to predict customer churn (classification) and perform regression using the `Churn_Modelling.csv` dataset. It covers all essential steps: data preprocessing, model training, hyperparameter tuning, and deployment as a web application using Streamlit.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Building (Classification & Regression)](#model-building-classification--regression)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)

---

## Dataset
The project uses the `Churn_Modelling.csv` file containing bank customer data. Key features used for prediction include:
- **CreditScore**
- **Geography**
- **Gender**
- **Age**
- **Tenure**
- **Balance**
- **NumOfProducts**
- **HasCrCard**
- **IsActiveMember**
- **EstimatedSalary**
- **Exited** (Target for classification: 1 if the customer churned)

---

## Data Preprocessing
Preprocessing steps, as performed and repeated across the notebooks:
1. **Remove Unnecessary Columns**: Dropping `RowNumber`, `CustomerId`, and `Surname`.
2. **Categorical Encoding**:
   - Gender: Label encoding (`Male`/`Female` → 0/1).
   - Geography: One-hot encoding (creates new binary columns for each location).
3. **Combining Encoded Features**: Concatenation of one-hot encoded columns with the dataset, dropping originals.
4. **Splitting Data**: Dividing into independent (`X`) and dependent (`y`) features. For classification, `Exited` is the target; for regression, `EstimatedSalary` is sometimes used (see corresponding notebook).
5. **Train/Test Split**: Splitting the data (commonly 80% train, 20% test).
6. **Feature Scaling**: Applying a `StandardScaler` to normalize numeric values.
7. **Saving Encoders & Scalers**: For reproducibility and deployment, encoders and scalers are serialized with Pickle.

---

## Model Building (Classification & Regression)
### Artificial Neural Network (ANN) - Classification
- **Architecture**: Two hidden layers (e.g., 64 and 32 neurons, ReLU activation), Output layer (1 neuron, sigmoid for binary classification).
- **Compilation**: Adam optimizer, binary crossentropy loss, accuracy as a metric.
- **Callbacks**: Early stopping and TensorBoard for monitoring.
- **Training**: Up to 100 epochs, validation on test set.
- **Saving**: Model is saved as `model.h5`.

### ANN - Regression
- Follows similar preprocessing, but predicts a continuous target (e.g., `EstimatedSalary`). Output layer is a single neuron with linear activation (default).
- Loss function: Mean Absolute Error (`mae`).
- Model is saved as `regression_model.h5`.


---

## Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV` and Keras wrappers (`KerasClassifier`). Parameters like the number of neurons, number of layers, and epochs are grid-searched for optimal accuracy.

---

## Model Evaluation
- **Classification**: Main metric is accuracy. TensorBoard and history plots help visualize training and validation accuracy/loss over epochs.
- **Regression**: Main metric is MAE (Mean Absolute Error).
- Early stopping ensures the best model is restored if validation loss doesn't improve.

---

## Usage
### Running the Streamlit App
The `app.py` file provides an interactive web interface (via Streamlit) for customer churn prediction. Users can enter feature values, and the model predicts churn probability.

#### Steps to Run the App:
1. **Install dependencies** if not already done:
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
3. **Use the interface** to input features and obtain predictions with detailed probability and outcome ("likely to churn" vs. "not likely to churn").

---

## Dependencies
Key packages:
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Pickle
- Matplotlib (for notebook experiments)

All dependencies are listed in `requirements.txt`.

---

## Project Structure
```
Customer Churn Prediction/
│
├── app.py                      # Streamlit web application
├── churnregression.ipynb       # Regression notebook
├── experiments.ipynb           # General ANN experiments
├── hyperparametertuningann.ipynb # Hyperparameter tuning notebook
├── practise.ipynb              # ANN practice notebook
├── predictions.ipynb           # Inference/prediction (script/notebook)
├── model.h5                    # Trained ANN for churn (classification)
├── regression_model.h5         # Trained ANN for regression
├── scaler.pkl                  # Saved StandardScaler
├── label_encoder_gender.pkl    # Saved LabelEncoder
├── onehot_encoder_geo.pkl      # Saved OneHotEncoder (Geography)
├── onehot_encoder_gender_geo.pkl # Another saved encoder (for hyperparam tuning)
├── Churn_Modelling.csv         # Source dataset
├── requirements.txt            # Python package dependencies
├── README.md                   # (This file)
└── ...
```

---

## Notes
- TensorBoard logs are written to the `logs/`, `regressionlogs/`, and `prlogs/` folders for training visualization.
- Notebooks and code are well-commented for clarity; follow cell order for reproducibility.
- Model/encoder filenames may differ in hyperparameter tuning or regression cases.

---
