# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/mnt/data/mteb-retrieval-data.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows to understand its structure
# data.head()

# Initial model using multiple independent variables
# Selecting independent and dependent variables
X = data[['ArguAna', 'FiQA2018', 'SCIDOCS', 'SciFact', 'NFCorpus']]
y = data['Avg']

# Splitting the data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)

# Predicting the Avg for the validation set
y_pred = model.predict(X_validation)

# Calculating and printing model performance metrics
mse = mean_squared_error(y_validation, y_pred)
r2 = r2_score(y_validation, y_pred)
# print(mse, r2)

# Creating a DataFrame to display validation results including true and predicted ranks
validation_results = pd.DataFrame({
    'True Avg': y_validation,
    'Predicted Avg': y_pred
})
validation_results['True Rank'] = validation_results['True Avg'].rank(ascending=False)
validation_results['Predicted Rank'] = validation_results['Predicted Avg'].rank(ascending=False)
validation_results = validation_results.sort_values('True Rank')
# print(validation_results)

# Model using only NFCorpus as the independent variable
X_nfc = data[['NFCorpus']]
y_nfc = data['Avg']

# Splitting the data for NFCorpus
X_train_nfc, X_validation_nfc, y_train_nfc, y_validation_nfc = train_test_split(X_nfc, y_nfc, test_size=0.2, random_state=42)

# Initializing and fitting the model
model_nfc = LinearRegression()
model_nfc.fit(X_train_nfc, y_train_nfc)

# Predicting and evaluating for NFCorpus
y_pred_nfc = model_nfc.predict(X_validation_nfc)
mse_nfc = mean_squared_error(y_validation_nfc, y_pred_nfc)
r2_nfc = r2_score(y_validation_nfc, y_pred_nfc)

# Displaying validation results for NFCorpus model
validation_results_nfc = pd.DataFrame({
    'True Avg': y_validation_nfc,
    'Predicted Avg': y_pred_nfc
})
validation_results_nfc['True Rank'] = validation_results_nfc['True Avg'].rank(ascending=False)
validation_results_nfc['Predicted Rank'] = validation_results_nfc['Predicted Avg'].rank(ascending=False)
validation_results_nfc = validation_results_nfc.sort_values('True Rank')
# print(mse_nfc, r2_nfc, validation_results_nfc)
