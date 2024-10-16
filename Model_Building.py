import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT_DIR))
import pandas as pd
from Overview import Overview
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


#Read the data
dataloader = Overview()
df_users, df_loans, df_emotional_data = dataloader.read_data()


### merge emotional_data with users data for model building
df_merged = pd.merge(df_emotional_data, df_users[["interest_rate", "loan_term", "user_id", "credit_limit"]], on="user_id",
                                   how="left")

print(df_merged.head())

###print dtypes
print(df_merged.dtypes)

##drop columns that are not needed in the model building
df_merged= df_merged.drop(columns = ["timestamp", "user_id"])


#get all the categorical columns in the dataframe
categorical_columns = df_merged.select_dtypes(include=['object']).columns

#loop through each and encode them
for column in categorical_columns:
    df_merged[column] = pd.Categorical(df_merged[column]).codes

X= df_merged[['intensity', 'relationship',
       'physical_state', 'preceding_event', 'situation',
        'location', 'weather', 'time_of_day',
       'grade']]

y= df_merged[['credit_limit']]

##create model for Light GBM
model= LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#print the error
mae = mean_absolute_error(y_pred, y_test)
print(mae)

# Get feature importance
importance = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

