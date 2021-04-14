import streamlit as st
import pandas as pd
# import shap ### Warning: I'm having issues with instalation of sharp
import base64
import os
import joblib
import math
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

st.write("""
# Real Estate Price Forecast - Competition - App

This app is made with one of the competitors code inside DataSource.ai. These are the results
""")
st.write('---')

# Loads Datasets

## pickles definitions - directory inside the environment
my_dir = os.path.dirname(__file__)
xgb_file_path = os.path.join(my_dir, 'pickles/real_estate_price_forecast/xgb.pkl')
y_pred_path = os.path.join(my_dir, 'datasets/real_estate_price_forecast/y_pred.csv')
y_true_path = os.path.join(my_dir, 'datasets/real_estate_price_forecast/y_true.csv')
X_train_path= os.path.join(my_dir, 'datasets/real_estate_price_forecast/X_train.csv')
X = pd.read_csv(X_train_path)


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    pais = st.sidebar.selectbox('Country',('Argentina','Colombia'))
    rooms = st.sidebar.slider('rooms', float(X.rooms.min()), float(X.rooms.max()), float(X.rooms.mean()))
    bedrooms = st.sidebar.slider('bedrooms', float(X.bedrooms.min()), float(X.bedrooms.max()), float(X.bedrooms.mean()))
    bathrooms = st.sidebar.slider('bathrooms', float(X.bathrooms.min()), float(X.bathrooms.max()), float(X.bathrooms.mean()))
    surface_total = st.sidebar.slider('surface_total', float(X.surface_total.min()), float(X.surface_total.max()), float(X.surface_total.mean()))

    data = {'pais': pais,
            'rooms': rooms,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'surface_total': surface_total}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Build Regression Model
model = joblib.load(xgb_file_path)
y_pred_df = pd.read_csv(y_pred_path)
y_true_df = pd.read_csv(y_true_path)

y_true_id = y_true_df["id"]
y_true = y_true_df["price"]
y_pred_id = y_pred_df["id"]
y_pred = y_pred_df["price"]

st.header('Score Results')

## score
score = math.sqrt(mean_squared_log_error(y_true, y_pred))
st.write(score)
st.write('---')

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

## Feature Engineering

# I have to do this because train and test datasets finalice with a different number
# of columns after doing dummy encopding with datasets separated
# im following this technicque: https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f

class GetDummies:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    # Dummy encoding
    def dummy_encoding(self, column_name):
        self.train['train'] = 1
        self.test['train'] = 0

        combined = pd.concat([self.train, self.test])
        column_encoded = pd.get_dummies(combined[column_name])
        column_encoded.columns = ['{}_{}'.format(column_name, x) for x in column_encoded.columns]

        combined = pd.concat([combined, column_encoded], axis=1)

        train = combined[combined['train'] == 1]
        test = combined[combined['train'] == 0]

        train.drop(['train'], axis=1, inplace=True)
        test.drop(['train'], axis=1, inplace=True)

        return train, test

    # Drop columns
    def drop_columns(self, df, columns):
        new_df = df.drop(columns, axis=1)
        return new_df

columns = ['pais', 'rooms', 'bedrooms', 'bathrooms','surface_total']
one_entry = pd.DataFrame(df, columns=columns)
get_dummies_one_entry_instance = GetDummies(X, one_entry)
train_one_entry_encoded, test_one_entry_encoded = get_dummies_one_entry_instance.dummy_encoding('pais')
new_one_entry = get_dummies_one_entry_instance.drop_columns(test_one_entry_encoded, ['pais', 'price'])

# Apply Model to Make Prediction
prediction = model.predict(new_one_entry)

st.header('Prediction of Price')
st.write(prediction)
st.write('---')

### Warning: I'm having issues with instalation of sharp

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')
#
# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
