import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
import streamlit as st

# Function to load and preprocess the data
@st.cache
def load_data(filepath):
    car_dataset = pd.read_csv(filepath)
    
    # Encoding categorical data
    car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
    
    return car_dataset

# Function to split the data
def split_data(car_dataset):
    X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
    Y = car_dataset['Selling_Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    return X_train, X_test, Y_train, Y_test

# Function to train and evaluate model
def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    
    # Prediction on Training data
    training_data_prediction = model.predict(X_train)
    
    # R squared Error for training data
    train_error_score = metrics.r2_score(Y_train, training_data_prediction)
    
    # Prediction on Test data
    test_data_prediction = model.predict(X_test)
    
    # R squared Error for test data
    test_error_score = metrics.r2_score(Y_test, test_data_prediction)
    
    return train_error_score, test_error_score, training_data_prediction, test_data_prediction

# Main function for Streamlit app
def main():
    st.title("Car Price Prediction")
    
    filepath = st.text_input("Enter the path to your CSV file:", 'C:/Users/user/Documents/File-file Kuliah/Tugas Semester 4 (File Jadi)/Kecerdasan Buatan/cars_data.csv')
    
    if filepath:
        car_dataset = load_data(filepath)
        
        # Display basic info about the dataset
        st.header("Dataset Information")
        st.write(car_dataset.head())
        st.write("Shape of the dataset:", car_dataset.shape)
        st.write("Missing values:", car_dataset.isnull().sum())
        
        # Display distribution of categorical data
        st.header("Categorical Data Distribution")
        st.write("Fuel Type Distribution:", car_dataset.Fuel_Type.value_counts())
        st.write("Seller Type Distribution:", car_dataset.Seller_Type.value_counts())
        st.write("Transmission Type Distribution:", car_dataset.Transmission.value_counts())
        
        # Split the data
        X_train, X_test, Y_train, Y_test = split_data(car_dataset)
        
        # Train and evaluate Linear Regression model
        lin_reg_model = LinearRegression()
        lin_train_error, lin_test_error, lin_train_pred, lin_test_pred = train_and_evaluate_model(lin_reg_model, X_train, Y_train, X_test, Y_test)
        
        st.header("Linear Regression Model")
        st.write("Training R squared Error:", lin_train_error)
        st.write("Test R squared Error:", lin_test_error)
        
        # Plotting results for Linear Regression
        fig, ax = plt.subplots()
        ax.scatter(Y_train, lin_train_pred)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual Prices vs Predicted Prices (Training Data)")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ax.scatter(Y_test, lin_test_pred)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual Prices vs Predicted Prices (Test Data)")
        st.pyplot(fig)
        
        # Train and evaluate Lasso Regression model
        lass_reg_model = Lasso()
        lass_train_error, lass_test_error, lass_train_pred, lass_test_pred = train_and_evaluate_model(lass_reg_model, X_train, Y_train, X_test, Y_test)
        
        st.header("Lasso Regression Model")
        st.write("Training R squared Error:", lass_train_error)
        st.write("Test R squared Error:", lass_test_error)
        
        # Plotting results for Lasso Regression
        fig, ax = plt.subplots()
        ax.scatter(Y_train, lass_train_pred)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual Prices vs Predicted Prices (Training Data)")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ax.scatter(Y_test, lass_test_pred)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual Prices vs Predicted Prices (Test Data)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
