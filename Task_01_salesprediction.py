# sales_prediction.py
"""
Sales Prediction Project for Internship Portfolio
Author: [Your Name]
Date: [Current Date]

Description:
This project predicts future sales based on advertising spend across different platforms (TV, Radio, Social Media).
It includes data cleaning, exploratory analysis, feature engineering, model training, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set styling for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load and explore dataset
def load_data():
    # Generate synthetic data if you don't have a real dataset
    np.random.seed(42)
    data = {
        'TV_Spend': np.random.uniform(1000, 5000, 200),
        'Radio_Spend': np.random.uniform(500, 2500, 200),
        'Social_Media_Spend': np.random.uniform(200, 1500, 200),
        'Target_Segment': np.random.choice(['Young', 'Adult', 'Senior'], 200),
        'Sales': np.random.uniform(5000, 30000, 200)
    }
    return pd.DataFrame(data)

# Data cleaning and preprocessing
def preprocess_data(df):
    # Handle missing values
    if df.isnull().sum().any():
        df = df.dropna()
    
    # Feature engineering
    df['Total_Spend'] = df['TV_Spend'] + df['Radio_Spend'] + df['Social_Media_Spend']
    df['Spend_Ratio_TV'] = df['TV_Spend'] / df['Total_Spend']
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Target_Segment'], drop_first=True)
    
    return df

# Exploratory Data Analysis
def perform_eda(df):
    print("\n📊 Exploratory Data Analysis:")
    print(df.describe())
    
    # Visualization 1: Spending Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['TV_Spend'], ax=axes[0], kde=True, color='skyblue')
    axes[0].set_title('TV Advertising Spend Distribution')
    
    sns.histplot(df['Radio_Spend'], ax=axes[1], kde=True, color='orange')
    axes[1].set_title('Radio Advertising Spend Distribution')
    
    sns.histplot(df['Social_Media_Spend'], ax=axes[2], kde=True, color='green')
    axes[2].set_title('Social Media Spend Distribution')
    plt.savefig('ad_spend_distribution.png')
    plt.close()
    
    # Visualization 2: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Visualization 3: Spend vs. Sales
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='TV_Spend', y='Sales', data=df, color='skyblue')
    plt.title('TV Spend vs Sales')
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='Radio_Spend', y='Sales', data=df, color='orange')
    plt.title('Radio Spend vs Sales')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='Social_Media_Spend', y='Sales', data=df, color='green')
    plt.title('Social Media Spend vs Sales')
    plt.savefig('spend_vs_sales.png')
    plt.close()

# Model Training and Evaluation
def train_and_evaluate(df):
    # Prepare features and target
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    # Evaluate models
    def evaluate_model(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{name} Performance:")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"R-squared: {r2:.2f}")
        return mse, r2
    
    lr_mse, lr_r2 = evaluate_model("Linear Regression", y_test, lr_pred)
    rf_mse, rf_r2 = evaluate_model("Random Forest", y_test, rf_pred)
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance (Random Forest)')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return lr, rf, scaler

# What-If Analysis Function
def what_if_analysis(model, scaler, tv_spend, radio_spend, social_spend, segment):
    # Create input DataFrame with all expected columns
    input_data = {
        'TV_Spend': [tv_spend],
        'Radio_Spend': [radio_spend],
        'Social_Media_Spend': [social_spend],
        'Total_Spend': [tv_spend + radio_spend + social_spend],
        'Spend_Ratio_TV': [tv_spend / (tv_spend + radio_spend + social_spend)],
        'Target_Segment_Adult': [1 if segment == 'Adult' else 0],
        'Target_Segment_Senior': [1 if segment == 'Senior' else 0],
        'Target_Segment_Young': [1 if segment == 'Young' else 0]  # Add this line
    }

    input_df = pd.DataFrame(input_data)

    # Reorder columns to match training data
    expected_columns = scaler.feature_names_in_
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    # Predict sales
    prediction = model.predict(scaled_input)[0]
    print(f"\n💰 Predicted Sales: ${prediction:,.2f}")

    return prediction

# Main execution
def main():
    print("🚀 Starting Sales Prediction Project")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # EDA
    perform_eda(df)
    
    # Train models
    lr_model, rf_model, scaler = train_and_evaluate(df)
    
    # Example what-if analysis
    print("\n🔍 Example What-If Analysis:")
    what_if_analysis(rf_model, scaler, tv_spend=3500, radio_spend=1200, social_spend=800, segment='Adult')
    
    print("\n✅ Project execution completed successfully!")

if __name__ == "__main__":
    main()
# Auto detect text files and perform LF normalization
