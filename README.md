# Wheat Guardian 🌾

**Wheat Guardian** is an integrated agricultural management system designed to predict wheat diseases, estimate crop yield, recommend fertilizers, and provide weather-based insights. The system uses a combination of machine learning, deep learning, and real-time data to assist farmers in making informed decisions, contributing to sustainable agriculture and aligning with global Sustainable Development Goals (SDGs).

## Features

### 1. Wheat Disease Detection
- **Model**: Convolutional Neural Network (CNN) using PyTorch.
- **Input**: Leaf images of wheat plants.
- **Output**: Detects diseases such as wheat rust and other leaf-related issues.

### 2. Yield Prediction
- **Model**: K-Nearest Neighbors (KNN) algorithm.
- **Input**: State, district, and month.
- **Output**: Predicts wheat yield based on environmental and location data.

### 3. Fertilizer Recommendation
- **Model**: Linear Regression model using scikit-learn.
- **Categories**: Recommends fertilizers based on inputs, including:
  - 10-26-26
  - 14-35-14
  - 17-17-17
  - 20-20
  - 28-28
  - DAP
  - Urea
- **Output**: Provides the optimal fertilizer type and quantity based on crop and environmental data.

### 4. Weather Insights
- **API**: Integrated with the Google Weather API.
- **Functionality**: Provides real-time weather updates, including temperature, humidity, and rain forecasts, to help farmers plan irrigation and other field activities.

## Project Workflow
- **Frontend**: Built using Flask/Django with Bootstrap templates for a user-friendly interface.
- **Backend**: Developed in Python, leveraging PyTorch for disease detection and scikit-learn for prediction models.
- **Search Functionality**: Allows users to upload or search for wheat leaf images, which are then analyzed for disease detection.
- **Weather Integration**: Real-time weather insights powered by Google API for accurate field recommendations.

## Technologies Used
- **Python**: Core language for building the application.
- **PyTorch**: For training and deploying the CNN model for disease detection.
- **scikit-learn**: For implementing KNN (yield prediction) and Linear Regression (fertilizer recommendation).
- **Flask/Django**: Web framework for building the application interface.
- **Google Weather API**: Integrated to provide real-time weather insights.

## Sustainable Development Goals (SDGs)
This project supports the following SDGs:
- **SDG 2: Zero Hunger**
- **SDG 12: Responsible Consumption and Production**
- **SDG 13: Climate Action**
- **SDG 15: Life on Land**

## Install Dependencies:
 - pip install -r requirements.txt

## Run the Application:
 -python app.py
