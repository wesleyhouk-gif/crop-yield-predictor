🌾 Crop Yield Prediction App
Overview

This project is an end-to-end machine learning application that predicts agricultural crop yield using environmental and farming inputs. The goal is to help estimate expected crop production (tons per hectare) based on key growing conditions.

The project follows the full data science workflow:

Problem definition

Exploratory Data Analysis (EDA)

Data cleaning and feature engineering

Regression modeling

Classification modeling

Deployment via Streamlit

Dataset

Dataset: Crop Yield Dataset
Source: Kaggle
File Used: crop_yield_dataset.csv

The dataset contains environmental and agricultural variables that influence crop production.

Problem Statement

Accurately estimating crop yield is critical for farmers, supply chain planners, and agricultural stakeholders. This project aims to predict crop yield (tons per hectare) using key environmental and farming inputs.

Additionally, crop yield is categorized into Low, Medium, and High yield levels to support easier interpretation for decision-making.

Target Variable

Regression Target: Crop Yield (tons per hectare)

Classification Target: Binned crop yield (Low / Medium / High)

Selected Features

After performing EDA and feature analysis, the final models use the following features:

Amount of fertilizer used

Humidity percentage

Rainfall

Temperature

These features were selected based on their relationship with crop yield and their practical relevance in agricultural settings.

Models Used
🔹 Regression Model

Model: Lasso Regression

The Lasso regression model predicts the numerical crop yield (tons per hectare) based on user inputs.

Lasso was chosen because:

It performs regularization

It reduces overfitting

It helps with feature selection

🔹 Classification Model

Model: Logistic Regression

The logistic regression model uses the same input features to classify predicted crop yield into:

Low Yield

Medium Yield

High Yield

This provides a simplified, decision-friendly output for users who prefer category-based predictions.

Streamlit Application

The project is deployed as a Streamlit web app where users can:

Input:

Fertilizer amount

Humidity

Rainfall

Temperature

Receive:

A predicted crop yield (tons per hectare)

A predicted yield category (Low / Medium / High)

The app demonstrates how machine learning models can be integrated into an interactive user interface for real-world usage.

Project Highlights

Completed full EDA and data cleaning pipeline

Built both regression and classification models from the same dataset

Implemented feature selection and regularization (Lasso)

Deployed working ML models in a live Streamlit application

Structured project following professional Git and ML workflow practices

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Git / GitHub