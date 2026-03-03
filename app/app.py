"""
Streamlit App for ML Model Deployment
=====================================

This is your Streamlit application that deploys both your regression and
classification models. Users can input feature values and get predictions.

HOW TO RUN LOCALLY:
    streamlit run app/app.py

HOW TO DEPLOY TO STREAMLIT CLOUD:
    1. Push your code to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Set the main file path to: app/app.py
    5. Deploy!

WHAT YOU NEED TO CUSTOMIZE:
    1. Update the page title and description
    2. Update feature input fields to match YOUR features
    3. Update the model paths if you changed them
    4. Customize the styling if desired

Author: Wesley Houk
Dataset: crop_yield_dataset.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# This must be the first Streamlit command!
st.set_page_config(
    page_title="Crop Yield Predictor",  # TODO: Update with your project name
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource  # Cache the models so they don't reload every time
def load_models():
    """Load all saved models and artifacts."""
    # Get the path to the models directory
    # This works both locally and on Streamlit Cloud
    base_path = Path(__file__).parent.parent / "models"

    models = {}

    try:
        # Load regression model and scaler
        models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
        models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
        models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

        # Load classification model and artifacts
        models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
        models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
        models['label_encoder'] = joblib.load(base_path / "label_encoder.pkl")
        models['classification_features'] = joblib.load(base_path / "classification_features.pkl")

        # Optional: Load binning info for display
        try:
            models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
        except:
            models['binning_info'] = None

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Make sure you've trained and saved your models in the notebooks first!")
        return None

    return models


def make_regression_prediction(models, input_data):
    """Make a regression prediction."""
    # Scale the input
    input_scaled = models['regression_scaler'].transform(input_data)
    # Predict
    prediction = models['regression_model'].predict(input_scaled)
    return prediction[0]


def make_classification_prediction(models, input_data):
    """Make a classification prediction."""
    # Scale the input
    input_scaled = models['classification_scaler'].transform(input_data)
    # Predict
    prediction = models['classification_model'].predict(input_scaled)
    # Get label
    label = models['label_encoder'].inverse_transform(prediction)
    return label[0], prediction[0]


# =============================================================================
# SIDEBAR - Navigation
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a model:",
    ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app deploys machine learning models trained on crop_yield_dataset.csv.

    - **Regression**: Predicts Crop Yield - Ton/Hectare
    - **Classification**: Predicts low, medium or high yield
    """
)
# TODO: UPDATE YOUR NAME HERE! This shows visitors who built this app.
st.sidebar.markdown("**Built by:** Wesley Houk")
st.sidebar.markdown("[GitHub Repo](https://github.com/wesleyhouk-gif/crop-yield-predictor.git)")


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.title("🤖 Machine Learning Prediction App")
    st.markdown("### Welcome!")

    st.write(
        """
        This application allows you to make predictions using trained machine learning models.

        **What you can do:**
        - 📈 **Regression Model**: Predict a numerical value
        - 🏷️ **Classification Model**: Predict a category

        Use the sidebar to navigate between different models.
        """
    )

    # TODO: Add more information about your specific project
    st.markdown("---")
    st.markdown("### About This Project")
    st.write(
        """
        **Dataset:** This dataset contains agricultural data for 1,000,000 samples aimed at predicting crop yield (in tons per hectare) based on various factors. The dataset can be used for regression tasks in machine learning, especially for predicting crop productivity.

        **Problem Statement:** I am trying to predict the crop yield in tons per hectare. This is very important for farmers. An increase in yeild for them is important for many reasons whether they are trying to grow food for their families, their animals, or just trying to turn a profit. With this information they can plan on what they need to grow based off of their climate, how much fertilizer they need to use, irrigation needed, planting density etc. 

        **Models Used:**
        - Regression: Lasso Regression Model
        - Classification: Logistic Regression Model
        """
    )

    # Show a sample of your data or an image (optional)
    # st.image("path/to/image.png", caption="Sample visualization")
    img_path = Path(__file__).parent / "data.PNG"   # if data.PNG is in the same folder as app.py
    st.image(str(img_path), caption="Sample visualization")
    
# =============================================================================
# REGRESSION PAGE
# =============================================================================
elif page == "📈 Regression Model":
    st.title("📈 Regression Prediction")
    st.write("Enter feature values to get a numerical prediction.")

    # Load models
    models = load_models()

    if models is None:
        st.stop()

    # Get feature names
    features = models['regression_features']

    st.markdown("---")
    st.markdown("### Enter Feature Values")

    # Create input fields for each feature
    # TODO: CUSTOMIZE THIS SECTION FOR YOUR FEATURES!
    # The example below creates number inputs, but you may need:
    # - st.selectbox() for categorical features
    # - st.slider() for bounded numerical features
    # - Different default values and ranges

    # Create columns for better layout
    col1, col2 = st.columns(2)

    input_values = {}

    feature_config = {
        "Rainfall_mm": (200.0, 1499.7, 800.00),
        "Humidity_pct": (30.0, 90.0, 60.00),
        "Temperature_C": (15.0, 35.0, 25.00),
        "Fertilizer_Used_kg": (50.0, 300.0, 175.00)
    }

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            min_val, max_val, default = feature_config[feature]

            input_values[feature] = st.slider(
                label=feature,
                min_value=min_val,
                max_value=max_val,
                value=default,
                help=f"Enter value for {feature}"
            )

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Make Regression Prediction", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame([input_values])

        # Make prediction
        prediction = make_regression_prediction(models, input_df)

        # Display result
        st.success(f"### Predicted Value: {prediction:,.2f}")

        # TODO: Add context to your prediction
        st.write(f"This is the predicted crop yield in tons per hectare based on these inputs")

        # Show input summary
        with st.expander("View Input Summary"):
            st.dataframe(input_df)


# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================
elif page == "🏷️ Classification Model":
    st.title("🏷️ Classification Prediction")
    st.write("Enter feature values to get a category prediction.")

    # Load models
    models = load_models()

    if models is None:
        st.stop()

    # Get feature names and class labels
    features = models['classification_features']
    class_labels = models['label_encoder'].classes_

    # Show the possible categories
    st.info(f"**Possible Categories:** {', '.join(class_labels)}")

    # Show binning info if available
    if models['binning_info']:
        with st.expander("How were categories created?"):
            binning = models['binning_info']
            st.write(f"Original target: **{binning['original_target']}**")
            st.write("Categories were created by binning the numerical values:")
            for i, label in enumerate(binning['labels']):
                if i == 0:
                    st.write(f"- **{label}**: < {binning['bins'][i+1]}")
                elif i == len(binning['labels']) - 1:
                    st.write(f"- **{label}**: >= {binning['bins'][i]}")
                else:
                    st.write(f"- **{label}**: {binning['bins'][i]} to {binning['bins'][i+1]}")

    st.markdown("---")
    st.markdown("### Enter Feature Values")

    # Create input fields
    # TODO: CUSTOMIZE THIS SECTION FOR YOUR FEATURES!

    col1, col2 = st.columns(2)

    input_values = {}

    feature_config = {
        "Rainfall_mm": (200.0, 1499.7, 800.00),
        "Humidity_pct": (30.0, 90.0, 60.00),
        "Temperature_C": (15.0, 35.0, 25.00),
        "Fertilizer_Used_kg": (50.0, 300.0, 175.00)
    }

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            min_val, max_val, default = feature_config[feature]

            input_values[feature] = st.slider(
                label=feature,
                min_value=min_val,
                max_value=max_val,
                value=default,
                key=f"class_{feature}",  # Unique key for classification inputs
                help=f"Enter value for {feature}"
            )

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Make Classification Prediction", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame([input_values])

        # Make prediction
        predicted_label, predicted_index = make_classification_prediction(models, input_df)

        # Display result with color coding
        # TODO: Customize colors based on your categories
        color_map = {
            'Low': '🔴',
            'Medium': '🟡',
            'High': '🟢'
        }
        emoji = color_map.get(predicted_label, '🔵')

        st.success(f"### Predicted Category: {emoji} {predicted_label}")

        # TODO: Add interpretation
        st.write(f"If it is a bad yield predicted the model will show low. If it is a medium, or average yeild then the model will predict medium. If it is a really good yield then the model will predict high.")

        # Show input summary
        with st.expander("View Input Summary"):
            st.dataframe(input_df)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built by Wesley Houk | Full Stack Academy AI & ML Bootcamp
    </div>
    """,
    unsafe_allow_html=True
)
# TODO: Replace [YOUR NAME] above with your actual name!
