import streamlit as st
import numpy as np
import joblib
import time

# Display the warning message for 5 seconds before loading the main app
if 'warning_shown' not in st.session_state:
    st.session_state.warning_shown = False

if not st.session_state.warning_shown:
    # Create three columns with specified width ratios
    col1, col2, col3 = st.columns([1, 2, 1])    
    with col2:
        st.image("intro.jpg", caption="Important Notice", use_container_width=True)
    # Display the warning message
    st.warning("This app is only for general screening, not for diagnosis. \n Contacting a doctor is a must.")
    time.sleep(5)  #5 second wait
    st.session_state.warning_shown = True
    st.rerun()
else:
    # Load trained model parameters
    model_params = joblib.load("cancer_model.pkl")
    w = model_params["weight"]
    b = model_params["bias"]
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Prediction function
    def predict(input_data):
        global w, b  # Ensure weights are loaded correctly
    
        input_data = np.array(input_data).reshape(-1, 1)  # Reshape dynamically
    
        if input_data.shape[0] != 30:  # Ensure correct feature count
            raise ValueError(f"Expected input_data shape (30,1), but got {input_data.shape}")
    
        z = np.dot(w.T, input_data) + b  # Perform matrix multiplication
        y_head = sigmoid(z)  # Apply sigmoid activation
        return 1 if y_head > 0.5 else 0  # Return binary classification
    
    st.set_page_config(
    page_title="Basic Cancer Predictor",
    page_icon="🎗️",  
    )

    st.sidebar.title("🎗️Machine Learning🎗️\
    Mini-Project")
    # Sidebar: Navigation
    st.sidebar.markdown("🔍 Navigation")
    page = st.sidebar.radio("Go to", ["About Breast Cancer", "Cancer Prediction Tool"])
    
    # ℹ️ **Sidebar - About the App**
    st.sidebar.title("ℹ️ About This Tool")
    st.sidebar.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color:rgb(57, 111, 221); border-left: 5px solidrgb(50, 131, 131);">
        <h4 style="color: #333;">ℹ️ Basic Info</h4>
        <p>This is a <b>Breast Cancer Prediction App</b> that helps screen for benign or malignant tumors based on input features.  
    It uses <i>Logistic Regression</i> trained on the <strong>Breast Cancer Dataset On Kaggle</strong>.</p>
    </div>
""", unsafe_allow_html=True)
    st.sidebar.markdown(""" """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color:rgb(79, 107, 162); border-left: 5px solidrgb(131, 169, 132);">
        <h4 style="color: #333;">👩‍💻 Developer Details</h4>
        <p><b>Developed by:</b> Disha</p>
        <p><b>Email:</b> imdisha4346@gmail.com</p>
        <p><b>GitHub:</b> <a href="https://github.com/Disha4346" target="_blank">github.com/your-github</a></p>
        <p><b>LinkedIn:</b> <a href="www.linkedin.com/in/disha-gupta-795024289" target="_blank">github.com/your-github</a></p>
    </div>
    """, unsafe_allow_html=True)

    if page == "About Breast Cancer":
        st.title("🩺 Understanding Breast Cancer")
        st.markdown("""
            **What is Breast Cancer?**  
            - Uncontrolled growth of abnormal breast cells.
            - Can start in ducts, lobules, or other breast tissues.

            **Symptoms to Look Out For:**  
            - Lump in the breast or armpit.
            - Unusual nipple discharge.
            - Change in breast size, shape, or texture.

            **Risk Factors:**  
            - Family history, genetic mutations, age, and hormonal factors.

            **Early Detection:**  
            - Regular mammograms and self-examinations increase chances of early detection.
        
            For more Information one can refer to the Following informative Video:
        """)
        # Section: Benign vs Malignant Tumors
        st.header("🔍 Benign vs Malignant Tumors")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Benign Tumors")
            st.markdown("""
            - Non-cancerous growths.
            - Do not spread to other parts of the body.
            - Usually slow-growing.
            - Less likely to return after removal.
            - Examples: Fibroadenomas, Cysts.
            """)

        with col2:
            st.subheader("⚠️ Malignant Tumors")
            st.markdown("""
            - Cancerous and aggressive.
            - Can invade nearby tissues and spread (metastasis).
            - Often require surgery, radiation, or chemotherapy.
            - Early detection is crucial for better outcomes.
            """)
        st.video("https://youtu.be/cwDITwyuuYk?si=Hrfbl31jTKNI0HYi")


    elif page=="Cancer Prediction Tool":
        # Streamlit UI
        st.title("🎗️Breast Cancer Prediction App")
        st.warning("📃Enter the tumor features to predict if it's benign or malignant.")

        # 📌 **Category 1: Size Features**
        st.header("📏 Size Features")
        col1, col2, col3 = st.columns(3)
        radius_mean = col1.number_input("Radius Mean", min_value=0.0, step=0.01, format="%.4f")
        perimeter_mean = col2.number_input("Perimeter Mean", min_value=0.0, step=0.01, format="%.4f")
        area_mean = col3.number_input("Area Mean", min_value=0.0, step=0.01, format="%.4f")

        radius_se = col1.number_input("Radius SE", min_value=0.0, step=0.01, format="%.4f")
        perimeter_se = col2.number_input("Perimeter SE", min_value=0.0, step=0.01, format="%.4f")
        area_se = col3.number_input("Area SE", min_value=0.0, step=0.01, format="%.4f")

        radius_worst = col1.number_input("Radius Worst", min_value=0.0, step=0.01, format="%.4f")
        perimeter_worst = col2.number_input("Perimeter Worst", min_value=0.0, step=0.01, format="%.4f")
        area_worst = col3.number_input("Area Worst", min_value=0.0, step=0.01, format="%.4f")

        # 📌 **Category 2: Boundary & Compactness Features**
        st.header("🛑 Boundary & Compactness Features")
        col4, col5, col6 = st.columns(3)
        compactness_mean = col4.number_input("Compactness Mean", min_value=0.0, step=0.01, format="%.4f")
        concavity_mean = col5.number_input("Concavity Mean", min_value=0.0, step=0.01, format="%.4f")
        concave_points_mean = col6.number_input("Concave Points Mean", min_value=0.0, step=0.01, format="%.4f")

        compactness_se = col4.number_input("Compactness SE", min_value=0.0, step=0.01, format="%.4f")
        concavity_se = col5.number_input("Concavity SE", min_value=0.0, step=0.01, format="%.4f")
        concave_points_se = col6.number_input("Concave Points SE", min_value=0.0, step=0.01, format="%.4f")

        compactness_worst = col4.number_input("Compactness Worst", min_value=0.0, step=0.01, format="%.4f")
        concavity_worst = col5.number_input("Concavity Worst", min_value=0.0, step=0.01, format="%.4f")
        concave_points_worst = col6.number_input("Concave Points Worst", min_value=0.0, step=0.01, format="%.4f")

        # 📌 **Category 3: Texture Features**
        st.header("🎨 Texture Features")
        col7, col8 = st.columns(2)
        texture_mean = col7.number_input("Texture Mean", min_value=0.0, step=0.01, format="%.4f")
        smoothness_mean = col8.number_input("Smoothness Mean", min_value=0.0, step=0.01, format="%.4f")

        texture_se = col7.number_input("Texture SE", min_value=0.0, step=0.01, format="%.4f")
        smoothness_se = col8.number_input("Smoothness SE", min_value=0.0, step=0.01, format="%.4f")

        texture_worst = col7.number_input("Texture Worst", min_value=0.0, step=0.01, format="%.4f")
        smoothness_worst = col8.number_input("Smoothness Worst", min_value=0.0, step=0.01, format="%.4f")

        # 📌 **Category 4: Fractal Complexity**
        st.header("🌀 Fractal Complexity")
        col9, col10 = st.columns(2)
        fractal_dimension_mean = col9.number_input("Fractal Dimension Mean", min_value=0.0, step=0.01, format="%.4f")
        symmetry_mean = col10.number_input("Symmetry Mean", min_value=0.0, step=0.01, format="%.4f")

        fractal_dimension_se = col9.number_input("Fractal Dimension SE", min_value=0.0, step=0.01, format="%.4f")
        symmetry_se = col10.number_input("Symmetry SE", min_value=0.0, step=0.01, format="%.4f")

        fractal_dimension_worst = col9.number_input("Fractal Dimension Worst", min_value=0.0, step=0.01, format="%.4f")
        symmetry_worst = col10.number_input("Symmetry Worst", min_value=0.0, step=0.01, format="%.4f")

        st.warning("\
            ✅✅ Kindly Write down all the information correctly for better Results. ✅✅\
                ")
        # 🖥 **Collect All Inputs**
        user_input = [
        radius_mean, perimeter_mean, area_mean, radius_se, perimeter_se, area_se, radius_worst, perimeter_worst, area_worst,
        compactness_mean, concavity_mean, concave_points_mean, compactness_se, concavity_se, concave_points_se,
        compactness_worst, concavity_worst, concave_points_worst,
        texture_mean, smoothness_mean, texture_se, smoothness_se, texture_worst, smoothness_worst,
        fractal_dimension_mean, symmetry_mean, fractal_dimension_se, symmetry_se, fractal_dimension_worst, symmetry_worst
        ]

        # Make prediction when button is clicked
        if st.button("Predict"):
            input_array = np.array(user_input).reshape(-1, 1)  # Ensure correct shape
            try:
                prediction = predict(input_array)
                result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
                st.success(f"Prediction: {result}")
            except ValueError as e:
                st.error(f"Error: {e}")
    