import pickle
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score
from breastcancer import preprocess_data  # Pastikan fungsi ini ada di breastcancer.py

# Load the model
model = pickle.load(open('breastcancer.sav', 'rb'))

# Load the test dataset
test_data = pd.read_csv('X_test.csv')  # Pastikan file ini tersedia di direktori yang sama

# Assuming the last column is the target variable
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Preprocess the test data
X_test_processed = X_test.apply(preprocess_data, axis=1)

# Calculate the accuracy on the test dataset
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the front end interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Pre-processed Data Results"])

# Prediction Page
if page == "Prediction":
    # Load and display an image
    image = Image.open("breast_cancer_image.png")  # Pastikan file gambar tersedia di direktori yang sama atau berikan path yang benar
    st.image(image, use_column_width=True)

    st.title('Breast Cancer Prediction')

    st.markdown(
        """
        ### Please enter the following details:
        """
    )

    # Layout for input fields
    col1, col2 = st.columns(2)

    with col1:
        clump_thickness = st.number_input('Clump Thickness')
        cell_size_uniformity = st.number_input('Cell Size Uniformity')
        cell_shape_uniformity = st.number_input('Cell Shape Uniformity')
        marginal_adhesion = st.number_input('Marginal Adhesion')
        single_epi_cell_size = st.number_input('Single Epithelial Cell Size')

    with col2:
        bare_nuclei = st.number_input('Bare Nuclei')
        bland_chromatin = st.number_input('Bland Chromatin')
        normal_nucleoli = st.number_input('Normal Nucleoli')
        mitoses = st.number_input('Mitoses')

    # Make predictions
    if st.button('Predict'):
        input_data = (
            clump_thickness, cell_size_uniformity, cell_shape_uniformity, 
            marginal_adhesion, single_epi_cell_size, bare_nuclei, 
            bland_chromatin, normal_nucleoli, mitoses
        )
        
        # Pre-process the input data
        preprocessed_data = preprocess_data(input_data)
        
        prediction = model.predict([preprocessed_data])

        st.markdown(
            """
            ### Prediction Result:
            """
        )

        if prediction[0] == 0:
            st.success('Kanker Anda Termasuk Kanker **Benign (Jinak)**.')
        else:
            st.error('Kanker Anda Termasuk Kanker **Malignant (Ganas)**.')

        # Display the model accuracy
        st.markdown(
            """
            ### Model Accuracy:
            """
        )
        st.write(f"The model accuracy on the test dataset is {accuracy:.2f}")

    # Footer
    st.markdown(
        """
        <hr>
        <p style="text-align:center;">Kelompok 1 - 21S1SI-Machine2(SI163) </p>
        """,
        unsafe_allow_html=True
    )

# Pre-processed Data Results Page
elif page == "Pre-processed Data Results":
    st.title('Pre-processed Data Results')

    st.markdown(
        """
        ### Here you can check the results of the pre-processed data.
        """
    )

    # Load the pre-processed dataset
    preprocessed_data_df = pd.read_csv('preprocessed_data.csv')  # Pastikan file ini tersedia di direktori yang sama

    # Display the pre-processed data
    st.dataframe(preprocessed_data_df)

    # Footer
    st.markdown(
        """
        <hr>
        <p style="text-align:center;">Kelompok 1 - 21S1SI-Machine2(SI163) </p>
        """,
        unsafe_allow_html=True
    )
