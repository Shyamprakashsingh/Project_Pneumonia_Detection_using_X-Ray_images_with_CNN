# import streamlit as st
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the trained model
# model = load_model("pneumonia_cnn_model.h5")
# class_labels = ['NORMAL', 'PNEUMONIA']

# # Title
# st.title("Pneumonia Detection from Chest X-ray")
# st.write("Upload a chest X-ray image (150x150 or any size, it will be resized).")

# # Upload image
# uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read and preprocess the image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     image_resized = cv2.resize(image, (150, 150))
#     image_array = img_to_array(image_resized) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # Show image
#     st.image(image, channels="BGR", caption="Uploaded X-ray", use_column_width=True)

#     # Predict
#     prediction = model.predict(image_array)
#     label_index = np.argmax(prediction)
#     label = class_labels[label_index]
#     confidence = float(prediction[0][label_index]) * 100

#     # Show prediction
#     st.subheader(f"Prediction: {label}")
#     st.write(f"Confidence: {confidence:.2f}%")

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Load model
model = load_model("pneumonia_cnn_model.h5")
class_labels = ['NORMAL', 'PNEUMONIA']

# App title and description
st.set_page_config(page_title="X-ray Pneumonia Detector", page_icon="🫁", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown("""
Welcome to the Pneumonia Detection App!  
This tool uses a deep learning model trained on chest X-ray images to determine whether signs of **Pneumonia** are present.

### 🩺 How it works:
1. Upload a chest X-ray image.
2. The AI model processes the image.
3. The result shows whether the X-ray is **Normal** or indicates **Pneumonia**.

**Note**: This is for educational and preliminary analysis only, not a substitute for medical diagnosis.
""")

# Sidebar info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Chest_Xray_PA_3-8-2010.png/440px-Chest_Xray_PA_3-8-2010.png", caption="Sample X-ray")
    st.markdown("""
    **About Pneumonia**  
    Pneumonia is an infection that inflames air sacs in one or both lungs. It can range in seriousness and is particularly dangerous for infants, older adults, and people with weakened immune systems.

    - Caused by bacteria, viruses, or fungi  
    - Detected through X-rays, physical exams, and lab tests  
    - Early detection can prevent complications  
    """)

# Upload image
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Show uploaded image
    st.image(image_rgb, caption="Uploaded X-ray Image", use_column_width=True)
    st.info("✅ Image successfully uploaded. Running prediction...")

    # Predict
    prediction = model.predict(image_array)[0]
    label_index = np.argmax(prediction)
    label = class_labels[label_index]
    confidence = float(prediction[label_index]) * 100

    # Display result
    st.subheader(f"🔍 Prediction: `{label}`")
    st.progress(confidence / 100)
    st.write(f"**Confidence:** `{confidence:.2f}%`")

    # Add helpful interpretation
    if label == "PNEUMONIA":
        st.warning("⚠️ The image may show signs of pneumonia. Please consult a medical professional.")
    else:
        st.success("✅ The image appears normal. However, consult a doctor for accurate diagnosis.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using TensorFlow and Streamlit")
