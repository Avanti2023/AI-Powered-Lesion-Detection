import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# Load the trained model
MODEL_PATH = 'skin_lesion_model.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Define class labels and additional info
class_labels = ['cellulitis', 'impetigo', 'athlete-foot', 'nail-fungus', 'ringworm',
                'cutaneous-larva-migrans', 'chickenpox', 'shingles']

treatments = {
    'cellulitis': "Antibiotics, rest, and elevating the affected area.",
    'impetigo': "Antibiotic creams or oral antibiotics.",
    'athlete-foot': "Antifungal medications, keeping feet dry.",
    'nail-fungus': "Oral antifungal drugs, medicated nail polish.",
    'ringworm': "Topical antifungal creams, proper hygiene.",
    'cutaneous-larva-migrans': "Anti-parasitic medications (albendazole, ivermectin).",
    'chickenpox': "Rest, hydration, antihistamines, vaccination for prevention.",
    'shingles': "Antiviral medications, pain relievers."
}

symptoms = {
    'cellulitis': "Red, swollen, tender skin with warmth and pain.",
    'impetigo': "Red sores or blisters with yellow crust.",
    'athlete-foot': "Itchy, burning, cracked skin between toes.",
    'nail-fungus': "Thickened, discolored, brittle nails.",
    'ringworm': "Red, circular, scaly rash with itchiness.",
    'cutaneous-larva-migrans': "Winding, red tracks on skin with itching.",
    'chickenpox': "Itchy red spots or blisters, spreading from face to body.",
    'shingles': "Painful rash with blisters, burning or tingling."
}

# Initialize session state for user history
if "history" not in st.session_state:
    st.session_state.history = []

# Image preprocessing function
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    
    # Top 5 predictions
    top_5_indices = np.argsort(prediction)[-5:][::-1]
    top_5_labels = [(class_labels[i], prediction[i]) for i in top_5_indices]
    
    # Most confident prediction
    max_confidence = np.max(prediction)
    predicted_label = class_labels[np.argmax(prediction)]
    
    if max_confidence < 0.5:  # Confidence threshold
        return "Uncertain Result. Seems like clear skin.", max_confidence, top_5_labels

    return predicted_label, max_confidence, top_5_labels

# Enhanced confidence score visualization
def show_confidence_chart(top_5_labels):
    df = pd.DataFrame({"Condition": [lbl for lbl, _ in top_5_labels],
                       "Confidence": [conf for _, conf in top_5_labels]})
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Condition"],
        x=df["Confidence"],
        orientation='h',
        marker=dict(color=df["Confidence"], colorscale='blues', showscale=True),
        text=[f"{conf*100:.2f}%" for conf in df["Confidence"]],
        textposition='inside',
        hoverinfo='text',
    ))
    
    fig.update_layout(
        title="🔢 Confidence Score Chart",
        xaxis_title="Confidence Level",
        yaxis_title="Predicted Conditions",
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50),
        height=400,
        width=600,
        xaxis=dict(range=[0, 1], showgrid=True),
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.set_page_config(page_title='Skin Lesion Classifier', layout='wide', page_icon='🌟')
st.title('🌟 Skin Lesion Classification App')
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("""## About  
- Uses a deep learning model (MobileNetV2) to classify skin lesions.  
- Upload an image and get instant predictions.  
- View confidence scores and medical recommendations.  
""")

st.sidebar.markdown("---\n")
st.sidebar.markdown("**Supported Classes:**")
st.sidebar.markdown("\n".join([f"- {cls.title()}" for cls in class_labels]))


uploaded_file = st.file_uploader("📷 Upload a skin lesion image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption='🖼 Uploaded Image', use_column_width=True)
    st.write("")  
    st.markdown("## 🔍 Classification Results")

    # Prediction with loading spinner
    with st.spinner("🔬 Analyzing the image..."):
        label, confidence, top_5_labels = predict(image)

    # Save to user history
    st.session_state.history.append({"Image": uploaded_file.name, "Prediction": label, "Confidence": confidence})

    st.subheader(f"🩺 Prediction: *{label}*")
    st.write(f"**Confidence: {confidence:.2f}")
    if confidence < 0.5:
        st.warning("⚠ Model is not confident. Try uploading a clearer image.")

    st.markdown("### 🏆 Top 5 Predictions")
    for lbl, conf in top_5_labels:
        st.markdown(f"- *{lbl.title()}*: {conf:.2f}")

    show_confidence_chart(top_5_labels)
    if label in treatments and label in symptoms:
        with st.expander("🩺 Health Insights: Symptoms & Treatment", expanded=True):
            st.markdown("### 🏥 Recommended Treatment")
            st.info(treatments[label])

            st.markdown("### 🩺 Symptoms")
            st.warning(symptoms[label])


    

    if len(st.session_state.history) > 0:
        st.markdown("---")
        st.markdown("## 📜 Your Prediction History")

        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download History", data=csv_data, file_name="prediction_history.csv", mime="text/csv")
        
            
            
    st.markdown("### ✅ Was this prediction helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No"):
            st.warning("We'll work on improving the model!")