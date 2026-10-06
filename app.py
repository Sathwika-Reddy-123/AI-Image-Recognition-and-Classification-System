import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="AI Image Recognition System",
    page_icon="🖼️"
)

st.title("AI Image Recognition and Classification System")
st.write("Upload an image to recognize and classify it.")

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the AI model.")
    st.error(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Recognize Image"):

        with st.spinner("Analyzing image..."):

            image_resized = image.resize((224, 224))

            image_array = np.array(
                image_resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = (
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            results = (
                tf.keras.applications.mobilenet_v2.decode_predictions(
                    predictions,
                    top=5
                )[0]
            )

        st.success("Image classification completed!")

        st.subheader("Prediction Results")

        for rank, (_, class_name, probability) in enumerate(
            results,
            start=1
        ):
            st.write(
                f"{rank}. {class_name} — "
                f"{probability * 100:.2f}%"
            )

        best_class = results[0][1]
        best_probability = results[0][2] * 100

        st.subheader("Final Prediction")

        st.success(
            f"{best_class} "
            f"({best_probability:.2f}% confidence)"
        )

else:
    st.info("Please upload a JPG, JPEG, or PNG image.")