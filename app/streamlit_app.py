
import streamlit as st
from PIL import Image, UnidentifiedImageError
import pandas as pd

from app.router import MedicalModelRouter

from utils.gradcam import generate_gradcam

from utils.image_validation import (
    load_image_validator,
    validate_medical_image,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multimodal Medical AI",
    page_icon="🧠",
    layout="wide",
)

st.title("Multimodal Medical AI")

st.caption(
    "AI-assisted medical image analysis and visualization"
)

st.warning(
    "Research prototype only. Predictions are not medical "
    "diagnoses. Do not use this application for clinical decisions."
)


# --------------------------------------------------
# LOAD MODELS AND VALIDATOR
# --------------------------------------------------

@st.cache_resource
def get_router():

    return MedicalModelRouter()


@st.cache_resource
def get_validator():

    return load_image_validator()


router = get_router()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Image Analysis")

modality_label = st.sidebar.selectbox(
    "Select imaging modality",
    ["MRI", "X-ray", "Ultrasound"],
)

modality = (
    modality_label.lower()
    .replace("-", "")
    .replace(" ", "")
)

subtype = None


# --------------------------------------------------
# ULTRASOUND SUBTYPE SELECTION
# --------------------------------------------------

if modality == "ultrasound":

    subtype = st.sidebar.selectbox(
        "Select ultrasound type",
        [
            "Breast",
            "Ovarian",
            "Abdominal",
            "Fetal",
        ],
    ).lower()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a medical image",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=False,
)


if uploaded_file is not None:

    # ----------------------------------------------
    # CHECK WHETHER FILE IS A VALID IMAGE
    # ----------------------------------------------

    try:

        image = Image.open(uploaded_file)

        image.verify()

        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):

        st.error(
            "Invalid image file. Please upload a valid JPG, "
            "JPEG, PNG or BMP image."
        )

        st.stop()

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True,
    )

    # ----------------------------------------------
    # ANALYZE BUTTON
    # ----------------------------------------------

    if st.button("Analyze Image", type="primary"):

        # ==========================================
        # STEP 1: MEDICAL IMAGE SCREENING
        # ==========================================

        try:

            with st.spinner(
                "Checking imaging modality and subtype..."
            ):

                validator = get_validator()

                is_valid, validation_message = (
                    validate_medical_image(
                        image=image,
                        modality=modality,
                        validator=validator,
                        subtype=subtype,
                    )
                )

            if not is_valid:

                st.error(validation_message)

                st.warning(
                    "Analysis stopped. No medical prediction was made."
                )

                st.stop()

            st.success(
                "Image passed preliminary modality screening."
            )

        except Exception as error:

            st.error(
                f"Image validation failed: {error}"
            )

            st.stop()

        # ==========================================
        # STEP 2: MODEL PREDICTION AND GRAD-CAM
        # ==========================================

        try:

            with st.spinner(
                "Analyzing medical image..."
            ):

                prediction = router.predict(
                    image=image,
                    modality=modality,
                    subtype=subtype,
                )

                model = router.get_model(
                    modality,
                    subtype,
                )

                # ----------------------------------
                # SELECT GRAD-CAM TARGET
                # ----------------------------------

                if modality == "xray":

                    scores = prediction["disease_scores"]

                    gradcam_target = max(
                        scores,
                        key=scores.get,
                    )

                elif (
                    modality == "ultrasound"
                    and subtype == "fetal"
                ):

                    gradcam_target = None

                else:

                    gradcam_target = prediction[
                        "predicted_class"
                    ]

                explanation = generate_gradcam(
                    image=image,
                    modality=modality,
                    model=model,
                    subtype=subtype,
                    target_class=gradcam_target,
                )

            st.success(
                "Analysis completed successfully!"
            )

            # ======================================
            # STEP 3: PREDICTION RESULTS
            # ======================================

            st.header("Prediction Results")

            # --------------------------------------
            # FETAL AOP REGRESSION
            # --------------------------------------

            if (
                modality == "ultrasound"
                and subtype == "fetal"
            ):

                st.metric(
                    "Predicted AOP Angle",
                    f"{prediction['predicted_aop_degrees']:.2f}°",
                )

                st.caption(
                    "Experimental regression estimate. "
                    "Not a clinical measurement."
                )

            # --------------------------------------
            # X-RAY MULTI-LABEL CLASSIFICATION
            # --------------------------------------

            elif modality == "xray":

                st.subheader(
                    "Chest X-ray Disease Scores"
                )

                score_df = pd.DataFrame(
                    [
                        {
                            "Finding": name,
                            "Model score": score,
                        }
                        for name, score in
                        prediction["disease_scores"].items()
                    ]
                )

                score_df["Model score"] = (
                    score_df["Model score"] * 100
                ).round(2)

                st.dataframe(
                    score_df.sort_values(
                        "Model score",
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "These are model scores, not confirmed disease "
                    "probabilities or diagnoses."
                )

            # --------------------------------------
            # MRI AND CLASSIFICATION ULTRASOUND
            # --------------------------------------

            else:

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Class",
                        prediction["predicted_class"],
                    )

                with col2:

                    st.metric(
                        "Model Score",
                        f"{prediction['confidence'] * 100:.2f}%",
                    )

                if "class_probabilities" in prediction:

                    probability_df = pd.DataFrame(
                        [
                            {
                                "Class": name,
                                "Score (%)": round(
                                    value * 100,
                                    2,
                                ),
                            }
                            for name, value in
                            prediction[
                                "class_probabilities"
                            ].items()
                        ]
                    )

                    st.subheader(
                        "Class-wise Scores"
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Classification scores are not clinically "
                        "validated probabilities."
                    )

            # ======================================
            # STEP 4: GRAD-CAM VISUALIZATION
            # ======================================

            st.header("Grad-CAM Visualization")

            st.caption(
                "Highlights image regions contributing to "
                "the selected model output. These are not "
                "validated anatomical or diagnostic boundaries."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Image")

                st.image(
                    image,
                    use_container_width=True,
                )

            with col2:

                st.subheader("Grad-CAM Overlay")

                st.image(
                    explanation["overlay"],
                    use_container_width=True,
                )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

            st.exception(error)


else:

    st.info(
        "Upload an image to begin analysis."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Multimodal Medical AI | Research and educational prototype"
)
