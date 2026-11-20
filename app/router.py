
from utils.xray_inference import (
    load_xray_model,
    predict_xray,
)

from utils.mri_inference import (
    load_mri_model,
    predict_mri,
)

from utils.ultrasound_inference import (
    load_ultrasound_model,
    predict_ultrasound,
)


# --------------------------------------------------
# MODEL ROUTER
# --------------------------------------------------

class MedicalModelRouter:

    def __init__(self):

        # Models are loaded only when first requested.
        self.loaded_models = {}

        print("Medical AI Router initialized.")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    def get_model(self, modality, subtype=None):

        modality = modality.lower()

        if modality == "xray":

            key = "xray"

            if key not in self.loaded_models:
                print("Loading X-ray model...")
                self.loaded_models[key] = load_xray_model()

            return self.loaded_models[key]

        elif modality == "mri":

            key = "mri"

            if key not in self.loaded_models:
                print("Loading MRI model...")
                self.loaded_models[key] = load_mri_model()

            return self.loaded_models[key]

        elif modality == "ultrasound":

            if subtype is None:
                raise ValueError(
                    "Select ultrasound subtype: "
                    "breast, ovarian, abdominal or fetal."
                )

            subtype = subtype.lower()

            valid_subtypes = [
                "breast",
                "ovarian",
                "abdominal",
                "fetal",
            ]

            if subtype not in valid_subtypes:
                raise ValueError(
                    f"Invalid ultrasound subtype: {subtype}"
                )

            key = f"ultrasound_{subtype}"

            if key not in self.loaded_models:
                print(f"Loading {subtype} ultrasound model...")

                self.loaded_models[key] = (
                    load_ultrasound_model(subtype)
                )

            return self.loaded_models[key]

        else:

            raise ValueError(
                "Invalid modality. Choose xray, mri or ultrasound."
            )

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, image, modality, subtype=None):

        modality = modality.lower()

        model = self.get_model(modality, subtype)

        if modality == "xray":

            result = predict_xray(image, model)

        elif modality == "mri":

            result = predict_mri(image, model)

        elif modality == "ultrasound":

            result = predict_ultrasound(image, model)

        else:

            raise ValueError(
                f"Unsupported modality: {modality}"
            )

        return result

    # --------------------------------------------------
    # AVAILABLE MODELS
    # --------------------------------------------------

    @staticmethod
    def available_models():

        return {
            "xray": [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia",
            ],

            "mri": [
                "glioma",
                "meningioma",
                "notumor",
                "pituitary",
            ],

            "ultrasound": {
                "breast": [
                    "benign",
                    "malignant",
                    "normal",
                ],

                "ovarian": [
                    "complex_cyst",
                    "dominant_follicle",
                    "healthy",
                    "poly_cyst",
                    "simple_cyst",
                ],

                "abdominal": [
                    "AA",
                    "GB",
                    "Hepatic",
                    "Kidney",
                    "Liver",
                    "Ovaries",
                    "Pancreas",
                    "Portal",
                    "Spleen",
                    "UB+prostrate+uterous+cervix",
                ],

                "fetal": "AOP angle regression",
            },
        }


# --------------------------------------------------
# QUICK CHECK
# --------------------------------------------------

if __name__ == "__main__":

    router = MedicalModelRouter()

    print("\nAvailable models:")

    for modality, details in router.available_models().items():

        print(f"\n{modality.upper()}:")

        if isinstance(details, dict):

            for subtype, classes in details.items():
                print(f"  {subtype}: {classes}")

        else:
            print(f"  {details}")

    print("\nRouter initialized successfully.")
