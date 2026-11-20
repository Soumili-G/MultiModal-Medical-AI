
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


# ==================================================
# 1. MODALITY PROMPTS
# ==================================================

MODALITY_PROMPTS = {
    "mri": [
        "A brain MRI scan",
        "A magnetic resonance imaging scan of the brain",
        "An axial brain MRI medical image",
    ],

    "xray": [
        "A frontal chest X-ray radiograph",
        "A chest X-ray showing the lungs",
        "A lateral chest radiograph",
    ],

    "ultrasound": [
        "A diagnostic medical ultrasound sonogram",
        "A grayscale ultrasound image of internal organs",
        "A medical sonography scan",
    ],
}


# ==================================================
# 2. ULTRASOUND SUBTYPE PROMPTS
# ==================================================

ULTRASOUND_SUBTYPE_PROMPTS = {

    "breast": [
        "A breast ultrasound scan showing breast tissue",
        "A sonogram of a breast lesion",
        "A medical ultrasound image of breast tissue",
        "An ultrasound scan of the human breast",
    ],

    "ovarian": [
        "A transvaginal ultrasound scan of the ovaries",
        "An ovarian ultrasound showing ovarian follicles",
        "A sonogram of an ovarian cyst",
        "A pelvic ultrasound showing the ovaries",
    ],

    "abdominal": [
        "An abdominal ultrasound scan of internal organs",
        "An ultrasound image of the liver and gallbladder",
        "A sonogram of abdominal organs",
        "An abdominal sonography examination",
        "An ultrasound image of the kidneys",
    ],

    "fetal": [
        "An obstetric fetal ultrasound scan",
        "A transperineal ultrasound showing the fetal head",
        "An ultrasound image used to measure angle of progression",
        "A fetal sonography image",
    ],
}


# ==================================================
# 3. LOAD CLIP VALIDATOR
# ==================================================

def load_image_validator():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.to(device)

    model.eval()

    return {
        "model": model,
        "processor": processor,
        "device": device,
    }


# ==================================================
# 4. GET SIMILARITY SCORES
# ==================================================

def get_similarity_scores(image, prompts, validator):

    inputs = validator["processor"](
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(validator["device"])
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = validator["model"](**inputs)

    return outputs.logits_per_image[0].cpu()


# ==================================================
# 5. VALIDATE MEDICAL IMAGE
# ==================================================

def validate_medical_image(
    image,
    modality,
    validator,
    subtype=None,
):

    modality = modality.lower()

    # ----------------------------------------------
    # CHECK SUPPORTED MODALITY
    # ----------------------------------------------

    if modality not in MODALITY_PROMPTS:

        return (
            False,
            "Unsupported imaging modality."
        )

    # ----------------------------------------------
    # STEP 1: MRI AND X-RAY SCREENING
    # ----------------------------------------------

    # Ultrasound is not rejected using the general
    # CLIP modality comparison because this comparison
    # was producing false rejections.

    if modality in ["mri", "xray"]:

        modality_names = list(MODALITY_PROMPTS.keys())

        all_prompts = []

        prompt_ranges = {}

        for name in modality_names:

            start = len(all_prompts)

            all_prompts.extend(
                MODALITY_PROMPTS[name]
            )

            prompt_ranges[name] = (
                start,
                len(all_prompts)
            )

        scores = get_similarity_scores(
            image,
            all_prompts,
            validator,
        )

        modality_scores = {
            name: scores[start:end].max().item()
            for name, (start, end) in prompt_ranges.items()
        }

        detected_modality = max(
            modality_scores,
            key=modality_scores.get,
        )

        if detected_modality != modality:

            return (
                False,
                f"Modality mismatch. The image appears more "
                f"similar to {detected_modality.upper()} than "
                f"{modality.upper()}. Please select the correct "
                f"imaging modality."
            )

    # ----------------------------------------------
    # STEP 2: ULTRASOUND SUBTYPE VALIDATION
    # ----------------------------------------------

    if modality == "ultrasound":

        if subtype is None:

            return (
                False,
                "Please select an ultrasound subtype."
            )

        subtype = subtype.lower()

        if subtype not in ULTRASOUND_SUBTYPE_PROMPTS:

            return (
                False,
                "Invalid ultrasound subtype."
            )

        subtype_names = list(
            ULTRASOUND_SUBTYPE_PROMPTS.keys()
        )

        subtype_prompts = []

        subtype_ranges = {}

        for name in subtype_names:

            start = len(subtype_prompts)

            subtype_prompts.extend(
                ULTRASOUND_SUBTYPE_PROMPTS[name]
            )

            subtype_ranges[name] = (
                start,
                len(subtype_prompts)
            )

        subtype_scores = get_similarity_scores(
            image,
            subtype_prompts,
            validator,
        )

        detected_subtype_scores = {
            name: subtype_scores[start:end].max().item()
            for name, (start, end) in subtype_ranges.items()
        }

        detected_subtype = max(
            detected_subtype_scores,
            key=detected_subtype_scores.get,
        )

        if detected_subtype != subtype:

            return (
                False,
                f"Ultrasound subtype mismatch. The image appears "
                f"more similar to {detected_subtype.upper()} "
                f"ultrasound than {subtype.upper()} ultrasound. "
                f"Please select the correct subtype."
            )

    # ----------------------------------------------
    # VALIDATION PASSED
    # ----------------------------------------------

    return (
        True,
        "Image passed preliminary modality screening."
    )
