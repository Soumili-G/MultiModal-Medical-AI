
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

MRI_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

XRAY_CLASSES = [
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
]

ULTRASOUND_CLASSES = {
    "breast": ["benign", "malignant", "normal"],
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
}


# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------

def get_image_transform(modality, subtype=None, bundle=None):

    if modality == "ultrasound" and bundle is not None:
        return bundle["transform"]

    if modality == "mri":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# --------------------------------------------------
# HEATMAP COLORING
# --------------------------------------------------

def colorize_heatmap(heatmap):

    heatmap = np.clip(heatmap, 0, 1)

    red = np.clip(2 * heatmap, 0, 1)
    green = np.clip(2 * heatmap - 0.5, 0, 1)
    blue = np.clip(2 * heatmap - 1, 0, 1)

    colored = np.stack(
        [red, green, blue],
        axis=-1
    )

    colored = (colored * 255).astype(np.uint8)

    return Image.fromarray(colored)


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def generate_gradcam(
    image,
    model,
    modality,
    subtype=None,
    prediction=None,
    target_class=None,

):
    """
    Generate a Grad-CAM explanation for a ResNet18 model.

    Supports:
        - MRI classification
        - X-ray multi-label classification
        - Breast ultrasound classification
        - Ovarian ultrasound classification
        - Abdominal ultrasound classification
        - Fetal ultrasound AOP regression

    For ultrasound, model must be the bundle returned by
    load_ultrasound_model().
    """

    modality = modality.lower()

    # ----------------------------------------------
    # GET MODEL AND DEVICE
    # ----------------------------------------------

    bundle = None

    if modality == "ultrasound":
        if not isinstance(model, dict):
            raise ValueError(
                "Ultrasound Grad-CAM requires the model bundle "
                "returned by load_ultrasound_model()."
            )

        bundle = model
        network = bundle["model"]
        device = bundle.get(
            "device",
            next(network.parameters()).device
        )

        model_type = bundle.get("model_type", subtype)

        if model_type is not None:
            subtype = model_type.lower()

    else:
        network = model
        device = next(network.parameters()).device

    network.eval()

    # ----------------------------------------------
    # CHECK MODEL LAYER
    # ----------------------------------------------

    if not hasattr(network, "layer3"):
        raise ValueError(
            "Grad-CAM expects a ResNet model containing layer3."
        )

    target_layer = network.layer3[-1]

    # ----------------------------------------------
    # PREPARE IMAGE
    # ----------------------------------------------

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    original_image = image.convert("RGB")

    transform = get_image_transform(
        modality,
        subtype,
        bundle
    )

    input_tensor = transform(original_image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # ----------------------------------------------
    # ACTIVATION AND GRADIENT STORAGE
    # ----------------------------------------------

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.update({"value": grad})
        )

    hook = target_layer.register_forward_hook(forward_hook)

    try:

        # ------------------------------------------
        # FORWARD PASS
        # ------------------------------------------

        with torch.enable_grad():

            network.zero_grad(set_to_none=True)

            outputs = network(input_tensor)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # --------------------------------------
            # SELECT TARGET
            # --------------------------------------

            if modality == "ultrasound" and subtype == "fetal":

                target_score = outputs.reshape(-1)[0]
                target_class = "AOP Regression"
                displayed_score = float(
                    target_score.detach().cpu().item()
                )

            elif modality == "xray":

                probabilities = torch.sigmoid(outputs)

                target_index = int(
                    probabilities[0].argmax().item()
                )

                target_score = outputs[0, target_index]
                target_class = XRAY_CLASSES[target_index]

                displayed_score = float(
                    probabilities[0, target_index]
                    .detach().cpu().item()
                )

            else:

                probabilities = torch.softmax(outputs, dim=1)

                target_index = int(
                    probabilities[0].argmax().item()
                )

                target_score = outputs[0, target_index]

                if modality == "mri":
                    class_names = MRI_CLASSES

                elif modality == "ultrasound":
                    class_names = ULTRASOUND_CLASSES.get(subtype)

                    if class_names is None:
                        raise ValueError(
                            f"Unknown ultrasound subtype: {subtype}"
                        )

                else:
                    raise ValueError(
                        f"Unsupported modality: {modality}"
                    )

                target_class = class_names[target_index]

                displayed_score = float(
                    probabilities[0, target_index]
                    .detach().cpu().item()
                )

            # --------------------------------------
            # BACKWARD PASS
            # --------------------------------------

            target_score.backward()

        # ------------------------------------------
        # GET ACTIVATIONS AND GRADIENTS
        # ------------------------------------------

        if "value" not in activations:
            raise RuntimeError("Grad-CAM activation was not captured.")

        if "value" not in gradients:
            raise RuntimeError("Grad-CAM gradient was not captured.")

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()

        # ------------------------------------------
        # CALCULATE GRAD-CAM
        # ------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * activation).sum(dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam = cam.detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # ------------------------------------------
        # CREATE HEATMAP
        # ------------------------------------------

        grayscale_heatmap = Image.fromarray(
            (cam * 255).astype(np.uint8)
        )

        colored_heatmap = colorize_heatmap(cam)

        # ------------------------------------------
        # CREATE OVERLAY
        # ------------------------------------------

        colored_heatmap = colored_heatmap.resize(
            original_image.size,
            Image.Resampling.BILINEAR
        )

        grayscale_heatmap = grayscale_heatmap.resize(
            original_image.size,
            Image.Resampling.BILINEAR
        )

        overlay = Image.blend(
            original_image,
            colored_heatmap,
            alpha=0.40
        )

        # ------------------------------------------
        # RETURN RESULTS
        # ------------------------------------------

        return {
            "modality": modality,
            "subtype": subtype,
            "target_class": target_class,
            "target_score": displayed_score,
            "heatmap": colored_heatmap,
            "grayscale_heatmap": grayscale_heatmap,
            "overlay": overlay,
        }

    finally:
        hook.remove()
        network.zero_grad(set_to_none=True)
