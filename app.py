
import gradio as gr
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from model import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=4, out_channels=1).to(device)

model.load_state_dict(
    torch.load("unet_brats.pth", map_location=device)
)

model.eval()


def normalize(volume):
    mask = volume > 0
    out = np.zeros_like(volume, dtype=np.float32)
    if mask.sum() > 0:
        out[mask] = (volume[mask] - volume[mask].mean()) / (volume[mask].std() + 1e-8)
    return out


def load_mri(filepath):
    volume = nib.load(filepath).get_fdata().astype(np.float32)

    return normalize(volume)


# --------------------------------------------------
# 4. Make a tumor prediction
# --------------------------------------------------

def predict(flair_file, t1_file, t1ce_file, t2_file, slice_index):

    files = [flair_file, t1_file, t1ce_file, t2_file]

    if any(file is None for file in files):
        raise gr.Error("Please upload all four MRI files.")

    # Load and normalize the four modalities
    flair = load_mri(flair_file)
    t1 = load_mri(t1_file)
    t1ce = load_mri(t1ce_file)
    t2 = load_mri(t2_file)

    # Ensure all modalities have matching dimensions
    if not all(volume.shape == flair.shape for volume in [t1, t1ce, t2]):
        raise gr.Error("The four MRI volumes must have matching dimensions.")

    if flair.ndim != 3:
        raise gr.Error("Expected 3D NIfTI MRI volumes.")

    if flair.shape != (240, 240, 155):
        raise gr.Error(f"Expected MRI dimensions (240, 240, 155), got {flair.shape}.")

    # Stack modalities into four input channels
    image = np.stack([flair, t1, t1ce, t2], axis=0)[:, 24:216, 24:216, :]

    # Select the requested 2D slice
    X_slice = image[:, :, :, int(slice_index)]

    # Convert to PyTorch tensor
    X = torch.from_numpy(X_slice.copy()).float()
    X = X.unsqueeze(0).to(device)

    # Run inference
    with torch.inference_mode():
        logits = model(X)

        probabilities = torch.sigmoid(logits)

        prediction = (probabilities > 0.5).float()

    # Convert prediction to NumPy
    predicted_mask = prediction[0, 0].cpu().numpy()

    # Get the corresponding FLAIR slice
    flair_slice = flair[24:216, 24:216, int(slice_index)]

    # --------------------------------------------------
    # 5. Visualize the results
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(flair_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Original FLAIR")

    axes[1].imshow(predicted_mask.T, cmap="gray", origin="lower")
    axes[1].set_title("Predicted Tumor Mask")

    axes[2].imshow(flair_slice.T, cmap="gray", origin="lower")

    overlay = np.ma.masked_where(
        predicted_mask.T == 0,
        predicted_mask.T
    )

    axes[2].imshow(
        overlay,
        cmap="autumn",
        alpha=0.5,
        origin="lower"
    )

    axes[2].set_title("Tumor Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Brain Tumor Segmentation - Slice {int(slice_index)}")

    fig.tight_layout()

    tumor_pixels = int(predicted_mask.sum())

    result = f"Predicted tumor pixels: {tumor_pixels}"

    return fig, result


with gr.Blocks(title="Brain Tumor Segmentation") as demo:

    gr.Markdown(
        """
        # Brain Tumor Segmentation using U-Net

        Upload the four MRI modalities of a patient to predict
        the tumor region on a selected 2D slice.
        """
    )

    gr.Markdown("### Upload MRI files")

    with gr.Row():
        flair_input = gr.File(
            label="FLAIR",
            file_types=[".nii", ".nii.gz"],
            type="filepath"
        )

        t1_input = gr.File(
            label="T1",
            file_types=[".nii", ".nii.gz"],
            type="filepath"
        )

    with gr.Row():
        t1ce_input = gr.File(
            label="T1ce",
            file_types=[".nii", ".nii.gz"],
            type="filepath"
        )

        t2_input = gr.File(
            label="T2",
            file_types=[".nii", ".nii.gz"],
            type="filepath"
        )

    slice_input = gr.Slider(
        minimum=0,
        maximum=154,
        value=77,
        step=1,
        label="Select MRI Slice"
    )

    predict_button = gr.Button("Predict Tumor", variant="primary")

    output_plot = gr.Plot(label="Segmentation Results")

    output_text = gr.Textbox(label="Prediction Information")

    predict_button.click(
        fn=predict,
        inputs=[
            flair_input,
            t1_input,
            t1ce_input,
            t2_input,
            slice_input
        ],
        outputs=[output_plot, output_text]
    )

    gr.Markdown(
        """
        **Disclaimer:** This application is for educational and
        research purposes only. It is not intended for clinical diagnosis.
        """
    )


demo.launch()
