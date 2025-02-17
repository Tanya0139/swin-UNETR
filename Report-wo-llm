import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNETR
from monai.transforms import Resize, ScaleIntensity, Compose
from fpdf import FPDF

# Paths
MODEL_PATH = r"C:\Users\tanya\OneDrive\Desktop\tantan\combined_work_pjt\saved_model_2020\unetr_finetuned.pth"
INPUT_IMAGE_PATH = r"C:\Users\tanya\OneDrive\Desktop\tantan\combined_work_pjt\BRATS_2020_DATA\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_018\BraTS20_Training_018_seg.nii"  # Change this
OUTPUT_PDF_PATH = r"C:\Users\tanya\OneDrive\Desktop\tantan\segmentation_report.pdf"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model(model_path):
    model = UNETR(in_channels=1, out_channels=1, img_size=(96, 96, 96), feature_size=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load and preprocess MRI scan
def load_mri_scan(image_path):
    img = nib.load(image_path).get_fdata()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize
    transform = Resize((96, 96, 96))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dim
    img = transform(img).unsqueeze(0)  # Add batch dim
    return img.to(device)

# Run inference
def predict(model, image):
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).cpu().numpy().squeeze()
    return output

# Save report as PDF
def generate_pdf(input_image, output_mask, output_pdf_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_image[:, :, 48], cmap='gray')
    axs[0].set_title("MRI Scan")
    axs[1].imshow(output_mask[:, :, 48], cmap='jet', alpha=0.7)
    axs[1].set_title("Segmentation Output")
    plt.savefig("report_image.png")
    plt.close()
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "MRI Segmentation Report", ln=True, align='C')
    pdf.image("report_image.png", x=10, y=30, w=180)
    pdf.ln(120)
    pdf.multi_cell(0, 10, "This report contains the segmentation results from the UNETR model.")
    pdf.output(output_pdf_path)

# Main
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    image = load_mri_scan(INPUT_IMAGE_PATH)
    segmentation_output = predict(model, image)
    generate_pdf(image.cpu().numpy().squeeze(), segmentation_output, OUTPUT_PDF_PATH)
    print(f"Report saved at {OUTPUT_PDF_PATH}")
