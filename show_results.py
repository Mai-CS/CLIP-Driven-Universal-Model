import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from PIL import Image
import cv2
import glob


# Function to load NIfTI image
def load_nifti_image(file_path, segmentation=False):
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()

    return img_data


# Calculate the middle slice index
def get_middle_slice(image_path):
    image = load_nifti_image(image_path)
    middle_slice = image.shape[2] // 2
    return middle_slice


def filter_masks(image_path, mask_path, save_path=None):
    # Load image and ground truth
    image = load_nifti_image(image_path)
    ground_truth = nib.load(mask_path)

    # Get the data from the NIfTI file
    data = ground_truth.get_fdata()

    # Get unique labels (masks) from the data (excluding background label 0)
    labels = np.unique(data)
    labels = labels[labels != 0]  # Exclude background (label 0)

    # Directory to save the separated masks
    output_dir = "separated_masks"
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each unique label and create a separate mask
    for label in labels:
        # Create a mask for the current label
        mask = (data == label).astype(np.uint8)
        # overlay this mask on the original image

        # Create a new NIfTI image for the mask
        mask_img = nib.Nifti1Image(
            mask, affine=ground_truth.affine, header=ground_truth.header
        )

        # Save the mask as a new NIfTI file
        mask_filename = os.path.join(output_dir, f"mask_label_{int(label)}.nii")
        nib.save(mask_img, mask_filename)
        print(f"Saved mask for label {int(label)} to {mask_filename}")

        # Plot the mask overlay on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image[:, :, image.shape[2] // 2], cmap="gray")
        plt.imshow(mask[:, :, image.shape[2] // 2], cmap="jet", alpha=0.5)
        plt.title(f"Overlay of Mask Label {int(label)} on Original Image")
        plt.axis("off")
        if save_path:
            overlay_save_path = os.path.join(
                save_path, f"overlay_label_{int(label)}.png"
            )
            plt.savefig(overlay_save_path, bbox_inches="tight", dpi=300)
            print(f"Saved overlay for label {int(label)} to {overlay_save_path}")
        plt.show()


# # Function to show slices of the image and multiple segmentations
# def show_nifti_image_with_segmentations(
#     image_path,
#     segmentation_paths,
#     ground_truth_path,
#     slice_num,
#     custom_titles=None,
#     save_path=None,
# ):
#     # Load image and ground truth
#     image = load_nifti_image(image_path)
#     ground_truth = load_nifti_image(ground_truth_path)

#     # Create subplots
#     num_segmentations = len(segmentation_paths)
#     cols_first_row = 2  # 2 images for the first row (Image + Ground Truth)
#     cols_other_rows = 5  # 5 images in each subsequent row
#     rows_other = (
#         num_segmentations + cols_other_rows - 1
#     ) // cols_other_rows  # Rows for segmentations

#     # Create a grid for the first row (2 columns) and the remaining segmentations (variable rows)
#     fig = plt.figure(figsize=(20, 5 * (1 + rows_other)))

#     # First row: 2 columns (Image + Ground Truth)
#     ax1 = plt.subplot2grid((1 + rows_other, cols_other_rows), (0, 0), colspan=1)
#     ax2 = plt.subplot2grid((1 + rows_other, cols_other_rows), (0, 1), colspan=1)

#     # Show original image slice in the first row, first column
#     ax1.imshow(image[:, :, slice_num], cmap="gray")
#     ax1.set_title("Image")
#     ax1.axis("off")  # Turn off axis for the original image

#     # Show ground truth slice in the first row, second column
#     ax2.imshow(image[:, :, slice_num], cmap="gray")
#     ax2.imshow(ground_truth[:, :, slice_num], cmap="jet", alpha=0.5)
#     ax2.set_title("Ground Truth")
#     ax2.axis("off")  # Turn off axis for the ground truth

#     # Show segmentations in the remaining rows
#     for i, seg_path in enumerate(segmentation_paths):
#         row = (i // cols_other_rows) + 1  # Row index starting from the second row
#         col = i % cols_other_rows  # Column index

#         ax = plt.subplot2grid((1 + rows_other, cols_other_rows), (row, col))

#         segmentation = load_nifti_image(seg_path)
#         segmentation_binary = segmentation > 0  # Assuming binary segmentation
#         ax.imshow(image[:, :, slice_num], cmap="gray")  # Grayscale image
#         ax.imshow(
#             segmentation_binary[:, :, slice_num], cmap="jet", alpha=0.5
#         )  # Overlay segmentation

#         # Use custom titles if provided
#         if custom_titles and i < len(custom_titles):
#             ax.set_title(custom_titles[i])
#         else:
#             ax.set_title(f"Segmentation {i + 1}")
#         ax.axis("off")  # Turn off axis for segmentations

#     plt.tight_layout()
#     plt.show()

#     if save_path:
#         plt.savefig(save_path)

colors = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#800000",
    "#008000",
    "#000080",
    "#808000",
    "#800080",
    "#008080",
    "#C0C0C0",
    "#FFA500",
    "#A52A2A",
    "#7FFF00",
    "#D2691E",
    "#FF7F50",
    "#6495ED",
    "#DC143C",
    "#00CED1",
    "#FF1493",
    "#00BFFF",
    "#696969",
    "#1E90FF",
    "#B22222",
    "#228B22",
    "#FF00FF",
    "#DDA0DD",
    "#B0E0E6",
    "#BC8F8F",
    "#FFD700",
    "#4B0082",
]

# Example usage:
# Define the base directory for images
base_dir = "/home/mai.kassem/sources/CLIP-Driven-Universal-Model/_samples/img"
gt_dir = "/home/mai.kassem/sources/CLIP-Driven-Universal-Model/_samples/label"

# TODO: uncomment this
# Get all image paths in the directory
image_paths = [
    os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".nii.gz")
]
image_paths = sorted(image_paths)
gt_paths = [
    os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".nii.gz")
]
gt_paths = sorted(gt_paths)

for i, image_path in enumerate(image_paths):
    image_name = os.path.basename(image_path).replace(".nii.gz", "")
    ground_truth_path = gt_paths[i]
    pred_dir = (
        "/home/mai.kassem/sources/CLIP-Driven-Universal-Model/_samples/_results/pred"
    )
    pred_dir = os.path.join(pred_dir, image_name)
    pred_paths = [
        os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".nii.gz")
    ]
    pred_paths = sorted(pred_paths)

    custom2_dir = (
        "/home/mai.kassem/sources/CLIP-Driven-Universal-Model/_samples/_results/custom2"
    )
    custom2_dir = os.path.join(custom2_dir, image_name)
    custom2_paths = [
        os.path.join(custom2_dir, f)
        for f in os.listdir(custom2_dir)
        if f.endswith(".nii.gz")
    ]
    custom2_paths = sorted(custom2_paths)

    # Step 1: Load the original CT image and masks (all NIfTI files)
    ct_image = load_nifti_image(image_path)
    gt_mask = load_nifti_image(ground_truth_path)

    # Step 3: Select a slice of the CT image and corresponding mask slices (assuming 3D volumes)
    slice_index = get_middle_slice(image_path)
    ct_slice = ct_image[:, :, slice_index]
    cmap = ListedColormap(colors)  # Use limited colors for masks
    gt_mask_slice = gt_mask[:, :, slice_index]
    pred_slices = []
    for pred_path in pred_paths:
        pred = load_nifti_image(pred_path)
        pred_slice = pred[:, :, slice_index]
        pred_slices.append(pred_slice)
    
    custom2_slices = []
    for custom2_path in custom2_paths:
        custom2 = load_nifti_image(custom2_path)
        custom2_slice = custom2[:, :, slice_index]
        custom2_slices.append(custom2_slice)
    # Add more mask slices if required

    # Step 4: Create a composite mask where each mask has a unique value
    composite_mask = np.zeros_like(pred_slices[0])
    for i, pred_slice in enumerate(pred_slices):
        pred_slice = np.where(pred_slice == 1, i, pred_slice)
        print(np.unique(pred_slice))
        composite_mask[pred_slice > 0] = i + 1
    print(np.unique(composite_mask))

    # Step 5: Plot the CT image in the first column and the overlay in the second column
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    # First column: Original CT image (no alpha, to avoid background color issues)
    axs[0].imshow(ct_slice, cmap="gray", interpolation="none")
    axs[0].set_title("Original CT Image")
    axs[0].axis("off")

    # Second column: Ground truth overlay (without red background)
    axs[1].imshow(ct_slice, cmap="gray", interpolation="none")  # CT image without alpha
    axs[1].imshow(
        gt_mask_slice, cmap=cmap, alpha=0.5
    )  # Ground truth mask with transparency
    axs[1].set_title("CT Image with Ground Truth Mask")
    axs[1].axis("off")

    # Third column: Predicted mask overlay (without red background)
    # Use ListedColormap for multiple predicted masks
    axs[2].imshow(ct_slice, cmap="gray", interpolation="none")  # CT image without alpha
    axs[2].imshow(
        composite_mask,
        cmap=cmap,
        alpha=0.5,
    )  # Composite mask overlay with transparency
    axs[2].set_title("CT Image with Predicted Mask Overlay")
    axs[2].axis("off")

    plt.show()

    # Optional: Save the figure with three columns as an image file
    fig.savefig(
        f"ct_image_with_gt_and_pred_overlay_{image_name}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
