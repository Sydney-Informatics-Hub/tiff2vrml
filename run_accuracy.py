import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def binarize_image(image, invert=False):
    if invert:
        _, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY_INV)
    else:
        _, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)
    return binary_image

def parse_prediction_number(file_name):
    # Extract the prediction number from the file name
    return int(file_name.split("predict")[1].split(".")[0])

def calculate_iou(gt_binary, pred_binary):
    intersection = np.logical_and(gt_binary, pred_binary)
    union = np.logical_or(gt_binary, pred_binary)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def plot_diagnostic(gt_binary, pred_binary, iou_score):
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    print(np.min(gt_binary), np.max(gt_binary), np.mean(gt_binary), np.median(gt_binary))
    gt_binary[gt_binary<1] = 0
    gt_binary[gt_binary>1] = 1
    gt_binary = np.ma.masked_where(gt_binary < 0.5, gt_binary)
    ll1=ax.imshow(gt_binary, cmap='winter', alpha=1, vmin=0, vmax=1)

    # Plot Prediction in red with opacity
    print(np.min(pred_binary), np.max(pred_binary), np.mean(pred_binary), np.median(pred_binary))
    pred_binary = np.ma.masked_where(pred_binary > 183, pred_binary)
    pred_binary[pred_binary==1] = 1
    pred_binary[pred_binary<1] = 0
    # pred_binary = np.ma.masked_where(pred_binary < 0.5, pred_binary)
    ll2=ax.imshow(pred_binary, cmap='cool', alpha=1,vmin=0, vmax=1)
    

    # Plot Overlap in green with opacity
    overlap = np.logical_and(gt_binary, pred_binary)
    print(np.min(overlap), np.max(overlap), np.mean(overlap), np.median(overlap))
    overlap = np.ma.masked_where(overlap < 0.5, overlap)
    # overlap[overlap<1] = 0
    # overlap[overlap>1] = 1
    ll3=ax.imshow(overlap, cmap='autumn', alpha=0.5, vmin=0, vmax=1)
    #plt.colorbar(ll3)
    
    # Create legend handles and labels
    legend_handles = [
        mpatches.Patch(color='cyan', label='Ground Truth'),
        mpatches.Patch(color='magenta', label='Predictions'),
        mpatches.Patch(color='orange', label='Overlap')
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc='lower right')

    l3=ax.set_title('Overlap (IoU={:.2f})'.format(iou_score))
    plt.show()
    
def plot_diagnostic_sep(gt_binary, pred_binary, iou_score):
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Ground Truth in blue
    axes[0].imshow(gt_binary, cmap='Blues', interpolation='none')
    axes[0].set_title('Ground Truth')

    # Prediction in red
    axes[1].imshow(pred_binary, cmap='Reds', interpolation='none')
    axes[1].set_title('Prediction')

    # Overlap in green
    overlap = np.logical_and(gt_binary, pred_binary)
    axes[2].imshow(overlap, cmap='Greens', interpolation='none')
    axes[2].set_title('Overlap (IoU={:.2f})'.format(iou_score))

    plt.show()

def calculate_iou_for_folder(gt_folder, pred_folder):
    iou_scores = []

    # Create a dictionary to map prediction numbers to file paths
    pred_dict = {}
    for file_name in os.listdir(pred_folder):
        if file_name.endswith(".tif") and "im0400." in file_name:
            pred_number = parse_prediction_number(file_name)
            pred_path = os.path.join(pred_folder, file_name)
            pred_dict[pred_number] = pred_path

    for file_name in os.listdir(gt_folder):
        if file_name.endswith(".tif"):
            gt_path = os.path.join(gt_folder, file_name)
            pred_number = int(file_name.split(".")[0].replace("seg", ""))-400

            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred_path = pred_dict.get(pred_number)

            if pred_path is not None:
                pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

                gt_binary = binarize_image(gt_image)
                pred_binary = binarize_image(pred_image, invert=True)
                # plot_diagnostic(gt_image, pred_image, 0)
                iou_score = calculate_iou(gt_binary, pred_binary)
                print(iou_score, gt_path,  pred_path)
                iou_scores.append(iou_score)
                # plot_diagnostic(gt_binary, pred_binary, iou_score)
                print(pred_number)
                if pred_number == 99:
                    plot_diagnostic(gt_image, pred_image, iou_score)

    average_iou = np.mean(iou_scores)
    return average_iou
# Example usage
gt_folder = "../MITO-EM/EM30-R-mito-train-val-v2/mito-val-v2"
pred_folder = "../MITO-EM/OUTPUT/EM30-R-mito-train-val-v2/PREDICTIONS_val"

iou_score = calculate_iou_for_folder(gt_folder, pred_folder)
print("Average IoU Score:", iou_score)