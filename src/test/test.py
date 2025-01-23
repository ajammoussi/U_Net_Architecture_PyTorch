import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from src.data.dataloaders import create_test_dataloader
from src.utils.utils import create_results_dir


def calculate_metrics(preds, targets, threshold=0.5):
    # Convert predictions to binary masks using a threshold
    preds = (preds > threshold).float()  # Apply threshold to get binary masks
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    # Flatten the arrays
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    # Calculate metrics
    precision = precision_score(targets_flat, preds_flat)
    recall = recall_score(targets_flat, preds_flat)
    f1 = f1_score(targets_flat, preds_flat)
    iou = jaccard_score(targets_flat, preds_flat)

    return precision, recall, f1, iou

def generate_predictions(model, test_loader, device, num_samples, results_dir):
    model.eval()
    create_results_dir(results_dir)

    all_precision = []
    all_recall = []
    all_f1 = []
    all_iou = []

    with torch.no_grad():
        for i, (input_img, target_img) in enumerate(test_loader):
            print(f"Processing batch {i + 1}")

            # Move data to device
            input_img = input_img.to(device)
            target_img = (target_img > 1.0).float() # Ensure targets are binary
            output = model(input_img)
            output = output.cpu()

            # Calculate metrics
            precision, recall, f1, iou = calculate_metrics(output, target_img)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_iou.append(iou)

            # Iterate over each sample in the batch
            for j in range(input_img.size(0)):
                if j >= num_samples:
                    break

                # Remove batch dimension and convert tensors to PIL images
                input_img_pil = transforms.ToPILImage()(input_img[j].cpu())  # Select j-th sample
                target_img_pil = transforms.ToPILImage()(target_img[j].cpu())  # Select j-th sample
                output_pil = transforms.ToPILImage()(output[j])  # Select j-th sample

                # Save images
                input_img_path = os.path.join(results_dir, f'input_{i * test_loader.batch_size + j}.png')
                target_img_path = os.path.join(results_dir, f'target_{i * test_loader.batch_size + j}.png')
                output_img_path = os.path.join(results_dir, f'output_{i * test_loader.batch_size + j}.png')

                try:
                    input_img_pil.save(input_img_path)
                    target_img_pil.save(target_img_path)
                    output_pil.save(output_img_path)
                    print(f"Images saved successfully: {input_img_path}, {target_img_path}, {output_img_path}")

                except Exception as e:
                    print(f"Error saving images: {e}")

                # Display images using matplotlib
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(input_img_pil)
                axes[0].set_title('Input Image')
                axes[0].axis('off')

                axes[1].imshow(target_img_pil, cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')

                axes[2].imshow(output_pil, cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')

                plt.show()

            # Break after processing the first batch
            print(f"Processed batch {i + 1}")
            break

    # Calculate average metrics
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_iou = np.mean(all_iou)

    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}")
    print(f"Average IoU: {avg_iou}")

def launch_test(device, model, config):
    # Define the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    working_dir = os.path.join(project_root, config['test']['working_dir'])
    results_dir = os.path.join(project_root, config['test']['results_dir'])
    test_batch_size = config['test']['test_batch_size']
    num_samples = config['test']['num_samples']

    print(f"Absolute path of working_dir: {working_dir}")
    print(f"Absolute path of results_dir: {results_dir}")

    val_dataset = create_test_dataloader(working_dir, test_batch_size)
    print(f"Number of batches in test_loader: {len(val_dataset)}")

    generate_predictions(model, val_dataset, device, num_samples, results_dir)
