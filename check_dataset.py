import torch
from torch.utils.data import DataLoader
from agriculture_vision_dataset import AgricultureVisionDataset, get_train_transforms, get_val_transforms

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        return False, "NaN"
    if torch.isinf(tensor).any():
        return False, "Inf"
    if tensor.min() < 0:
        return False, "Negative"
    return True, None

def scan_loader_summary(loader, dataset_name):
    stats = {
        "nan_images": 0,
        "inf_images": 0,
        "neg_images": 0,
        "nonint_labels": 0,
        "out_of_range_labels": 0
    }

    for sample in loader:
        image = sample['image']
        label = sample['label']

        valid_image, err_type = check_tensor(image)
        if not valid_image:
            if err_type == "NaN":
                stats["nan_images"] += 1
            elif err_type == "Inf":
                stats["inf_images"] += 1
            elif err_type == "Negative":
                stats["neg_images"] += 1

        if not torch.all(label == label.int()):
            stats["nonint_labels"] += 1
        if label.max() >= loader.dataset.num_classes:
            stats["out_of_range_labels"] += 1

    print(f"\n{dataset_name} dataset check summary:")
    print(f"Total images checked: {len(loader.dataset)}")
    print(f"Images with NaN values: {stats['nan_images']}")
    print(f"Images with Inf values: {stats['inf_images']}")
    print(f"Images with negative values: {stats['neg_images']}")
    print(f"Labels with non-integer values: {stats['nonint_labels']}")
    print(f"Labels with out-of-range values: {stats['out_of_range_labels']}\n")

if __name__ == "__main__":
    DATA_ROOT = "dataset/finetuning_dataset/agriculture_vision_2019"

    # Use num_workers=0 to avoid multiprocessing issues during checks
    train_dataset = AgricultureVisionDataset(DATA_ROOT, split='train', transform=None)
    val_dataset = AgricultureVisionDataset(DATA_ROOT, split='val', transform=None)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    scan_loader_summary(train_loader, "Train")
    scan_loader_summary(val_loader, "Validation")
