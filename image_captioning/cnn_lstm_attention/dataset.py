import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions


def get_loader(transform, root_dir:str, annFile_dir:str, batch_size:int=64, shuffle:bool=True, num_workers:int=6):
    """
    Create a DataLoader for COCO Captions using torchvision's built-in dataset.

    Args:
        transform ():
        root_dir (str): Path to the COCO images directory
        annFile_dir (str): Path to the annotations json file
        batch_size (int): Number of samples per batch, defaults to 64
        shuffle (bool): Whether to shuffle data in dataloader,defaults to True
        num_workers (int): Number of worker processes for data loading, defaults to 5

    Returns:

    """
    # default transformation
    if not transform:
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # dataset definition
    dataset = CocoCaptions(
        root=root_dir,
        annFile=annFile_dir,
        transform=transform,
    )

    # dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader