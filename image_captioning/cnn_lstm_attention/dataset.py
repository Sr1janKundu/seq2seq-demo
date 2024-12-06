import torch, json, os
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from collections import Counter


# def get_loader(transform, root_dir:str, annFile_dir:str, batch_size:int=64, shuffle:bool=True, num_workers:int=6):
#     """
#     Create a DataLoader for COCO Captions using torchvision's built-in dataset.
#
#     Args:
#         transform ():
#         root_dir (str): Path to the COCO images directory
#         annFile_dir (str): Path to the annotations json file
#         batch_size (int): Number of samples per batch, defaults to 64
#         shuffle (bool): Whether to shuffle data in dataloader,defaults to True
#         num_workers (int): Number of worker processes for data loading, defaults to 5
#
#     Returns:
#
#     """
#     # default transformation
#     if not transform:
#         transform = v2.Compose([
#             v2.Resize((224, 224)),
#             v2.ToImage(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
#
#     # dataset definition
#     dataset = CocoCaptions(
#         root=root_dir,
#         annFile=annFile_dir,
#         transform=transform,
#     )
#
#     # dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     return dataloader

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '\\{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '\\{}_captions.json'.format(split_type), 'r'))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.split_type == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index])

        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)