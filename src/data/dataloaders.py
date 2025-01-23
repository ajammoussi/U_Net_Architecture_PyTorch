import os
from torch.utils.data import DataLoader
from src.data.dataset import OxfordIIITPetsAugmented
from src.utils.utils import transform_dict

def create_train_dataloader(data_dir, batch_size_train=64):
    pets_path_train = os.path.join(data_dir, 'OxfordPets', 'train')
    download = not os.path.exists(pets_path_train)

    pets_train = OxfordIIITPetsAugmented(
        root=pets_path_train,
        split="trainval",
        target_types="segmentation",
        download=download,
        **transform_dict,
    )

    pets_train_loader = DataLoader(
        pets_train,
        batch_size=batch_size_train,
        shuffle=True,
    )

    return pets_train_loader

def create_test_dataloader(data_dir, batch_size_test=21):
    pets_path_test = os.path.join(data_dir, 'OxfordPets', 'test')
    download = not os.path.exists(pets_path_test)

    print(pets_path_test)

    pets_test = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        target_types="segmentation",
        download=download,
        **transform_dict,
    )

    pets_test_loader = DataLoader(
        pets_test,
        batch_size=batch_size_test,
        shuffle=True,
    )

    return pets_test_loader