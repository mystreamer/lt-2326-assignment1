import torch
import torchvision
from PIL import Image
from .TrainingConfig import TrainingConfig

# Add some transformation logic
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(TrainingConfig.image_resize),
])

class ThaiOCRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, onehotencoder):
        self.dataframe = dataframe
        self.onehotencoder = onehotencoder
        self.img_resize = TrainingConfig.image_resize

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            torchvision.transforms.functional.to_tensor(
                preprocess(Image.open(row["image_path"]).convert("RGB"))
                ),
                self.onehotencoder.transform([[row["label"]]]).toarray()[0],
        )
