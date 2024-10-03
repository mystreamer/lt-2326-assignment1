import torch
import torchvision
from PIL import Image

# Add some transformation logic
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
# preprocessed_image = preprocess(image)

class ThaiOCRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, onehotencoder):
        self.dataframe = dataframe
        self.onehotencoder = onehotencoder

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
