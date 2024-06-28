from torch.utils.data import Dataset
from PIL import Image

# Load dataset
class LargeDataset(Dataset):
    def __init__(self, img_path_df: 'DataFrame', transform: 'function'=None):
        """
        Args:
            img_path_df (DataFrame): Dataframe with two columns: one with path to an image and onther with the corresponding class.
            transform (callable, optional): Optional transform to be applied on a sample when loaded.
        """
        self.img_data = img_path_df
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.img_data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label