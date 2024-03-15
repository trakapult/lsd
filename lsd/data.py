import os
from torch.utils.data import Dataset
from PIL import Image

class CLEVRTEXDataset(Dataset):
    def __init__(self, path, tfm):
        super(CLEVRTEXDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")])
        self.transform = tfm
        print(self.files)
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        if self.transform is not None:
            im = self.transform(im)
        return im