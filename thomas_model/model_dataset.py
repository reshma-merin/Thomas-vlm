import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PretrainImageDataset(Dataset):
    def __init__(self, root_dirs, img_size=512):

        self.paths = []
        for d in root_dirs:
            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(d, fname))

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root_dirs}")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),  # scales to [0, 1]
            # Sentinel-2 RGB PNGs were already normalized/clipped on export (min=0, max=0.5
            # in the Earth Engine getThumbURL call), so a simple [0,1] scaling is a reasonable
            # starting point. If reconstructions look poor, revisit normalization stats.
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Skip corrupted/missing files without crashing the whole run.
            return self.__getitem__((idx + 1) % len(self))
        return self.transform(img)