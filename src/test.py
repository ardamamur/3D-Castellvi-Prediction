import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from torchvision import transforms, datasets


class MNIST(Dataset):
    def __init__(self, train, **args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2), # so the input size is 32,32 and divideable by 8
            ]
        )

        self.ds = datasets.MNIST("./../notebook", download=True, train=train, transform=None)

        self.dataset_len = len(self.ds)
        assert self.dataset_len > 1, "MNIST: Dataset not found in provided filepath!"

    def __len__(self):
        # Important function for a dataset, so pytorch lightning knows how many samples there are
        return self.dataset_len

    def __getitem__(self, item):
        # Most important function: returns a dictionary of whatever the model needs
        # Typically "target" for the image, and "class" for some sort of label (GT)
        input, label = self.ds[item]
        print("label:", label)
        if self.transform is not None:
            input = self.transform(input)
        return {"target": input, "class": label}

    def __str__(self):
        return "mnist"



ds = MNIST(train=True)
train_loader = DataLoader(dataset=ds, batch_size=1, num_workers=1, drop_last=True, pin_memory=True, shuffle=True)
# Convert DataLoader to iterator
data_iter = iter(train_loader)

# Get the first batch
first_batch = next(data_iter)

# Access and print input and target
input = first_batch["target"]
label = first_batch["class"]

print("Input: ", input)
print("Label: ", label)
