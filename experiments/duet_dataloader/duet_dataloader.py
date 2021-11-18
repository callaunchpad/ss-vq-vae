import torch
import torchvision
from torch.utils import data
import numpy as np
from sklearn.preprocessing import StandardScaler

class DuetDataset(data.Dataset):

    def __init__(self):
        # style_output = np.loadtxt('/datasets/duet/genres/jay-ss-vq-vae/experiments/run-test/full/style_output', delimiter=",", dtype = np.float32)
        # genre_labels = np.loadtxt('/datasets/duet/genres/jay-ss-vq-vae/experiments/run-test/full/style_labels', delimiter=" ", dtype = np.int_)

        style_output = np.loadtxt('/datasets/duet/genres/jay-ss-vq-vae/experiments/run-test/full2/style_output', delimiter=",", dtype = np.float32)
        genre_labels = np.loadtxt('/datasets/duet/genres/jay-ss-vq-vae/experiments/run-test/full2/labels', delimiter=" ", dtype = np.int_)

        

        print("MEAN ======= ", np.mean(style_output))
        print("STD ======= ", np.std(style_output))
        style_output = StandardScaler().fit_transform(style_output)
        print('Normalizing dataset...')
        self.style_output = torch.from_numpy(style_output)
        
        self.genre_labels = torch.from_numpy(genre_labels)
        
        print("MEAN ======= ", np.mean(style_output))
        print("STD ======= ", np.std(style_output))

        print("SHAPE =", self.style_output.shape)

    def __len__(self):
        
        return self.style_output.shape[0]

    def __getitem__(self, idx):
        style = self.style_output[idx]
        genre = self.genre_labels[idx]
        return style, genre

    

loader = data.DataLoader(DuetDataset(), batch_size=1, shuffle=True)

print("Full dataset")
for batch in loader:
    print(batch)
print("\n")
