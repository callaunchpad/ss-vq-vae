import torch
import torchvision
from torch.utils import data
import numpy as np
from sklearn.preprocessing import StandardScaler

class RNNDataset(data.Dataset):

    def __init__(self):

        codebook_ids = np.loadtxt('/datasets/duet/rnn/ss-vq-vae/experiments/model/encoded_content.txt', delimiter=" ", dtype = np.int_)

        self.codebook_ids = torch.from_numpy(codebook_ids)
                
        print("SHAPE =", self.codebook_ids.shape)

    def __len__(self):
        
        return self.codebook_ids.shape[0]

    def __getitem__(self, idx):
        id = self.codebook_ids[idx]
        return id

    

loader = data.DataLoader(RNNDataset(), batch_size=4, shuffle=True)

print("Full dataset")
for batch in loader:
    print(batch)
print("\n")
