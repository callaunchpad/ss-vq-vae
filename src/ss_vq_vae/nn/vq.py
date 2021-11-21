# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import confugue
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


@confugue.configurable
class VQEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, use_codebook_loss=True, axis=-1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self._use_codebook_loss = use_codebook_loss
        self._cfg['init'].bind(nn.init.kaiming_uniform_)(self.embedding.weight)
        self._axis = axis

    def forward(self, input):
        if self._axis != -1:
            input = input.transpose(self._axis, -1)
        
        mu_i = input[:,:,:input.shape[2]//2]
        sigma_i = torch.exp(0.5 * input[:, :, input.shape[2]//2:])

        mu_e = self.embedding.weight[:,:self.embedding.weight.shape[1]//2]
        sigma_e = torch.exp(0.5 * self.embedding.weight[:, self.embedding.weight.shape[1]//2:])

        numerator = matrix_mse(mu_i, mu_e) + torch.sum(sigma_i**2, axis=-1).unsqueeze(-1)
        denominator = torch.sum(2*sigma_e**2, axis=-1).unsqueeze(0).unsqueeze(0)
        second_term = numerator/denominator

        sigma_e = sigma_e.unsqueeze(0).unsqueeze(0)
        sigma_i = sigma_i.unsqueeze(axis=-2).repeat(1, 1, self.num_embeddings, 1)
        log_term = torch.log(sigma_e/sigma_i).sum(axis=-1)

        distances = log_term + second_term
        

        # distances = (torch.sum(input ** 2, axis=-1, keepdim=True)
        #              - 2 * torch.matmul(input, self.embedding.weight.T)
        #              + torch.sum(self.embedding.weight ** 2, axis=-1))
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)
        print("Codebook shape: ", quantized.shape)
        
        losses = {
            'commitment': ((quantized.detach() - input) ** 2).mean(axis=-1)
        }
        if self._use_codebook_loss:
            losses['codebook'] = ((quantized - input.detach()) ** 2).mean(axis=-1)

            # Straight-through gradient estimator as in the VQ-VAE paper
            # No gradient for the codebook
            quantized = (quantized - input).detach() + input
        else:
            # Modified straight-through gradient estimator
            # The gradient of the result gets copied to both inputs (quantized and non-quantized)
            quantized = input + quantized - input.detach()

        if self._axis != -1:
            quantized = quantized.transpose(self._axis, -1).contiguous()

        ones = torch.ones([quantized.shape[0], quantized.shape[1], quantized.shape[2]//2]).to(device=quantized.device)
        z = torch.randn(quantized.shape[0], quantized.shape[1], quantized.shape[2]//2).to(device=quantized.device)

        sigmaZ = torch.exp(0.5 * quantized[:, :, quantized.shape[2]//2:]) * z
        quantized = quantized[:,:,:quantized.shape[2]//2] + sigmaZ
        return quantized, ids, losses

    def matrix_mse(m1, m2):
        return torch.sum(m1 ** 2, axis=-1, keepdim=True) - 2 * torch.matmul(m1, m2.T) + torch.sum(m2 ** 2, axis=-1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # randn_like as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample
