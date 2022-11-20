import torch.nn as nn
import timm
import torch
from transformers import ElectraTokenizer, ElectraModel


class MMModel(nn.Module):
    def __init__(self, embed_dim, num_class, device='cuda', freeze_backbone=False):
        super().__init__()
        self.image_extractor = timm.models.resnet.resnet34(pretrained=True)
        self.image_extractor.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim // 4),
            nn.Linear(embed_dim // 4, num_class),
        )

        self.tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-small-discriminator")
        self.language_encoder = ElectraModel.from_pretrained(
            "google/electra-small-discriminator")
        self.device = device
        for param in self.language_encoder.parameters():
            param.requires_grad = False
        if freeze_backbone:
            for param in self.image_extractor.parameters():
                param.requires_grad = False

        self.to(self.device)

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, image, name, description):
        img_embedding = self.image_extractor(
            image)  # [B, C, 224, 224] -> [B, 512]
        name_embedding = self.language_encoder(
            **self.tokenizer(name, padding=True, return_tensors='pt'
                             ).to(self.device)).last_hidden_state.mean(dim=1)  # [B, text_len] -> [B, 256]
        description_embedding = self.language_encoder(
            **self.tokenizer(description, padding=True, return_tensors='pt'
                             ).to(self.device)).last_hidden_state.mean(dim=1)  # [B, text_len] -> [B, 256]

        # [B, 512], [B, 256], [B, 256] -> [B, 1024]
        embedding_concat = torch.cat(
            [img_embedding, name_embedding, description_embedding], dim=1)
        out = self.head(embedding_concat)  # [B, 1024] -> [B, num_classes]
        return out
