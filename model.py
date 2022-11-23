import torch.nn as nn
import timm
import torch
from transformers import ElectraTokenizer, ElectraModel

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_class):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.BatchNorm1d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, num_class),
        )

    def forward(self, x):
        return self.head(x)

class ImageExtractor(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.extractor = getattr(timm.models.resnet, arch)(pretrained=True)
        self.extractor.fc = nn.Identity()

    def forward(self, x):
        return self.extractor(x)

class TextExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-small-discriminator")
        self.language_encoder = ElectraModel.from_pretrained(
            "google/electra-small-discriminator")
        for param in self.language_encoder.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, text):
        embedding = self.language_encoder(
            **self.tokenizer(text, padding=True, return_tensors='pt'
                             ).to(self.device)).last_hidden_state.mean(dim=1)  # [B, text_len] -> [B, 256]
        return embedding

class MMModel(nn.Module):
    def __init__(self, embed_dim, num_class, img_extractor='resnet34', device='cuda'):
        super().__init__()
        self.image_extractor = ImageExtractor(img_extractor)
        self.text_extractor = TextExtractor(device)
        self.head = ClassificationHead(embed_dim, num_class)
        self.device = device
        self.to(self.device)

    def forward(self, image, name, description):
        img_embedding = self.image_extractor(image)  # [B, C, 224, 224] -> [B, 512]
        name_embedding = self.text_extractor(name)
        description_embedding = self.text_extractor(description)

        # [B, 512], [B, 256], [B, 256] -> [B, 1024]
        embedding_concat = torch.cat(
            [img_embedding, name_embedding, description_embedding], dim=1)
        out = self.head(embedding_concat)  # [B, 1024] -> [B, num_classes]
        return out
