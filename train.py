import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from kornia.losses.focal import BinaryFocalLossWithLogits
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import ALLOWED_GENRES, MMIMDBDatasetNew
from model import MMModel
from sklearn.metrics import f1_score


def train(lr, batch_size, num_classes, epochs, loss_type='bce', alpha_focal=1, device='cuda', img_extractor='resnet34', inner_features_size=1024):
    assert loss_type in ['bce', 'focal']
    train_dataset = MMIMDBDatasetNew('mmimdb_parced/',
                                     split='train',
                                     img_transforms=A.Compose([
                                         A.HorizontalFlip(p=0.5),
                                         A.RandomBrightnessContrast(p=0.2),
                                         A.ColorJitter(),
                                         A.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225],
                                         ),
                                         ToTensorV2(),
                                     ]))
    val_dataset = MMIMDBDatasetNew('mmimdb_parced/', split='test',
                                   img_transforms=A.Compose([A.Normalize(
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225],
                                   ),
                                       ToTensorV2()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print_freq = len(train_loader) // 5
    model = MMModel(inner_features_size, num_classes, img_extractor=img_extractor,
                    device=device)
    # checkpoint = torch.load(f'models/model_epoch_19.pth')
    # model.load_state_dict(checkpoint)
    criterion = torch.nn.BCEWithLogitsLoss() if loss_type == 'bce' else BinaryFocalLossWithLogits(alpha_focal, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        for i, (images, names, descriptions, genres) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)

            genres = torch.stack(genres).to(device).T.to(torch.float)

            forward_time = time.time()
            prediction = model(images, names, descriptions)
            forward_time = time.time() - forward_time
            loss = criterion(prediction, genres)
            loss.backward()
            optimizer.step()
            if i % print_freq == 0:
                prediction_indices = torch.sigmoid(prediction) > 0.5
                acc = (prediction_indices == genres).sum() / \
                    (batch_size * num_classes)
                f1 = f1_score(prediction_indices.detach().cpu(),
                              genres.cpu(), average="samples")
                print(
                    f'Epoch {epoch} [{i} / {len(train_loader)}], loss {loss.item():0.6f}, acc {acc:0.4f}, f1 {f1:0.4f}, forward time {forward_time:0.4f}')
        os.makedirs('models', exist_ok=True)
        print(
            f'Epoch execution time: {time.time() - epoch_start:0.4f} seconds')
        torch.save(model.state_dict(), f'models/model_epoch_{epoch}.pth')
        model.eval()
        val_acc = 0
        val_f1 = 0
        print(f'Finished epoch {epoch}, evaluating model')
        for images, names, descriptions, genres in tqdm(val_loader, desc='evaluating'):

            images = images.to(device)
            prediction = model(images, names, descriptions)
            prediction_indices = torch.sigmoid(prediction) > 0.5
            genres = torch.stack(genres).to(device).T.to(torch.float)
            val_acc += (prediction_indices == genres).sum() / \
                (batch_size * num_classes)
            val_f1 += f1_score(prediction_indices.detach().cpu(),
                               genres.cpu(), average="samples")
        val_acc = val_acc / len(val_loader)
        val_f1 = val_f1 / len(val_loader)
        print('*' * 100)
        print(f'Epoch {epoch} | val acc {val_acc:0.4f}, val f1 {val_f1:0.4f}')
        print('*' * 100)


if __name__ == '__main__':
    fire.Fire(train)
