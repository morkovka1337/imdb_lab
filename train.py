import fire
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import ALLOWED_GENRES, MMIMDBDatasetNew
from model import MMModel
from sklearn.metrics import f1_score

def train(lr, batch_size, num_classes, epochs, freeze_backbone=False):
    print_freq = 10
    dataset = MMIMDBDatasetNew('mmimdb_parced/')
    train_len = int(len(dataset) * 0.7)
    train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MMModel(1024, num_classes, freeze_backbone=freeze_backbone).to(device)
    checkpoint = torch.load(f'model_epoch_1.pth')
    model.load_state_dict(checkpoint)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        if epoch > 0:
            for i, (images, names, descriptions, genres) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(device)

                genres = torch.stack(genres).to(device).T.to(torch.float)


                prediction = model(images, names, descriptions)
                loss = criterion(prediction, genres)
                loss.backward()
                optimizer.step()
                if i % print_freq == 0:
                    prediction_indices = torch.sigmoid(prediction) > 0.5
                    acc = (prediction_indices == genres).sum() / (batch_size * num_classes)
                    f1 = f1_score(prediction_indices.detach().cpu(), genres.cpu(), average="samples")
                    print(f'Epoch {epoch} [{i} / {len(train_loader)}], loss {loss.item():0.6f}, acc {acc:0.4f}, f1 {f1:0.4f}')

        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        model.eval()
        val_acc = 0
        val_f1 = 0
        print(f'Finished epoch {epoch}, evaluating model')
        for images, names, descriptions, genres in tqdm(val_loader, desc='evaluating'):

            images = images.to(device)
            prediction = model(images, names, descriptions)
            prediction_indices = torch.sigmoid(prediction) > 0.5
            genres = torch.stack(genres).to(device).T.to(torch.float)
            val_acc += (prediction_indices == genres).sum() / (batch_size * num_classes)
            val_f1 += f1_score(prediction_indices.detach().cpu(), genres.cpu(), average="samples")
        val_acc = val_acc / len(val_loader)
        val_f1 = val_f1 / len(val_loader)
        print(f'Epoch {epoch} | val acc {val_acc:0.4f}, val f1 {val_f1:0.4f}')

if __name__ == '__main__':
    fire.Fire(train)
