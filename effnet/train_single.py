import os
import argparse
from freezen import freeze
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import random
from dataload_multi import data_get, Patient_dataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from torchvision.transforms import transforms


class My_dataset(Dataset):
    def __init__(self, data_path, target, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.target = np.array(target)

    def __getitem__(self, item):
        if self.transform:
            return self.transform(np.array(Image.open(self.data_path[item]))), torch.tensor(self.target[item])
        return torch.tensor(np.array(Image.open(self.data_path[item]))), torch.tensor(self.target[item])

    def __len__(self):
        return len(self.data_path)


class TransferModel(nn.Module):
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0, path=None):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'efficientnet-b4':
            self.model = EfficientNet.from_pretrained('efficientnet-b4')
        # Replace fc
        num_ftrs = self.model._fc.in_features
        if not dropout:
            self.model._fc = nn.Linear(num_ftrs, num_out_classes)
        else:
            self.model._fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )
        l = torch.load(path)
        self.model.load_state_dict(l, strict=False)

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None, path=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """

    if modelname == 'efficientnet-b4':
        return TransferModel(modelchoice='efficientnet-b4', dropout=dropout,
                             num_out_classes=num_out_classes, path=path), \
            224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='../data',
        help="data path",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="Total epoch",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of the input images",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Selection of GPU ID",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the training and validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Setting for the random seed",
    )
    parser.add_argument(
        "--degree",
        type=str,
        default='0',
        help="Choose degree",
    )
    args = parser.parse_args()
    degree = args.degree
    epoch = args.epoch
    bs = args.batch_size
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"epoch:{args.epoch}")
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # preparing for training
    criterion = nn.CrossEntropyLoss()
    size = args.size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"size: {size}")
    img, target = data_get(args.data_path)

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(kf.split(img)):
        os.makedirs(f'./results/model_{degree}/fold{fold}', exist_ok=True)

        print(f"Fold {fold}/{num_folds}")
        model, image_size, *_ = model_selection('efficientnet-b4', num_out_classes=2,
                                                path = './pretrain_weight/pretrained.pth')

        model = model.to(device)
        model = freeze(model, device)
        print("model has frozen")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        print("use Adam")
        schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[75, 90])
        img_train = [img[i] for i in train_indices]
        img_val = [img[i] for i in val_indices]

        target_train = [target[i] for i in train_indices]
        target_val = [target[i] for i in val_indices]

        train_dataset = Patient_dataset(img_train, target_train, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

        val_dataset = Patient_dataset(img_val, target_val, transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

        # training
        best_acc = 0
        TPA = 0
        for i in range(epoch):
            model.train()
            t = tqdm(train_loader)
            for x, y in t:
                x = x[degree].float().to(device)
                y = y.to(device).requires_grad_()
                outputs = model(x)
                loss = criterion(outputs, y.long())

                t.set_description(f"Epoch {i}")
                t.set_postfix(loss=loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schduler.step()
            # validation
            correct = 0
            res = []
            true_label = []
            v = tqdm(val_loader)
            model.eval()
            val_loss = 0
            for x, y in v:
                x = x[degree].float().to(device)
                y = y.to(device)
                with torch.no_grad():
                    output = model(x).softmax(1)
                    val_loss += criterion(output, y.long())
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                res.append(output[0][1].cpu())
                true_label.append(y.cpu())

            acc = correct / len(val_loader.dataset)
            print(f"acc: {acc}, correct:{correct} / all:{len(val_loader.dataset)}")
            print(f"val_loss: {val_loss / len(val_loader.dataset)}")
            fpr, tpr, thresholds = roc_curve(true_label, res)
            roc_auc = auc(fpr, tpr)
            print(f"curr_roc:{roc_auc}")
            if acc > best_acc:
                best_acc = acc
                # AUC = roc_auc
                torch.save(model.state_dict(), f'./results/model_{degree}/fold{fold}/best_model.pth')
                print('#############################')
                print(f"best_acc:{acc}")
