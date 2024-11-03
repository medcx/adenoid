import numpy as np
from torchvision.transforms import transforms
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch import nn
import copy
import argparse
from dataload_multi import data_get, Patient_dataset
from utils import (
    round_filters,
    round_repeats,
    get_same_padding_conv2d,
    get_model_params,
    MemoryEfficientSwish,
    calculate_output_image_size
)
from sklearn.metrics import roc_curve, auc
import random
import os
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    def __init__(self, model_name, override_params):
        super().__init__()
        blocks_args, global_params = get_model_params(model_name, override_params)
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        image_size = calculate_output_image_size(image_size, 2)

        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs):
        # Pooling and final linear layer
        o = self._swish(self._bn1(self._conv_head(inputs)))
        p = o
        o = self._avg_pooling(o)
        if self._global_params.include_top:
            o = o.flatten(start_dim=1)
            o = self._dropout(o)
            o = self._fc(o)
        return o


class TransferModel(nn.Module):
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'efficientnet-b4':
            self.model = ExtractNet.from_pretrained('efficientnet-b4')
        # Replace fc
        num_ftrs = self.model._fc.in_features
        if not dropout:
            self.model._fc = nn.Linear(num_ftrs, num_out_classes)
        else:
            self.model._fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )

    def forward(self, x):
        x = self.model(x)
        return x


class CLASS_model(nn.Module):
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(CLASS_model, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'efficientnet-b4':
            over = {'num_classes': 1000}
            self.model = Classifier('efficientnet-b4', over)
        # Replace fc
        num_ftrs = self.model._fc.in_features
        if not dropout:
            self.model._fc = nn.Linear(num_ftrs, num_out_classes)
        else:
            self.model._fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None, transition=False, name='TransferModel'):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """

    if modelname == 'efficientnet-b4':
        if name == 'TransferModel':
            return TransferModel(modelchoice='efficientnet-b4', dropout=dropout,
                                 num_out_classes=num_out_classes), \
                224, True, ['image'], None
        elif name == 'decision':
            return CLASS_model(modelchoice='efficientnet-b4', dropout=dropout,
                               num_out_classes=num_out_classes), \
                224, True, ['image'], None
        else:
            return None
    else:
        raise NotImplementedError(modelname)


class ExtractNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        f = x
        x = self._swish(self._bn1(self._conv_head(x)))
        return x, f

    def forward(self, inputs):
        _, f = self.extract_features(inputs)
        return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        "--data_path",
        type=str,
        default='../data',
        help="data path",
    )
    args = parser.parse_args()
    epoch = args.epoch
    bs = args.batch_size
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"epoch:{args.epoch}")
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_path_0 = f'./results/model_0'
    model_path_1 = f'./results/model_1'
    model_path_2 = f'./results/model_2'

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

    model_0, image_size, *_ = model_selection('efficientnet-b4', num_out_classes=2)
    model_0 = model_0.to(device)

    model_1 = copy.deepcopy(model_0)
    model_1 = model_1.to(device)

    model_2 = copy.deepcopy(model_0)
    model_2 = model_2.to(device)
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(kf.split(img)):
        print(f"fold: {fold}")

        os.makedirs(f'./results/model_multi/fold{fold}', exist_ok=True)
        weight_0 = torch.load(f'{model_path_0}/fold{fold}/best_model.pth', map_location=device)
        weight_1 = torch.load(f'{model_path_1}/fold{fold}/best_model.pth', map_location=device)
        weight_2 = torch.load(f'{model_path_2}/fold{fold}/best_model.pth', map_location=device)
        model_0.load_state_dict(weight_0)
        model_0.eval()
        model_1.load_state_dict(weight_1)
        model_1.eval()
        model_2.load_state_dict(weight_2)
        model_2.eval()

        classifier, image_size, *_ = model_selection('efficientnet-b4', num_out_classes=2, name='decision')
        classifier = classifier.to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        print("use Adam")
        schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[75, 120])
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
            classifier.train()
            t = tqdm(train_loader)
            for x, y in t:
                x_0 = x['0'].float().to(device)
                x_1 = x['1'].float().to(device)
                x_2 = x['2'].float().to(device)
                label = y.to(device).requires_grad_()
                with torch.no_grad():
                    feature0 = model_0(x_0)
                    feature1 = model_1(x_1)
                    feature2 = model_2(x_2)
                feature0 = feature0.requires_grad_()
                feature1 = feature1.requires_grad_()
                feature2 = feature2.requires_grad_()
                features = torch.cat((feature0, feature1, feature2), dim=1)
                out = classifier(features)

                loss = criterion(out, label.long())
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
            classifier.eval()
            val_loss = 0
            for x, y in v:
                x_0 = x['0'].float().to(device)
                x_1 = x['1'].float().to(device)
                x_2 = x['2'].float().to(device)
                label = y.to(device)
                with torch.no_grad():
                    feature0 = model_0(x_0)
                    feature1 = model_1(x_1)
                    feature2 = model_2(x_2)
                    features = torch.cat((feature0, feature1, feature2), dim=1)
                    output = classifier(features).softmax(1)
                    val_loss += criterion(output, label.long())

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
                res.append(output[0][1].cpu())
                true_label.append(label.cpu())

            acc = correct / len(val_loader.dataset)
            print(f"acc: {acc}, correct:{correct} / all:{len(val_loader.dataset)}")
            print(f"val_loss: {val_loss / len(val_loader.dataset)}")
            fpr, tpr, thresholds = roc_curve(true_label, res)
            roc_auc = auc(fpr, tpr)
            # roc_auc = auc(fpr, tpr)
            print(f"curr_roc:{roc_auc}")
            if acc > best_acc:
                best_acc = acc
                # AUC = roc_auc
                torch.save(classifier.state_dict(), f'./results/model_multi/fold{fold}/best_model.pth')
                print('#############################')
                print(f"best_acc:{acc}")
