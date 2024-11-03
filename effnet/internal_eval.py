import os

from torchvision.transforms import transforms
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import copy
from dataload_multi import data_get, Patient_dataset
from train_multi import model_selection
import random
import argparse


if __name__ == '__main__':

    # Internal dataset validation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='../data',
        help="data path",
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
        "--seed",
        type=int,
        default=42,
        help="Setting for the random seed",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default='all',
        help="Choose fold for evaluation",
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fold = args.fold
    data_path = args.data_path
    model_path_0 = f'./results/model_0'
    model_path_1 = f'./results/model_1'
    model_path_2 = f'./results/model_2'
    classifier_path = f'./results/model_multi'

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    size = args.size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_0, _, *_ = model_selection('efficientnet-b4', num_out_classes=2)
    model_0 = model_0.to(device)
    model_1 = copy.deepcopy(model_0)
    model_1 = model_1.to(device)
    model_2 = copy.deepcopy(model_0)
    model_2 = model_2.to(device)
    classifier, _, *_ = model_selection('efficientnet-b4', num_out_classes=2, name='decision')
    classifier = classifier.to(device)
    if fold == 'all':
        folds = [i for i in range(5)]
    else:
        folds = [int(fold)]

    num = 0
    res = []
    label = []

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    img, target = data_get(data_path)

    for fold, (train_indices, val_indices) in enumerate(kf.split(img)):
        img_val = [img[i] for i in val_indices]
        target_val = [target[i] for i in val_indices]
        val_dataset = Patient_dataset(img_val, target_val, transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

        print(f'predict {fold}')
        weight_0 = torch.load(f'{model_path_0}/fold{fold}/best_model.pth', map_location=device)
        weight_1 = torch.load(f'{model_path_1}/fold{fold}/best_model.pth', map_location=device)
        weight_2 = torch.load(f'{model_path_2}/fold{fold}/best_model.pth', map_location=device)
        weight_class = torch.load(f'{classifier_path}/fold{fold}/best_model.pth', map_location=device)
        model_0.load_state_dict(weight_0)
        model_0.eval()
        model_1.load_state_dict(weight_1)
        model_1.eval()
        model_2.load_state_dict(weight_2)
        model_2.eval()
        classifier.load_state_dict(weight_class)
        classifier.eval()

        v = tqdm(val_loader)
        for x, y in v:
            x_0 = x['0'].float().to(device)
            x_1 = x['1'].float().to(device)
            x_2 = x['2'].float().to(device)
            with torch.no_grad():
                feature0 = model_0(x_0)
                feature1 = model_1(x_1)
                feature2 = model_2(x_2)
                features = torch.cat((feature0, feature1, feature2), dim=1)
                output = classifier(features).softmax(1)
            res.append(float(output[0][1].cpu()))
            label.append(int(y.cpu()))

    import csv
    os.makedirs('./results', exist_ok=True)
    file_name = "./results/internal_result.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['pred', 'label'])

        for p, l in zip(res, label):
            writer.writerow([float(p), l])

        print(f"save dir: {file_name}")


