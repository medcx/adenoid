from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch
from tqdm import tqdm
import copy
from train_multi import model_selection
import time
import glob
import random
import argparse
import json
from sklearn.metrics import roc_curve, auc


if __name__ == '__main__':

    # External dataset validation

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

    # preparing for external dataset
    path_0 = '../data/normal/test/*'
    patient_normal = glob.glob(path_0)

    path_1 = f'../data/disease/test/*'
    patient_cancer = glob.glob(path_1)

    patient_all = patient_normal + patient_cancer
    target = [0.] * len(patient_normal) + [1.] * len(patient_cancer)

    if fold == 'all':
        folds = [i for i in range(5)]
    else:
        folds = [int(fold)]

    num = 0
    res = []
    label = []

    start_time = time.time()
    for fold in folds:
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
        t = tqdm(zip(patient_all, target), total=len(patient_all))
        for data, o in t:
            patient = data.replace('\\', '/')
            patient_id = patient.split('/')[-1]
            patient_type = patient.split('/')[-3].split('_')[0]
            p_0 = f'{args.data_path}/{patient_type}_0/test/{patient_id}/*'
            p_1 = f'{args.data_path}/{patient_type}_1/test/{patient_id}/*'
            p_2 = f'.{args.data_path}/{patient_type}_2/test/{patient_id}/*'
            data_0 = glob.glob(p_0)
            random.shuffle(data_0)
            if len(data_0) == 0:
                d_0 = np.zeros((512, 512, 3))
            else:
                d_0 = np.array(Image.open(data_0[0]))

            data_1 = glob.glob(p_1)
            random.shuffle(data_1)
            if len(data_1) == 0:
                d_1 = np.zeros((512, 512, 3))
            else:
                d_1 = np.array(Image.open(data_1[0]))

            data_2 = glob.glob(p_2)
            random.shuffle(data_2)
            if len(data_2) == 0:
                d_2 = np.zeros((512, 512, 3))
            else:
                d_2 = np.array(Image.open(data_2[0]))
            d_0 = transform(d_0).unsqueeze(0).float().to(device)
            d_1 = transform(d_1).unsqueeze(0).float().to(device)
            d_2 = transform(d_2).unsqueeze(0).float().to(device)
            with torch.no_grad():
                feature0 = model_0(d_0)
                feature1 = model_1(d_1)
                feature2 = model_2(d_2)
                features = torch.cat((feature0, feature1, feature2), dim=1)
                output = classifier(features).softmax(1)
            if output.max(1)[1] == o:
                num += 1

            res.append(output[0][1].cpu())
            label.append(o)
        end_time = time.time()
        print(end_time - start_time)

    import csv

    file_name = f"./results/external_result.csv"

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pred', 'label'])
        for p, l in zip(res, label):
            writer.writerow([float(p), l])

    print(f"save dir: {file_name}")

    correct = num / len(label)
    print(f"correct:{correct}, num:{num}")
    fpr, tpr, thresholds = roc_curve(label, res)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    splits = {
        'acc': correct,
        'auc': roc_auc
    }
    b = json.dumps(splits)
    f2 = open(f'results/external_metrics.json', 'w')
    f2.write(b)
    f2.close()

