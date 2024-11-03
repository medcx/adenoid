# coding: utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
import torch
from load_model import model_selection
from tqdm import tqdm


def read_image_with_chinese_path(file_path):
    # 将路径转换为系统默认编码
    with open(file_path, 'rb') as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
    # 使用cv2.imdecode读取图片
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def save_image_with_chinese_path(file_path, image):
    cv2.imencode('.jpg', image)[1].tofile(file_path)  # 保存图片


def draw_CAM(model, img_path, out_path, data, type, transform=None, visual_heatmap=False):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param out_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0).to('cuda')

    # 获取模型输出的feature/score
    model.eval()
    output, features = model(img)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()

    pred_type = '正常' if pred == 0 else '异常'
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = read_image_with_chinese_path(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + 0.6 * img  # 这里的0.4是热力图强度因子
    true_false = type == pred_type
    save_path = os.path.join(out_path, f'{data}_{true_false}.jpg')
    save_image_with_chinese_path(save_path, superimposed_img)


if __name__ == '__main__':
    for typ in ['异常', '正常']:
        img_path = f'../data/sorted_img/{typ}/'
        for ID in tqdm(os.listdir(img_path)):
            for n in ['0', '1', '2']:
                if n == '0':
                    out_type = '正脸'
                elif n == '1':
                    out_type = '侧脸45'
                else:
                    out_type = '侧脸90'
                out_path = f'../data/heat_map/{typ}/{ID}/{out_type}'
                os.makedirs(out_path, exist_ok=True)
                model, image_size, *_ = model_selection('efficientnet-b4', num_out_classes=2)
                model = model.to('cuda')
                weight = torch.load(f'./five_model_{n}/fold0/best_model.pth', map_location='cuda:0')
                model.load_state_dict(weight)
                transform = transforms.Compose([
                    transforms.ToTensor(),  # 转换为Tensor
                    transforms.Resize((512, 512)),  # 调整大小
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
                ])
                for data in os.listdir(os.path.join(img_path, ID, out_type)):
                    data_path = os.path.join(img_path, ID, out_type, data)
                    draw_CAM(model, data_path, out_path, data.split('.')[0], typ, transform=transform, visual_heatmap=False)
