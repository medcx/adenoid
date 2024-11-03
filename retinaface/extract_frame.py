import os
from os.path import join
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode

np.random.seed(0)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)
    return model


def detect(img_list, output_path, resize=1):
    os.makedirs(output_path, exist_ok=True)

    im_height, im_width, _ = img_list[0].shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img_x = torch.stack(img_list, dim=0).permute([0, 3, 1, 2])
    scale = scale.to(device)

    # batch size
    batch_size = args.bs
    # forward times
    f_times = img_x.shape[0] // batch_size
    if img_x.shape[0] % batch_size != 0:
        f_times += 1
    locs_list = list()
    confs_list = list()
    for _ in range(f_times):
        if _ != f_times - 1:
            batch_img_x = img_x[_ * batch_size:(_ + 1) * batch_size]
        else:
            batch_img_x = img_x[_ * batch_size:]  # last batch
        batch_img_x = batch_img_x.to(device).float()
        l, c, _ = net(batch_img_x)
        locs_list.append(l)
        confs_list.append(c)
    locs = torch.cat(locs_list, dim=0)
    confs = torch.cat(confs_list, dim=0)
    del locs_list
    del confs_list
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    img_cpu = img_x.permute([0, 2, 3, 1]).cpu().numpy()
    i = 0
    for img, loc, conf in zip(img_cpu, locs, confs):
        boxes = decode(loc.data, prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        if len(dets) == 0:
            continue
        det = list(map(int, dets[0]))
        x, y, size_bb_x, size_bb_y = get_boundingbox(det, img.shape[1], img.shape[0])
        cropped_img = img[y:y + size_bb_y, x:x + size_bb_x, :] + (104, 117, 123)
        cv2.imwrite(join(output_path, '{:04d}.png'.format(i)), cropped_img)
        i += 1
    pass


def detect2(img_list, output_path, images, resize=1):
    os.makedirs(output_path, exist_ok=True)
    im_height, im_width = img_list.shape[0], img_list.shape[1]

    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img_list = img_list[None]
    img_x = img_list.permute([0, 3, 1, 2])
    scale = scale.to(device)

    # forward times
    locs_list = list()
    confs_list = list()

    img_x = img_x.to(device).float()
    l, c, _ = net(img_x)

    locs_list.append(l)
    confs_list.append(c)

    locs = torch.cat(locs_list, dim=0)
    confs = torch.cat(confs_list, dim=0)

    del locs_list
    del confs_list

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    img_cpu = img_x.permute([0, 2, 3, 1]).cpu().numpy()

    i = 0
    for img, loc, conf in zip(img_cpu, locs, confs):

        boxes = decode(loc.data, prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        if len(dets) == 0:
            continue
        det = list(map(int, dets[0]))
        x, y, size_bb_x, size_bb_y = get_boundingbox(det, img.shape[1], img.shape[0])
        cropped_img = img[y:y + size_bb_y, x:x + size_bb_x, :] + (104, 117, 123)
        # cropped_img = cv2.resize(cropped_img, ori_size)
        cv2.imwrite(join(output_path, images), cropped_img)
        i += 1
    pass


def extract_frames(data_path, interval=1):
    """Method to extract frames"""
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    frames = list()
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image) - torch.tensor([104, 117, 123])
        if frame_num % interval == 0:
            frames.append(image)
        frame_num += 1
    reader.release()
    if len(frames) > args.max_frames:
        samples = np.random.choice(
            np.arange(0, len(frames)), size=args.max_frames, replace=False)
        return [frames[_] for _ in samples]
    return frames


def get_boundingbox(bbox, width, height, scale=1.3, minsize=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    size_bb_x = int((x2 - x1) * scale)
    size_bb_y = int((y2 - y1) * scale)
    if minsize:
        if size_bb_x < minsize:
            size_bb_x = minsize
        if size_bb_y < minsize:
            size_bb_y = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb_x // 2), 0)
    y1 = max(int(center_y - size_bb_y // 2), 0)
    # Check for too big bb size for given x, y
    size_bb_x = min(width - x1, size_bb_x)
    size_bb_y = min(height - y1, size_bb_y)
    return x1, y1, size_bb_x, size_bb_y


def extract_method_images(in_path, out_path):
    input_img_path = in_path
    out_img_path = out_path
    for i in tqdm(os.listdir(input_img_path)):
        for images in os.listdir(join(input_img_path, i)):
            print(join(input_img_path, i))
            num_unqualified = 0
            x = join(input_img_path, i, images)
            in_img = cv2.imread(x)
            # ori_size = in_img.shape[:2]
            scale_factor = 1/3
            new_size = (int(in_img.shape[1] * scale_factor), int(in_img.shape[0] * scale_factor))
            #
            # new_size = (350, 450)
            in_img = cv2.resize(in_img, new_size)
            cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
            in_img = torch.tensor(in_img) - torch.tensor([104, 117, 123])

            detect2(in_img, join(out_img_path, i), images)

    print("Total unqualified: ", num_unqualified)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # p.add_argument('--data_path', '-p', type=str, help='path to the data')
    p.add_argument('--confidence_threshold', default=0.5,
                   type=float, help='confidence threshold')
    p.add_argument('--top_k', default=5, type=int, help='top_k')
    p.add_argument('--nms_threshold', default=0.5,
                   type=float, help='nms threshold')
    p.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
    p.add_argument('--bs', default=1, type=int, help='batch size')
    p.add_argument('--frame_interval', '-fi', default=1, type=int, help='frame interval')
    p.add_argument('--device', "-d", default="cuda:0", type=str, help='device')
    p.add_argument('--max_frames', default=2000, type=int, help='maximum frames per video')
    p.add_argument('--input_dir', default='../data/', type=str, help='image path')
    p.add_argument('--output_dir', default='../data/', type=str, help='output path')
    args = p.parse_args()

    torch.set_grad_enabled(False)
    # use resnet-50
    cfg = cfg_re50

    in_path = args.input_dir
    out_path = args.output_dir

    pretrained_weights = './weights/Resnet50_Final.pth'

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(device)

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, pretrained_weights, args.device)
    net.eval()
    print('Finished loading model!')

    import os


    extract_method_images(in_path, out_path)
