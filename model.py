import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import matplotlib

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() == True \
    else torch.device('cpu')
model.to(device)
import wget


def plot_preds(numpy_img, preds):
    boxes = preds['boxes'].cpu().detach().numpy()
    for box in boxes:
        numpy_img = cv2.rectangle(
            np.array(numpy_img),
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            255, 3)
    return numpy_img


def look_image(name, model=model, threshold=0.75):
    # Перевод файла в формат для модели
    img_path = './app/static/' + name
    img_numpy = cv2.imread(img_path)[:, :, ::-1]
    img = torch.from_numpy(img_numpy.astype('float32')).permute(2, 0, 1)
    img = img / 255.
    # Получение предсказания
    model = model.eval()
    with torch.no_grad():
        predictions = model(img[None, ...].to(device))
    # Выбор боксов
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > threshold]
    boxes_dict = {'boxes': boxes}
    # Вывод предсказаний
    img_with_boxes = plot_preds(img_numpy, boxes_dict)
    # Сборка нового имени для картинки с боксами
    split = name.split('.')
    split_start = ''
    for i in range(len(split) - 1):
        split_start = split_start + split[i] + '.'
    boxed = split_start + 'boxed.' + split[-1]
    matplotlib.image.imsave('./app/static/'+boxed, img_with_boxes)
    return boxed
