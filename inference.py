from __future__ import print_function, division, absolute_import

import argparse

import pretrainedmodel.utils as utils
import torch
from pretrainedmodel.resnext101_32x4d import resnext101_32x4d

parser = argparse.ArgumentParser(description='Resnext101_32x4d')
parser.add_argument('--path_img', type=str, default='data/cat.jpg')

arch = 'Resnext101_32x4d'


def run_classifier(model, path_img):
    model.eval()
    # Load and Transform one input image
    load_img = utils.LoadImage()
    tf_img = utils.TransformImage(model)

    input_data = load_img(path_img)  # 3x400x225
    input_data = tf_img(input_data)  # 3x299x299
    input_data = input_data.unsqueeze(0)  # 1x3x299x299
    input = torch.autograd.Variable(input_data)

    # Load Imagenet Synsets
    with open('data/imagenet_synsets.txt', 'r') as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}

    with open('data/imagenet_classes.txt', 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Make predictions
    output = model(input)  # size(1, 1000)
    max, argmax = output.data.squeeze().max(0)
    class_id = argmax.item()
    class_key = class_id_to_key[class_id]
    classname = key_to_classname[class_key]
    return max, classname, class_key


def main():
    global args
    args = parser.parse_args()
    model = resnext101_32x4d(num_classes=1000,
                             pretrained='imagenet')

    path_img = args.path_img

    max, classname, class_key = run_classifier(model, path_img)

    print("'{}': '{}' is a '{}' | Confidence: {}".format(arch, path_img, classname, round(max.item() * 100, 3)))


if __name__ == '__main__':
    main()
