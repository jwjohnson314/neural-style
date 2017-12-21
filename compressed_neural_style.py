import argparse
import numpy as np
import os

from keras import backend as K
from keras.applications import VGG19 as vgg
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time


parser = argparse.ArgumentParser()

# images and directory options
parser.add_argument('--img_size', '-s', default=256, type=int)
parser.add_argument('--style_image', default='./images/starry_night.jpg')
parser.add_argument('--content_image', default='./images/ct.jpg')
parser.add_argument('--output_dir', default='./output/')

# model options
parser.add_argument('--num_iters', '-n', default=20, type=int)
parser.add_argument('--avg_pool', type=bool, default=True)

args = parser.parse_args()

total_variation_weight = 1e-4
style_weight = 1
content_weight = 0.025


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(args.img_size, args.img_size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gram(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))


def content_loss(content_tensor, generated_tensor):
    return K.sum(K.square(content_tensor - generated_tensor))


def style_loss(style_tensor, generated_tensor):
    S = gram(style_tensor)
    G = gram(generated_tensor)
    channels = 3
    size = args.img_size * args.img_size
    return K.sum(K.square(S - G)) / (4 * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :args.img_size - 1, :args.img_size - 1, :] - x[:, 1:, :args.img_size - 1, :])
    b = K.square(x[:, :args.img_size - 1, :args.img_size - 1, :] - x[:, :args.img_size - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


content_image = K.constant(preprocess_image(args.content_image))
style_image = K.constant(preprocess_image(args.style_image))
generated_image = K.placeholder((1, args.img_size, args.img_size, 3))

input_tensor = K.concatenate([content_image, style_image, generated_image], axis=0)

base_model = vgg(input_tensor=input_tensor, include_top=False, weights='imagenet')

if args.avg_pool:
    from keras.layers import AveragePooling2D

    print('replacing max pooling layers with average pooling layers...')
    for i, layer in enumerate(base_model.layers):
        if 'pool' in layer.name:
            base_model.layer = AveragePooling2D(pool_size=(2, 2),
                                                strides=(2, 2),
                                                padding='valid')

    base_model.compile(optimizer='adam', loss='categorical_crossentropy')


output_dict = dict([(layer.name, layer.output) for layer in base_model.layers])

style_layers = []
for i in range(1, 6, 1):
    style_layers.append('block'+str(i)+'_conv1')

content_layer = 'block5_conv2'

loss = K.variable(0.)
layer_features = output_dict[content_layer]
content_image_features = layer_features[0, :, :, :]
generated_image_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_image_features, generated_image_features)

for layer_name in style_layers:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    generated_image_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, generated_image_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(generated_image)


grads = K.gradients(loss, generated_image)[0]
fetch_loss_and_grads = K.function([generated_image], [loss, grads])


class Evaluator:

    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, args.img_size, args.img_size, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

iterations = args.num_iters

x = preprocess_image(args.content_image)
x = x.flatten()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

for i in range(iterations):
    print('start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((args.img_size, args.img_size, 3))
    img = deprocess_image(img)
    fname = 'output_at_iteration_%d.png' % i
    imsave(os.path.join(args.output_dir, fname), img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
