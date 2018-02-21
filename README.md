# Neural-Style

This is a Keras/Tensorflow implementation of the neural-style algorithm from the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Gatys *et. al.* Most of the code here isn't original; it largely follows Francois Chollet's implementation as laid out in his book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff), and in the notebooks that accompany the book [here](https://github.com/fchollet/deep-learning-with-python-notebooks). I wanted a Python-based executable for quick experimentation along the lines of [this Torch implementation](https://github.com/jcjohnson/neural-style); this repo is the result.

## What It Does
The algorithm combines the content of an image with the styles and textures of another image using features derived from a convolutional neural network. For example:
<div align="center">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/cubist_9.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/chopin.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/cubist_chopin.png" height="223">
</div>

<br>

<div align="center">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/il_peccato.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/brooklyn_bridge.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/il_brooklyn.png" height="223">
</div>

## Implementation Differences

* In the paper by Gatys *et. al.*, they note the use of a vgg network with normalized weights. Here, we use non-normalized weights. 
* The Chollet implementation uses max-pooling layers. Here, max-pooling layers are converted by default to average-pooling layers as in the original implementation. 

## Basic Usage

```
python neural_style.py --style_image=<style_image.jpg> --content_image=<content_image.jpg>
```

## Options

### Images and Directory Options
* --image_size: Maximum side length of output image; default is 512.
* --style_image: image to use for style and texture.
* --content_image: image to use for content.
* --output_dir: default is tempdir/neural_style_output. 

### Model Options
* --num_iters: number of iterations, default 20.
* --avg_pool: convert max_pool layers to avg_pool, as in original paper? Default is True.
* --content_weight: weight on content image, default is 0.001
* --style_weight: weight on style image, default is 3.0
* --total_variation_weight, default is 0.001

## Improving the Quality of the Generated Images
Fork the repo and try incorporating the results in [this paper!](https://arxiv.org/abs/1611.07865).

Tested on Ubuntu 17.10, Keras 2.1.3, Tensorflow 1.4.0
