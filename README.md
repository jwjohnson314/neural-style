# neural-style

This is a Keras/Tensorflow implementation of the neural-style algorithm as first described in [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Gatys *et. al.* Disclaimer: most of the code here isn't original; it largely follows Francois Chollet's implementation as laid out in his book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff), and in the notebooks on Github that accompany the book [here](https://github.com/fchollet/deep-learning-with-python-notebooks). One notable difference between this implementation and the original is that gatys *et. al.* normalized the weights of the VGG19 network; here, we use non-normalized weights. 


## What It Does
The algorithm combines the content of an image with the styles and textures of another image using features derived from a convolutional neural network. For example:
<div align="center">
  <img src="https://github.com/jwjohnson314/neural-style/master/images/cubist_9.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/master/images/chopin.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/master/images/cubist_chopin.jpg" height="710">
</div>

<div align="center">
  <img src="https://github.com/jwjohnson314/neural-style/master/images/il_peccato.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/blob/master/images/brooklyn_bridge.jpg" height="223px">
  <img src="https://github.com/jwjohnson314/neural-style/master/images/il_brooklyn.jpg" height="710">
</div>

## Basic Usage

```
python neural_style.py --style_image=<style_image.jpg> --content_image=<content_image.jpg>
```

## Options

## Images and Directory Options
* --image_size: Maximum side length of output image; default is 512.
* --style_image: image to use for style and texture.
* --content_image: image to use for content.
* --output_dir: default is tempdir/neural_style_output. 

## Model Options
* --num_iters: number of iterations, default 20.
* --avg_pool: convert max_pool layers to avg_pool, as in original paper? Default is True.
* --content_weight: weight on content image, default is 0.001
* --style_weight: weight on style image, default is 3.0
* --total_variation_weight, default is 0.001
