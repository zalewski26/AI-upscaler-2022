# Image Upscaler
Project implemented as an engineering thesis entitled "Implementation of artificial intelligence algorithms to increase the resolution of images" at Wrocław University of Science and Technology. The program allows you to train and use the convolutional network model (SRCNN) and generative adversarial network (SRGAN) to increase the image resolution. The model architectures have been proposed in scientific works:
- Image Super-Resolution Using Deep Convolutional Networks (https://arxiv.org/abs/1501.00092)
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (https://arxiv.org/abs/1609.04802)

### Data preparation
The process of preparing a training and validation set.
```
python3 prepare_data.py
```
### Training
The process of training a selected model for a specified number of input channels.
```
python3 train.py --arch $[srcnn/srgan] --channels $[1/3]
```
### Increasing resolution
The process of increasing the resolution of a given photo using the selected model.
```
python3 run.py --arch $[srcnn/srgan] --channels $[1/3] --img-path $image_to_upscale --weights-path $path_to_saved_weights
```
If the 'weights-path' parameter is omitted, the weights will be selected from the 'data/Saved' directory.
