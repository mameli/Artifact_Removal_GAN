# Artifact removal GAN

In this work, we devise a neural network to remove from an image the artefacts of the most used compression format: JPEG.
The structure of the model is based on the Generative Adversarial Network (GAN) design. This particular architecture is composed of two elements that are competing with each other: the Generator and the Discriminator. 
    
U-net is used as the Generator of the model. This component takes a JPEG image as input and it outputs the same image without the artefacts.
    
To train the model we use the NoGAN technique. Thanks this method, the training of the neural network is more stable and the model can reach better results. 
    
The key feature of the development of the model is the usage of the perceptual loss function: Learned Perceptual Image Patch Similarity (LPIPS).  This function was created to mimic human vision judgements and it is used to measure the similarity between two images.

---
## Architecture

![original](./imgs/gan.jpg)

The neural net is based on the [DeOldify](https://github.com/jantic/DeOldify) model.

We used [MobileNet](https://github.com/rwightman/gen-efficientnet-pytorch) as the U-Net encoder, [LPIPS](https://github.com/richzhang/PerceptualSimilarity) as loss function and [DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as dataset.

The metrics used are:

- [SSIM]([https://link](https://github.com/jorge-pessoa/pytorch-msssim))
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [BRISQUE](https://github.com/bukalapak/pybrisque)
- [NIQE](http://www.scikit-video.org/stable/modules/generated/skvideo.measure.niqe.html#skvideo.measure.niqe)



---
## Results

<style>
figure{
    display: inline-block;
    width: 33%;
    margin: 0;
    text-align: center;
    padding: 0;
}
</style>

### Original:
![original](./imgs/0803_base.jpg)

### GAN:
![GAN](./imgs/0803_GAN.png)

<p style="padding-bottom: 1cm;"/>

### Crop

<figure style= "width: 45%;">
    <img src="./imgs/0803_base_Crop.png" alt='JPEG quality=20' style="width:100%"/>
    <figcaption>JPEG quality=20</figcaption>
</figure>
<figure style= "width: 45%;">
    <img src="./imgs/0803_GAN_Crop.png" alt='GAN' style="width:100%"/>
    <figcaption>GAN</figcaption>
</figure>


### Original:
![original](./imgs/0416_base.jpg)

### GAN:
![GAN](./imgs/0416_GAN.png)

<p style="padding-bottom: 1cm;"/>

<figure style= "width: 49.5%;">
    <img src="./imgs/0416_base_Crop.png" alt='JPEG quality=20' style="width:100%"/>
    <figcaption>JPEG quality=20</figcaption>
</figure>
<figure style= "width: 49.5%;">
    <img src="./imgs/0416_GAN_Crop.png" alt='GAN' style="width:100%"/>
    <figcaption>GAN</figcaption>
</figure>

<p style="padding-bottom: 1cm;"/>

## Comparison to the ground truth
<figure>
    <img src="./imgs/0222_HR_Crop.png" alt='High resolution' style="width:100%"/>
    <figcaption>High resolution</figcaption>
</figure>
<figure>
    <img src="./imgs/0222_base_Crop.png" alt='JPEG quality=20' style="width:100%"/>
    <figcaption>JPEG quality=20</figcaption>
</figure>
<figure>
    <img src="./imgs/0222_GAN_Crop.png" alt='GAN' style="width:100%"/>
    <figcaption>GAN</figcaption>
</figure>



   

