# Image-Denoising

### Background
Image noise is a random change in a pixel hue or saturation value of a pixel in an image. There can be multiple sources of image noise. Noise can get introduced inherently at different stage of image capture pipeline from light variation, camera optics, image sensor to image storage.

### The Problem
One of the fundamental challenge in the field of Image processing and Computer vision is Image denoising where the goal is to estimate the original image by suppressing noise from the contaminated region in an Image. This has numerous applications such as:
* digitisation and restoration of old images/documents.
* satellite imagery, etc

The aim of this project is to extract a clean image Ix from the noisy image Iy, with noisy component as In, which is explained by Iy=Ix+In.

### Metrics

#### PSNR(Peak Signal-to-Noise Ratio)
PSNR is most easily defined via the mean squared error (MSE). Given a noise-free m×n monochrome image I and its noisy approximation K, MSE is defined as:

<img src="https://latex.codecogs.com/svg.latex?%24%24MSE%20%3D%20%5Cfrac%7B1%7D%7Bm*n%7D%5Csum%5Climits_m%20%5Csum%5Climits_n%28I_%7By%7D-I_%7Bx%7D%29%5E2%24%24" />

<img src="https://latex.codecogs.com/svg.latex?PSNR%20%3D%2020%20*%20%5Clog%20%5Cmax%28f%29%20/%20%28MSE%5E%7B0.5%7D%29" />




#### SSIM

The difference with respect to other techniques mentioned previously such as MSE or PSNR is that these approaches estimate absolute errors; on the other hand, SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. Structural information is the idea that the pixels have strong inter-dependencies especially when they are spatially close. These dependencies carry important information about the structure of the objects in the visual scene. Luminance masking is a phenomenon whereby image distortions (in this context) tend to be less visible in bright regions, while contrast masking is a phenomenon whereby distortions become less visible where there is significant activity or "texture" in the image.


<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/63349f3ee17e396915f6c25221ae488c3bb54b66" />

### Data 

### Approaches
#### Unsupervised
##### Vanilla PCA

[TODO]:  add link to the notebook, check if PCA can be done componentwise and add result here, and review.

Principal component analysis is a orthogonal transformation which seeks the direction of maximum variance in the data and commonly used in dimensionality reduction of the data. Data with maximum variance contains most of the data needed to present the whole dataset. In image denoising one has to take care of the compromise between noisy data and preserving the high variance image data detail. We can start by looking into the plain PCA analysis to see how PCA inherently tries to reduce the noise in an image.

The basic intuition behind denoising the image is that any components with variance much larger than the effect of the noise should be relatively unaffected by the noise. So if you reconstruct the data using just the largest subset of principal components, you should be preferentially keeping the signal and throwing out the noise. This is the very basic idea behind how a PCA simply can reduce noise from the image. Though this is not an efficient approach(we will look at better approach through modified PCA in next section), we can look how a plain vanilla PCA can improve the PSNR(peak signal to noise ration) over an image.
We tried the plain vanilla PCA method in the mnist digit data set, an then in the RGB images.The approach is:
* Take the mnist dataset
* Add some random gaussian noise to the image
* Plot the variance vs Component curve to determine the component storing the highest variance.
* Apply inverse PCA to get the image back using the components derived in above step.
* Visualize the dataset again to see difference.

Before PCA transformation the digit dataset looks like this:
![Mnist data before denoising](assets/vanilla_pca/mnist_digit_before.png)

After this we add some random gaussian noise to it, to make pixels more blurr and add some noise to it.
After adding random gaussian noise the digit dataset looks like this:
![Adding Random Gaussian noise to the data](assets/vanilla_pca/mnist_noisy.png)

Now we try to see the number of components which can capture most of the variance in the data. From the below
figure we can see that first 10 component can capture 80 percent of the variance in the data.
![Plotting Component vs variance graph](assets/vanilla_pca/mnist_var_comp.png)

Next we try to plot the digit data for our noisy image using the first 10 component, and we can clearly see that 
it PCA preserves the signals and loses the noise from the data:
![Denoised data](assets/vanilla_pca/mnist_denoised.png)


Let's run the same experiment in a RGB image to see if there an improvement in PSNR after PCA analysis.
The method remains the same:
* Take an Noisy RGB image
* Flattens the Image across 3 channels.
* Do PCA analysis to get the max number of components restoring maximum vaiance.
* Do inverse PCA transform to retrieve the same image using the component derived in the above step.
* Calculate the PSNR value for original,noisy image and original,denoised image and see if there is an improvement.

We ran the above process for the CBSD68-dataset provided by Berkeley. It contains both noisy and original image with different gaussian noise level.
Here below you can see the original image and then denoise image. 
[Image]

We plotted the psnr graphs for all the noisy datasets and from the figure below you can observe that when there is no
noise or very less gaussian noise than it is hard for the PCA to denoise the data, but when you started increasing the noise in the image(upto 50 gaussian noise), you can observe that psnr value improves for all images.

| | 
|:-------------------------:|
**Gaussian Noise level-5**
![Gaussian noise 5](assets/vanilla_pca/noise_5_psnr.png "Gaussian Noise level-5")
**Gaussian Noise level-15**
![Gaussian noise 15](assets/vanilla_pca/noise_15_psnr.png "Gaussian Noise level-15") 
**Gaussian Noise level-25**
![Gaussian noise 25](assets/vanilla_pca/noise_25_psnr.png "Gaussian Noise level-25") 
**Gaussian Noise level-35**
![Gaussian noise 35](assets/vanilla_pca/noise_35_psnr.png "Gaussian Noise level-35") 
**Gaussian Noise level-50**
![Gaussian noise 50](assets/vanilla_pca/noise_50_psnr.png "Gaussian Noise level-50") 


##### locally adaptive PCA
#### Supervised
##### Sateesh


## Approach 2

In this approach, we have used supervised learning to learn the clean image given an noisy image. The function approximator chosen is a neural network comprising of convolutinal and residual blocks as shown []. 

## Experiment 

Two datasets were used in this experiment [PASCAL] and [CBSD]. The PASCAl training data contains approximately () images. This dataset is split into training, valid and test datasets with (),() and () respectively. As shown in (), the architecture takes an Input image, then it is passed through convolutional layers having 64 filters of 9x9 kernel size, 32 filters of 5x5 kernel size and 1 filter of 5x5 filter size respectively. Relu activations are used in all the layers. Stride used is of size 1, so the output size is reduced by 8 pixels in all directions. To accomodate this, we can either pad the image or give an larger input size. We chose to go with the latter as chosen in []. So the input image size used is  33x33 and output size is 17x17. So as input images have varied dimensions in PASCAL dataset(or other datasets), during preprocessing we have cropped the images. Note that crop can be at random part of the image. So, this acts as data augmentation technique as well. The 33x33 input image should have noise as well. The added noise is random from 10-50.

##### architecture

 <img src="https://drive.google.com/uc?export=view&id=1e1CNawerSWRO6VuyaDZwvvmqQ3-zFBZB" width="425" height = "700"/> <img src="https://drive.google.com/uc?export=view&id=1QXtY1UzQ3NPQJbD6wGcGBFkEELuc4uE6" width="425" height = "600"/> 

##### Training network: 

Pytorch is used to wirte the code and network is trained in google colab [] using GPU's. 
Training is done batchwise using 128 batches of 33*33 noisy input images. MSE loss and Adam optimzer were used with learning rate of 0.001. Using the clean target image of 17x17, the MSE loss is calculated from the networks output image. Training is done for 100 epochs at this configuration. As loss got stagnated here we reduced learning rate to 0.0001 and trained another 50 epochs. After this, we added a residual block to the network and initialized its weights to random weights, with other layers weights unchanged. This network is trained for another 50 epochs with learning rate 0.01. We have stopped training at this point due to longer training periods (50 epochs approximately took 2 hours), even though it been shown in [], that adding more residual blocks will improve the PSNR scores further. At all stages of training, validation loss (as shown) is calculated and monitored as well to see if the network is generalizing to unseen data.

## loss curves.

 ![image](https://drive.google.com/uc?export=view&id=1tEq0Vf-vPjtD-smrQXUVQ9Vc0-qc2qJo) 

## Results

PASCAL total average PSNR on test set for both models. <done>
  
  <add two tables for two datasets>

Generate graphs for CBSD dataset. <tbd, do we need it? as I have added in tabular format>

Results on selected images. <done>

  <to be added>


##### Ramesh

