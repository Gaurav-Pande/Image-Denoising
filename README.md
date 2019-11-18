# Image-Denoising

## Background

Image noise is random variation of brightness or color information in images. There can be multiple sources of image noise. Noise can get introduced inherently at a different stage of image capture pipeline from light variation, camera optics, image sensor to image storage.

## The Problem/Motivation

One of the fundamental challenges in the field of Image processing and Computer vision is Image denoising, where the goal is to estimate the original image by suppressing noise from the contaminated region in an Image. Image Denoising has numerous applications, such as:

- digitization and restoration of old photos/documents.
- satellite imagery, etc

Also Image denoising is useful as a preprocessing step for several computer vision tasks where obtaining the original image content is crucial for strong performance:

- visual tracking
- image segementation
- object detection
- image classification

This project aims to extract a clean image <img src="https://latex.codecogs.com/gif.latex?%24I_%7Bx%7D%24" /> from the noisy image <img src="https://latex.codecogs.com/gif.latex?%24I_%7By%7D%24" />, with noisy component as <img src="https://latex.codecogs.com/gif.latex?%24I_%7Bn%7D%24" />, which is explained by

<center>

<img src="https://latex.codecogs.com/gif.latex?%24I_%7By%7D%3DI_%7Bx%7D&plus;I_%7Bn%7D%24" />.

</center>

## Problem Scope

We are limiting the problem scope to tackle additive guassian white noise(AGWN) only and will demonstrate how supervised and unsupervised techniques could be used to denoise images with AGWN.

## Metrics

##### PSNR(Peak Signal-to-Noise Ratio)

PSNR, is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. PSNR is most easily defined via the mean squared error (MSE). Given a noise-free m×n monochrome image I and its noisy approximation K, MSE is defined as:

<center>

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/3a34719b4f391dba26b3e8e4460b7595d62eece4" />

</center>

The PSNR (in dB) is defined as:

<center>

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fc22801ed1232ff1231c4156b589de5c32063a8a" />

</center>

##### SSIM

The difference with respect to other techniques mentioned previously such as MSE or PSNR is that these approaches estimate absolute errors; on the other hand, SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. Structural information is the idea that the pixels have strong inter-dependencies, especially when they are spatially close. These dependencies carry essential information about the structure of the objects in the visual scene. Luminance masking is a phenomenon whereby image distortions (in this context) tend to be less visible in bright regions, while contrast masking is a phenomenon whereby distortions become less visible where there is a significant activity or "texture" in the image.

<center>

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/63349f3ee17e396915f6c25221ae488c3bb54b66" />

</center>

## Data

As we have multiple approaches and experiments, we have chosen a common dataset CBSD68 [3] to analyze results. The CBSD68 dataset is a dataset commonly used for benchmarking in Image denoising domain. It contains 68 images and corresponding noisy images at different sigma levels.

Note that as this dataset has quite less no. of samples, for supervised learning approach we have also used other datasets for training. We have explored other datasets for unsupervised approach as well as mentioned below.

##### Supervised:

- PASCAL dataset [2]
- DIV2K dataset [8]

##### Unsupervised

- Digits dataset from scikit learn.
- RGB images from CBSD68 dataset for PCA decomposition.

## Approach1(Supervised)

In this approach, we have used supervised learning to learn the clean image given a noisy image. The function approximator chosen is a neural network comprising of convolutional and residual blocks, as shown in figure below. Two experiments were conducted, one with pure convolutional layers and the other with mix of convolutional and residual block as detailed below.

## Experiment1(Deep CNNs)

The code is available [here](https://colab.research.google.com/drive/1ViNx_b5FlwXjzjIqRuYkdNsgF6ZExqRk) and [here](https://colab.research.google.com/drive/1-LJ12r-DJXY3HI0hzVIom2r9RpCdL8Gd).

##### Datasets

Two datasets were used in this experiment PASCAL VOC 2010 [2] and CBSD68. The PASCAl training data contains approximately 10k images. This dataset is split into training, valid and test datasets with ratios 80, 10 and 10 respectively. The CBSD68 is used for testing purpose only, in this experiment.

##### Architecture

As shown in the figure below, the architecture takes an Input image, and then it is passed through convolutional layers having 64 filters of 9x9 kernel size, 32 filters of 5x5 kernel size, and 1 filter of 5x5 filter size respectively. Relu activations are used in all the layers. Stride used is of size 1, so the output size is reduced by 8 pixels in all directions. To accommodate this, we can either pad the image or give a larger input size. we chose to go with the latter as chosen in [1].

<p align="middle">
<img src="https://drive.google.com/uc?export=view&id=1e1CNawerSWRO6VuyaDZwvvmqQ3-zFBZB" width="400" height = "700"/> 
<img src="assets/architecture/Screen Shot 2019-11-17 at 4.46.55 PM.png" width="400" height = "700"/>
</p>
 <center> Figure 1. Network architectures used in (left) Experiment 1 and (right) Experiment 2. The graph is generated using the Netron app [5] </center>
 
 
##### Data Augmentation/Pre-processing

The input image size used is of size 33x33 and output size is 17x17. As input images have varied dimensions in PASCAL dataset(or other datasets), during preprocessing, we have cropped the images. Note that crop can be at random part of the image. So, this acts as data augmentation technique as well. The 33x33 input image should have noise as well. The added noise is random from 10-50 sigmas. The corresponding clean mid-portion of this image is target.

##### Training:

Pytorch [4] is used to write the code, and network is trained in google colab using GPU's. Training is done batchwise using 128 batches of 33x33 noisy input images and 17x17 corresponding clean target images. MSE loss and Adam optimzer were used with learning rate of 0.001. Using the clean target image of 17x17, the MSE loss is calculated from the networks output image. Training is done for 100 epochs at this configuration. As loss got stagnated here we reduced learning rate to 0.0001 and trained another 50 epochs. After this, we added a residual block to the network and initialized its weights to random weights, with other layers weights unchanged. This network is trained for another 50 epochs with learning rate 0.01. We have stopped training at this point due to longer training periods (50 epochs approximately took 2 hours), even though it been shown in [1], that adding more residual blocks will improve the PSNR scores further. At all stages of training, validation loss is calculated and monitored as well to see if the network is generalizing to unseen data.

Note, to experiment further with residual blocks, experiment 2 is performed, which will be detailed below.

##### Traning and validation loss graph:

<center>

<p><img src="https://drive.google.com/uc?export=view&amp;id=1tEq0Vf-vPjtD-smrQXUVQ9Vc0-qc2qJo" alt="image" /></p>

</center>

##### Results and observations:

The average PSNR scores and SSIM scores on the test set of PASCAL, for the best model was given below. Note that best model is 3 layered, as 5 layered one couldn't be trained completely due to computing constraints. Input crop size of 200 was used to show the results instead of 33. Also, left value in the column indicates average PSNR compared with noisy input, while the right bolded one indicates the average PSNR with the denoised output. Similar case with SSIM.

<center>

<table>
<thead>
<tr>
<th><strong>Sigma</strong></th>
<th><strong>PSNR</strong></th>
<th><strong>SSIM</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>10</td>
<td>28.33 to <strong>31.92</strong></td>
<td>0.73 to <strong>0.90</strong></td>
</tr>
<tr>
<td>25</td>
<td>20.63 to <strong>28.94</strong></td>
<td>0.44 to <strong>0.83</strong></td>
</tr>
<tr>
<td>50</td>
<td>15.13 to <strong>25.66</strong></td>
<td>0.24 to <strong>0.70</strong></td>
</tr>
<tr>
<td>50(crop 33)</td>
<td>15.16 to <strong>26.77</strong></td>
<td>0.22 to <strong>0.69</strong></td>
</tr>
</tbody>
</table>

</center>

The same model is tested on the CBSD dataset, Average PSNR and SSIM score are as follows,

<center>

<table>
<thead>
<tr>
<th><strong>Sigma</strong></th>
<th><strong>PSNR</strong></th>
<th><strong>SSIM</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>10</td>
<td>28.26 to <strong>33.33</strong></td>
<td>0.75 to <strong>0.93</strong></td>
</tr>
<tr>
<td>25</td>
<td>20.48 to <strong>29.45</strong></td>
<td>0.45 to <strong>0.85</strong></td>
</tr>
<tr>
<td>50</td>
<td>14.97 to <strong>25.67</strong></td>
<td>0.25 to <strong>0.71</strong></td>
</tr>
<tr>
<td>50(crop 33)</td>
<td>15.04 to <strong>26.68</strong></td>
<td>0.23 to <strong>0.69</strong></td>
</tr>
</tbody>
</table>

</center>

The above results indicate the **model is generalising well** to other datasets having similar noise as AWGN. Also, the net PSNR achieved is a bit a lower than from the paper's [1] best, as we are only using 3 layers for training.

## Experiment2 (Deep ResNets)

In this experiment we implement the residual network connections in the convolutional denoising network. Since residual networks are memory intensive, we train the network on a different dataset [DIV2K] which is smaller and test the network on our validation set : [CBSD]. The DIV2K[8] dataset consists of 800 very high resolution images.

##### Residual networks

It is known that very deep neural networks have very high representational power, but comes very difficult to train compared to shallow networks. This can be attributed the vanishing gradients during backpropagation i.e very little information / learning is happening in the first few layers of the network. This is fixed by creating residual connections between layers. These residual connections allow gradients to flow directly to the earlier layers thus enabling more efficient learning. Essentially the network formed by these residual connections is comparable to a shallow network present within the deeper network. Thus we retain the generalizing power of shallow network.

##### Dataset

We use a pytorch dataloader for setting up the data pipeline, we extract random 128x128 crops of the images as the input image.
We randomly flip it horizontally and vertically as our data augmentation steps.
Then we add gaussian noise to the this image and consider that as our noisy image.
We return this pair of original and noisy image as the training input to our network.
We set the batch size to 8 as it is the maximum allowable size by the constraints of our GPU.

##### Architecture

We use 8 convolutional layers with a skip connection between every convolutional layer and the output of the layer following it. These skip connections allow us to train a much deeper network. The network learns to output the noise component of the image i.e it learns to separate the noise from the ground truth image. So to obtain the denoised image, we subtract the output of our model from the noisy image.

##### Implementation and Hyperparameters

Each convolutional layer consists of 64 filters, kernel size of 3, stride of 1 and padding of 1. This combination allows the layer to preserve the size of the input image after the forward pass, allowing us to arbitrarily stack these layers (as needed for resnet architectures). We use RELU activation function after each convolutional layer.
We also disable the bias components of the layers, this reduced the amount of artifacts present in the output image after denoising.
For optimization we use the ADAM optimizer with learning rate of 0.001 and train the network for 5 epochs.
In order to improve convergence, we also use learning rate scheduler to reduce learning rate by factor of 10 if there is no improvement for 3 epochs.

##### Results and Observations

<p align="middle">
  <img src="assets\deep_resnet\loss_vs_iterations_sigma_25.jpg" width="400" />
  <img src="assets\deep_resnet\psnr_vs_iterations_sigma_25.jpg" width="400" /> 
  <img src="assets\deep_resnet\ssim_vs_iterations_sigma_25.jpg" width="400" />
</p>

During evaluation, we apply the network on the whole image as the convolutional operations can be applied on any image size.

We obtain the results as documented in the tables below. We obtain reasonable improvements to PSNR (25.6) and SSIM scores. We get PSNR results comparable to our other models. We notice that the training / validation loss are very close which implies that there is possiblility of more improvement which can be explored with more compute resources.

Another novelty that we applied is passing the denoised image back into the model for further refinement, we observe that the PSNR values get a slight reduction but the SSIM score improves by about 0.1 (especially with larger noise ranges). This approach is similar to our PCA approach with iterative application.

## Approach2 (Unsupervised)
In this approach we used unsupervised learning techniques to solve the problem of image denoising.There are 2 experiments here as follows:

## Experiment3 (Vanilla PCA)

Principal component analysis is an orthogonal transformation that seeks the direction of maximum variance in the data and commonly used in dimensionality reduction of the data. Data with maximum variance contains most of the data needed to present the whole dataset. In image denoising, one has to take care of the compromise between noisy data and preserving the high variance image data detail. We can start by looking into the PCA analysis to see how PCA inherently tries to reduce the noise in an image.

The basic intuition behind denoising the image is that any components with variance much larger than the effect of the noise should be relatively unaffected by the noise. So if you reconstruct the data using just the most significant subset of principal components, you should be preferentially keeping the signal and throwing out the noise. Though this is not an efficient approach(we will look at better approach through modified PCA in the next section), we can examine how a plain vanilla PCA can improve the PSNR(peak signal to noise ration) over an image.

We tried the plain vanilla PCA method in the mnist digit data set, and then in the RGB images. The approach is:

- Take the Digits dataset
- Add some random Gaussian noise to the image
- Plot the variance vs Component curve to determine the component storing the highest variation.
- Apply inverse PCA to get the image back using the components derived in the above step.
- Visualize the dataset again to see the difference.

Before PCA transformation the digit dataset looks like this:

<center>

<p><img src="assets/vanilla_pca/mnist_digit_before.png" alt="Digits data before denoising" /></p>

</center>

After this we add some random Gaussian noise to it, to make pixels more blurr and add some noise to it.
After adding random gaussian noise, the digit dataset looks like this:

<center>

<p><img src="assets/vanilla_pca/mnist_noisy.png" alt="Adding Random Gaussian noise to the data" /></p>

</center>

Now we try to see the number of components which can capture most of the variance in the data. From the below
figure, we can see that first 10 components can capture 80 percent of the variance in the data.

<center>

<p><img src="assets/vanilla_pca/mnist_var_comp.png" alt="Plotting Component vs variance graph" /></p>

</center>

Next, we try to plot the digit data for our noisy image using the first 10 components, and we can see that
it PCA preserves the signals and loses the noise from the data:

<center>

<p><img src="assets/vanilla_pca/mnist_denoised.png" alt="Denoised data" /></p>

</center>

Let's run the same experiment in a RGB image to see if there is an improvement in PSNR after PCA analysis.
The method remains the same:

- Take a Noisy RGB image
- Flattens the Image across 3 channels.
- Do PCA analysis to get the max number of components restoring maximum variance.
- Do inverse PCA transform to retrieve the same image using the component derived in the above step.
- Calculate the PSNR value for original, noisy image and original, denoised image and see if there is an improvement.

We ran the above process for the CBSD68-dataset provided by Berkeley. It contains both noisy and original image with different gaussian noise level.In the below figures there is comparison which is been made to see how the psnr value and how smim values improves after doing PCA decomposition in a noisy image, but the **limitation** of vanilla PCA is that it is not necessary that it will reduce the noise always, but it always captures the data with higher variance. To make the point consider the result on a original and noisy and its denoised part below:

<center>

<table>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;"></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><strong>Original Image</strong></td>
<td style="text-align: center;"><strong>Noisy Image-50</strong></td>
<td style="text-align: center;"><strong>Denoised Image-50</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/original_png/0064.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/noisy50/0064.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise50/0064_denoised.png" width="300" height = "150"/></td>
</tr>
</tbody>
</table>

</center>

You can observe from above that the results are not that good but there is an improvement in the psnr and smim values, because the denoised part tries to capture the pixels with higher variance. That is why most part of the image in denoised is a bit brown as that is the prominent color in our original image as well.

<center>

<table>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><strong>Gaussian Noise level-50</strong></td>
<td style="text-align: center;"><strong>Gaussian Noise level-25</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise_50_psnr.png " width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise_25_psnr.png " width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR comparison accross images</em></td>
<td style="text-align: center;"><em>PSNR comparison accross images</em></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise_50_smim.png " width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise_25_smim.png " width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>SMIM comparison accross images</em></td>
<td style="text-align: center;"><em>SMIM comparison accross images</em></td>
</tr>
</tbody>
</table>

</center>

To rerun the experiment, please clone this repository and run PCA.ipynb notebook under notebooks directory.

<!--
**Gaussian Noise level-5**
![Gaussian noise 5](assets/vanilla_pca/noise_5_psnr.png "Gaussian Noise level-5")
**Gaussian Noise level-15**
![Gaussian noise 15](assets/vanilla_pca/noise_15_psnr.png "Gaussian Noise level-15")
**Gaussian Noise level-25**
![Gaussian noise 25](assets/vanilla_pca/noise_25_psnr.png "Gaussian Noise level-25")
**Gaussian Noise level-35**
![Gaussian noise 35](assets/vanilla_pca/noise_35_psnr.png "Gaussian Noise level-35")
-->

## Experiment4 (Local Pixel Grouping - Principle Component Analysis)

##### Approach

This approach uses principal component analysis (PCA) with local pixel grouping (LPG) to do image denoising. It ensures that the image local features are well preserved after PCA transfromation but the noise components is removed. Each pixels and its nearest neighbours are modeled as vector variables and training samples are selected from local window using block matching based local pixel grouping.
It is evident with experiments that this approach can be iteratively applied with appropriate adaptive noise parameter tuning to improve the denoising performance.

##### Intuition

In this approach a pixel and pixels spaitially local to it make a single feature vector. If two pixels which are next to each other are to be considered, then it is almost right to assume that pixels will have almost same values in ideal scenario and the only variation that can be there is because of noise. So, it is right to approximate the value of the these pixels with single value which can be approximated by projecting these points into the principle components of the feature space. As we now, there are two perspective to look of PCA,

- One is to reduce the distance between the point and projected point
- Second is to increase the variance in the principle component direction

Following picture illustrates the pixel to be denoised, freature vector and training block.

<center>
<img src="assets/lgp_pca/doc/LPG_PCA_feature.png" width="400" height = "200"/>
</center>

Following is the 2-stage pipeline using LPG-PCA:

<center>
<img src="assets/lgp_pca/doc/LPG_PCA_pipeline.png" width="600" height = "200"/>
</center>

##### Limitations

Following are limitations for this approach:

- This approach expects an estimated noise level for better perforance. So, it may be required to fine tune the algorithm
  parameters for any new system.
- This approach is compute intensive as it tries to do pixel processing for denoising each pixel.

## Results comparison across approaches:

### Average PSNR on CBSD68 dataset for all experiments:

<center>
<table>
<thead>
<tr>
<th><strong>Sigma</strong></th>
<th><strong>Deep CNNs</strong></th>
<th><strong>Deep ResNets</strong></th>
<th><strong>Vanilla PCA</strong></th>
<th><strong>LPG_PCA</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>10</td>
<td>28.26 to <strong>33.33</strong></td>
<td>28.26 to <strong>30.29</strong></td>
<td>28.26 to <strong>26.46</strong></td>
<td>28.26 to <strong>32.93</strong></td>
</tr>
<tr>
<td>25</td>
<td>20.48 to <strong>29.45</strong></td>
<td>20.48 to <strong>25.9</strong></td>
<td>20.48 to <strong>22.93</strong></td>
<td>20.48 to <strong>27.72</strong></td>
</tr>
<tr>
<td>50</td>
<td>14.97 to <strong>25.67</strong></td>
<td>14.97 to <strong>22.8</strong></td>
<td>14.97 to <strong>18.60</strong></td>
<td>14.97 to <strong>17.93</strong></td>
</tr>
</tbody>
</table>
</center>

### Average SSIM on CBSD68 dataset for all experiments:

<center>
<table>
<thead>
<tr>
<th><strong>Sigma</strong></th>
<th><strong>Deep CNNs</strong></th>
<th><strong>Deep ResNets</strong></th>
<th><strong>Vanilla PCA</strong></th>
<th><strong>LPG_PCA</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>10</td>
<td>0.75 to <strong>0.93</strong></td>
<td>0.75 to <strong>0.937</strong></td>
<td>0.75 to <strong>0.9</strong></td>
<td>0.75 to <strong>0.95</strong></td>
</tr>
<tr>
<td>25</td>
<td>0.45 to <strong>0.85</strong>.</td>
<td>0.45 to <strong>0.85</strong></td>
<td>0.45 to <strong>0.72</strong></td>
<td>0.45 to <strong>0.85</strong></td>
</tr>
<tr>
<td>50</td>
<td>0.25 to <strong>0.71</strong></td>
<td>0.25 to <strong>0.78</strong></td>
<td>0.25 to <strong>0.46</strong></td>
<td>0.25 to <strong>0.38</strong></td>
</tr>
</tbody>
</table>
</center>

## Qualitative Results (from all approaches/experiments):


## Results with sigma = 50

<center>

<table>
<thead>
<tr>
<th style="text-align: center;"><strong>Original</strong></th>
<th style="text-align: center;"><strong>Nosiy Input with sigma=50</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><img src="assets/original_png/0064.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/noisy50/0064.png" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>NA</em></td>
<td style="text-align: center;"><em>PSNR=14.91, SSIM=0.31</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Deep CNN denoised output</strong></td>
<td style="text-align: center;"><strong>Deep Resnet Denoised output</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/dn_resnet_150/50/0.jpg" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/deep_resnet/50/0.jpg" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR = 24.25, SSIM = 0.73</em></td>
<td style="text-align: center;"><em>PSNR=21.6, SSIM=0.70</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vanilla PCA Denoised output</strong></td>
<td style="text-align: center;"><strong>LPG_PCA</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise50/0064_denoised.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/lgp_pca/img64_noisy50_denoised2_img.png" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR=19.15, SSIM=0.58</em></td>
<td style="text-align: center;"><em>PSNR=22.79, SSIM=0.72</em></td>
</tr>
</tbody>
</table>

</center>

## Results with sigma = 25

<center>

<table style="text-align:center;">
<thead>
<tr>
<th style="text-align: center;"><strong>Original</strong></th>
<th style="text-align: center;"><strong>Nosiy Input with sigma=25</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><img src="assets/original_png/0047.png" width="150" height = "300"/></td>
<td style="text-align: center;"><img src="assets/noisy25/0047.png" width="150" height = "300"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>NA</em></td>
<td style="text-align: center;"><em>PSNR=20.19, SSIM=0.21</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Deep CNN denoised output</strong></td>
<td style="text-align: center;"><strong>Deep Resnet Denoised output</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/dn_resnet_150/25/5.jpg" width="150" height = "300"/></td>
<td style="text-align: center;"><img src="assets/deep_resnet/25/5.jpg" width="150" height = "300"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR = 32.78, SSIM = 0.84</em></td>
<td style="text-align: center;"><em>PSNR=29.07, SSIM=0.86</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vanilla PCA output</strong></td>
<td style="text-align: center;"><strong>LPG_PCA</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise25/0047_denoised.png" width="150" height = "300"/></td>
<td style="text-align: center;"><img src="assets/lgp_pca/img47_noisy25_denoised2_img.png" width="150" height = "300"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR=19.15, SSIM=0.58</em></td>
<td style="text-align: center;"><em>PSNR=29.0, SSIM=0.81</em></td>
</tr>
</tbody>
</table>

</center>

## Results with sigma = 10

<center>

<table>
<thead>
<tr>
<th style="text-align: center;"><strong>Original</strong></th>
<th style="text-align: center;"><strong>Nosiy Input with sigma=10</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><img src="assets/original_png/0011.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/noisy10/0011.png" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>NA</em></td>
<td style="text-align: center;"><em>PSNR=28.12, SSIM=0.61</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Deep CNN Denoised output</strong></td>
<td style="text-align: center;"><strong>Deep Resnet Denoised output</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/dn_resnet_150/10/8.jpg" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/deep_resnet/10/8.jpg" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR = 35.41, SSIM = 0.94</em></td>
<td style="text-align: center;"><em>PSNR=32.5, SSIM=0.943</em></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vanilla PCA Denoised output</strong></td>
<td style="text-align: center;"><strong>LPG_PCA</strong></td>
</tr>
<tr>
<td style="text-align: center;"><img src="assets/vanilla_pca/noise10/0011_denoised.png" width="300" height = "150"/></td>
<td style="text-align: center;"><img src="assets/lgp_pca/img11_noisy10_denoised2_img.png" width="300" height = "150"/></td>
</tr>
<tr>
<td style="text-align: center;"><em>PSNR=19.15, SSIM=0.58</em></td>
<td style="text-align: center;"><em>PSNR=33.97, SSIM=0.95</em></td>
</tr>
</tbody>
</table>

</center>

<h2 id="conclusion">Conclusion</h2>

<p>In this project we conducted different experiments for superevised and unsupervised machine learning algorithms for image denoising. We began with vanilla PCA to understand how high variance pixels are retained in the image giving us the intuition for removal of noised pixels components. We saw how various techniques like DnResNet, LPG PCA gives better results with good psnr values.</p>

<h2 id="references">References:</h2>

<ol>
<li>Ren, H., El-Khamy, M., &amp; Lee, J. (2019). DN-ResNet: Efficient Deep Residual Network for Image Denoising. Computer Vision – ACCV 2018 Lecture Notes in Computer Science, 215–230. doi: 10.1007/978-3-030-20873-8_14</li>

<li>pascal-voc-2010. (n.d.). The {PASCAL} {V}Isual {O}Bject {C}Lasses {C}Hallenge 2010 {(VOC2010)} {R}Esults. Retrieved from <a href="http://www.pascal-network.org/challenges/VOC/voc2010/workshop/index.html">here</a></li>

<li>Clausmichele. (n.d.). clausmichele/CBSD68-dataset. Retrieved from <a href="https://github.com/clausmichele/CBSD68-dataset">here</a>.</li>

<li>pytorch/pytorch. Retrieved from <a href="https://github.com/pytorch/pytorch">here</a>.</li>

<li>lutzroeder/netron. Retrieved from <a href="https://github.com/lutzroeder/netron">here</a></li>

<li>Vanderplas, Jacob T. Python Data Science Handbook: Tools and Techniques for Developers. OReilly, 2016.</li>

<li>Zhang, L., Dong, W., Zhang, D., &amp; Shi, G. (2010). Two-stage image denoising by principal component analysis with local pixel grouping. Pattern Recognition, 43(4), 1531–1549. doi: 10.1016/j.patcog.2009.09.023</li>

<li>DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges</li>
</ol>

<h4 id="contributions">Contributions</h4>

<p>All members of the project have contributed equally in discussions, project formulation, Report generation. For individual learning, each member focused on the following:</p>

<ul>
<li>Sai Sateesh Gudapati : Literature Survey, Deep CNNs, Report</li>

<li>Ramesh Arvind Ravi: Literature Survey, Deep ResNets, Report</li>

<li>Varun Dagar: Literature Survey, LPG_PCA, Report</li>

<li>Gaurav Pande: Literature Survey, Vanilla PCA, Report</li>

<li>Sri Vivek Vanga: Literature Survey, Vanilla PCA, LPG_PCA, Report</li>
</ul>
