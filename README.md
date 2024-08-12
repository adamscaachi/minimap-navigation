# minimap-navigation

Many video game genres utilise UI elements known as minimaps to provide the player with an overview of their surroundings. A typical minimap consists of an arrowhead to convey the location and direction of the player's character, circular icons to represent the locations of enemy monsters, and a translucent triangle (known as a vision cone) to indicate the direction of the player's camera. Attempting to extract the player's camera direction from the minimap using classical computer vision approaches can be unreliable due to the constantly changing background, therefore a supervised machine learning approach is used instead.

## Generator

The generator class creates labelled minimap images by repeatedly selecting a random part of the world map and overlaying an arrowhead and vision cone at randomly generated angles, additional details such as icons indicating nearby monsters are also placed at random positions. 10,000 datapoints were generated using this method, a randomly selected subset of them are shown below.

![grid](https://github.com/user-attachments/assets/f0e1d222-3435-4bcb-ae8e-d3515f919bd9)

## Model 

A CNN architecture is used to process the minimap images and the output is flattened for further processing by two fully connected layers. The model outputs a prediction for the vector (α, β), where α = cos(θ) and β = sin(θ). Here, θ is the rotation angle of the vision cone. Predicting the sine and cosine of the rotation angle is necessary as there is a discontinuity where the angle passes through 360° and becomes 0° that the loss function (mean square error) does not interpret well.

## Results

The model was trained for 30 epochs and used to predict values of α and β on a test dataset consisting of images generated using a different world map to the one used for generating the training data. The root mean square error of the predictions computed over the test dataset was found to be 3.7°. A demonstration of the model on a random sample of images from the test dataset is shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7e92df8-734b-4eeb-b293-5da75585ff10" width="150" />
</p>

