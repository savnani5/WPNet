# Towards Online Waypoint Generation for a Quadrotor Using Enhanced Monocular Depth Estimation.
 
Exploring and navigating autonomously in uneven and unstructured environments is an unsolved problem because
flights have a high risk of collision and have a battery bottleneck due to limited storage onboard. Also due to environmental uncertainites, an intelligent autonomous aerial
system is essential to navigate these environments. Current literature on autonomous flights focuses on mapping the environment first then navigating them. Our approach doesn’t
assume that the environment is mapped, we take in real time images and try to navigate the environment. Also having a dense depth map of the real-world can be very useful in
applications including navigation and scene understanding, augmented reality. Based on our experimental analysis of existing architectures and training strategies we set out with the design goal to develop a simpler architecture that makes training and future modifications easier. Our architecture is a modification
of the DenseDepth to compute depth and this feeds into a novel parallized 2-way waypoint prediction network. We rely on transfer learning of the Encoder-Decoder weights
for the depth network and train the WPNet from Ground zero. A novel thing about this architecture is that we can train each module seperately and it does not affect the performance of the network. This modular approach is advantageous if in future we want to substitute any network or improve the performance of any submodule. Our approach uses Depth Estimation with Multi-Task Regression-based Learning to individually learn the position of waypoints in X, Y, Z coordinates within the scene
in addition to learning the qw, qz, qy, qz quaternions. The flight controller can change position and speed in real time to perform obstacle avoidance. Also it uses the inertialand visual information to predict waypoints so its a very robust approach in GPS-denied environments. 

## Approach


### Depth Network


### Waypoint Prediction Network


## Results


## Future Work

For more recent future work we would like to code another evaluation metric for the network output like the collision free distance moved additionally controlling the outputs with a traditional controller like LQR and simulating the results in airsim. We can code this based on the depth
information and the check if the predicted waypoint is colliding with the nearby obstacle or not. Also we would like
to change the Encoder-Decoder part of the Densedepth to a more parameter efficient network like MobileNet without
compromising the performance. Furthermore for a longer timeline we are planning to make this network totally endto-end where it will also learn the control policies after the
waypoint trajectory prediction, although generating such a custom dataset for this approach might be a big problem.
