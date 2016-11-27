# Comparison-of-biophysically-plausible-Implementation-of-neural-fields

Publication :
https://www.researchgate.net/publication/306324147_Comparison_of_biophysically_plausible_Implementation_of_neural_fields


 Modeling of biophysically plausible neural networks in various scales has provided in sights in studies ranging from basic function of neural circuitry to mechanism of memory and sleep. This approach has been shown more promising than ever as more realistic models can be implemented thanks to the rapid advance of computer technologies. Nevertheless, model complexity and size still pose significant challenges to simulation speed and reproducibility. The simulation can be accelerated either by introducing concepts of software design, or by reduce the complexity of the model. Here, we demonstrate the computational utility is optimized by employing both strategies. We transport models of different neural levels from MatLab to NEST, and compare the results of simulation and the performance of the two software. On the other hand, we reduce the complexity of single neuron model and discuss the limitation of the simplified model. Finally, the computational speed is compared. This study shows NEST enables evaluation of the relations between psychophysical data and biophysical data by realizing implementation of complicated, large-scaled biophysically plausible neural fields .


## Folders

**CurveFitting** is the folder contianing Matlab codes used to fit the F-I curves. The folder **Least Square Fit** contains  curve fitting functions used to fit the simulation results produced by Python files. **functions** contains many functions that produce F-I curve data. These functions of several single neuron models are used in curve fitting functions. Simplified Aeif models are in this folder. 

**Python Simulation of neural field** is the folder contianing Python codes used to simulate the biophysical models used in the paper. The models are based on [ NEST (NEural simulation tool)](http://www.nest-simulator.org) The folders under this directory contains the results (images) and the source codes of the simluations that were used to evaluate the reproducibility of the two simulation tools (Matlab and NEST).


![alt text](https://github.com/Po-Hsuan-Huang/Comparison-of-biophysically-plausible-Implementation-of-neural-fields/blob/master/Readme_img1.png)
