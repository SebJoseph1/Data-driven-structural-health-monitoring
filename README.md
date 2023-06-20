# 5ARIP0 Code Repository
## Authors
- J. Betran Menz
- A. Brugnera
- V.J.A. Houben
- S. Joseph
- S. Yao
## Abstract
This study presents an innovative approach to automating the detection and measurement of crack growth in metal specimens under controlled conditions, using an Artificial Intelligence (AI) system. Current methods for crack detection and estimation in steel specimens are time-consuming, subjective, and miss out on certain details of the crack growth. Importantly, these methods primarily quantify the crack length as a single numerical value, neglecting the complex nature of crack propagation, which doesn't always follow a linear path. 
Addressing these issues, this paper presents a efficient and reliable system using AI that is designed for this task. The project comprises of five distinct stages: Data Collection, Data Labeling, Neural Network Implementation, Curve Measurement, and Image Acquisition. Custom data set was gathered and labeled in order to train the neural network, for which three different architectures and ideas were explored, from which Unet with VGG backbone showed the best results. The trained network was employed to detect cracks and estimate their growth. Furthermore, for the image acquisition process a physical system had to be designed and developed that was then optimized to capture the instances of maximum crack visibility. The successful implementation of this system serves a indicator for future exploration and innovation in the application of AI within material science and engineering. This study serves as a foundational step towards a more automated and accurate approach to material testing.

## Weights
The file containing the weights of the model is too large to upload directly, but can be accessed with the following link:
https://tuenl-my.sharepoint.com/:u:/g/personal/v_j_a_houben_student_tue_nl/EQl_NUorPk9Ji6I1qW5uZRIBLRyC_9unDsUF40tiQi7Mhw?e=q4Vpfb

## Running the GUI
The required libraries can be installed by
```
pip install -r requirements.txt
```

The GUI can be executed by 
```
python ui/application.py
```
