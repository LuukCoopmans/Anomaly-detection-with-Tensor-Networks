# Anomaly-detection-with-Tensor-Networks
Python code to perform anomaly detection of datasets such as MNIST images with Tensor Networks. 
This is my implementation of the code in the paper https://arxiv.org/abs/2006.02516, 
which I wrote for a previous job interview assignment at google in 2021.

I wanted to improve the code, but since then I have been busy with my new job.
Making the code public now such that other people can play around with it.

A description of the code is in the "Anomaly Detection with Tensor Networks" python notebook.
And ML_tensor python file includes all the helper functions.
The code makes use of automatic differentation of tensor networks with JAX, but it is possible
to switch to other autodiff libraries as can be seen in the notebook.

Package requirements (versions of 12-4-2021):
JAX,
Tensorflow,
pickle,
tensornetwork,
matplotlib

