# SIRo-TL: Statistical Inference for High-Dimensional Robust Regression after Transfer Learning

![License](https://img.shields.io/github/license/DAIR-Group/SIRo-TL)

SIRo-TL is a Python package that implements a framework for conducting valid statistical inference in High-Dimensional Robust Regression under Transfer Learning (TL) scenarios. The main idea is to synergize the inherent robustness of Huber regression with the rigorous guarantees of the selective inference framework to handle datasets plagued by outliers and heavy-tailed noise. Our proposed method leverages auxiliary source domain information to stabilize estimation and provides valid $p$-values for the selected features. This ensures that the False Positive Rate (FPR) is kept under control while maximizing the True Positive Rate (TPR), offering a resilient and interpretable solution for complex real-world environments.
## The SIRo-TL framework
SIRo-TL enhances outlier detection reliability by calibrating p-values to filter out spurious discoveries and identify true anomalies.
![SIRo-TL enhances outlier detection reliability by calibrating p-values to filter out spurious discoveries and identify true anomalies.](./figure/SIRo_TL .png)

## Truncated Region Z
Visualization of the truncation region identification. The region Z (high-
lighted in red) is obtained by intersecting the valid intervals from the homotopy path with the intervals satisfying the outlier detection criteria
![Visualization of the truncation region identification. The region Z (high-
lighted in red) is obtained by intersecting the valid intervals from the homotopy path with the intervals satisfying the outlier detection criteria](./figure/Truncation_Region_Z.png)
## Requirements
This package has the following requirements:

    numpy
	cvxpy
	scipy
	matplotlib
	statsmodels
	mpmath

## Usage

We provide several Jupyter notebooks demonstrating how to use SIRo-TL framework in action.

- Examples for conducting inference for High-Dimensional Robust Regression and Outlier Detection after Transfer Learning
```
>> ex1_outlier_detection_after_TL.ipynb
```
- Check the uniformity of the pivot
```
>> ex2_validity_of_p_value.ipynb
```
