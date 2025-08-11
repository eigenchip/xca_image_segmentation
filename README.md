# X-ray Coronary Artery Image Segmentation

A lightweight approach for automatic segmentation of 2D X-ray coronary artery images using shallow multi-layer perceptrons and morphological filtering.

## Theory

### Hessian Matrix
The Hessian matrix stores second-order spatial derivatives of pixel intensity values for detecting tubular structures:

```math
H = \begin{bmatrix}
I_{xx} & I_{xy} \\
I_{yx} & I_{yy}
\end{bmatrix}
```

Where `I_xx` and `I_yy` are second-order partial derivatives along x and y axes, and `I_xy = I_yx` is the mixed partial derivative.

### Frangi Vesselness Filter
The Frangi filter enhances narrow, elongated structures while suppressing blobs and background:

```math
V(\sigma) = \begin{cases}
0, & \text{if } \lambda_2 > 0 \\
f(\gamma) \exp\left(-\frac{R^2}{2\alpha^2}\right) \left[1 - \exp\left(-\frac{S^2}{2\beta^2}\right)\right], & \text{otherwise}
\end{cases}
```

Where:
- ```math R = \frac{|\lambda_1|}{|\lambda_2|}``` differentiates between blobs and tubes
- ```math S = \sqrt{\lambda_1^2 + \lambda_2^2}``` measures combined eigenvalue magnitude
- `α` controls blob structure penalization
- `β` controls plate-like structure penalization
- `γ` modulates sensitivity to background texture

### Multi-Layer Perceptron
A shallow neural network with:
- Single input neuron (pixel intensity)
- 2 hidden layers with 9 neurons each
- ReLU activation for hidden layers
- Softmax output activation
- Weighted binary cross-entropy loss to handle class imbalance

### Otsu's Adaptive Thresholding
Finds optimal threshold by maximizing inter-class variance:

```math
\sigma^2(\tau) = w_1(\tau)w_2(\tau)[\mu_1(\tau) - \mu_2(\tau)]^2
```

Where `w_1`, `w_2` are class probabilities and `μ_1`, `μ_2` are class means.

## Methodology

1. **Data Augmentation**: Horizontal flipping for robustness
2. **Preprocessing**: 
   - Frangi multiscale filtering (σ ∈ {1.8, 1.9, ..., 4.0})
   - Small-object removal (5000 pixel threshold) to eliminate artifacts
3. **Training**: 5-fold cross-validation with 212 training samples
4. **Segmentation**: Otsu's method applied to predicted foreground probabilities

## Dataset

**DCA1 Database**: 130 grayscale X-ray coronary angiograms (300×300 pixels) with expert-labeled ground truth from the Mexican Social Security Institute Cardiology Department.

- Training: 212 samples
- Validation: 56 samples (hold-out)
- Class imbalance: Vessel pixels comprise 15-19% of total pixels

## Benchmarking

### Performance Metrics
- **AUROC**: 0.9478 (filtered) vs 0.7597 (unfiltered)
- **Dice Coefficient**: 0.61 (filtered) vs 0.15 (unfiltered) 
- **Sensitivity**: 0.79 (filtered) vs 0.21 (unfiltered)
- **Specificity**: 0.95 (filtered) vs 0.91 (unfiltered)
- **IoU**: 0.44 (filtered) vs 0.10 (unfiltered)

### Key Findings
- Preprocessing improved Dice score by 4x and sensitivity by 3.8x
- Otsu thresholds: 0.10-0.14 (filtered) vs 10⁻³-10⁻² (unfiltered)
- Performance comparable to Cervantes et al. with much simpler architecture

## Tools

- **Python 3.10.12** on Google Colab
- **Hardware**: Intel Core i5-1135G7 (2.40GHz, 16GB RAM)
- **Libraries**: scikit-image for morphological operations
- **Optimization**: Adam optimizer with 0.001 learning rate

## References

1. Young, I. (1983). Image analysis and mathematical morphology. *Cytometry*, 4, 184-185.
2. Frangi, A. et al. (2000). Multiscale Vessel Enhancement Filtering. *Medical Image Computing and Computer-Assisted Intervention*, 1496.
3. Luo, Y. & Sun, L. (2023). Digital subtraction angiography image segmentation based on multiscale Hessian matrix. *Journal of Radiation Research and Applied Sciences*, 16(3).
4. Ma, G., Yang, J., & Zhao, H. (2020). A coronary artery segmentation method based on region growing. *Technology and Health Care*, 28, S463-S472.
5. Cervantes-Sanchez, F. et al. (2019). Automatic Segmentation of Coronary Arteries using Multiscale Analysis and Artificial Neural Networks. *Applied Sciences*, 9(24), 5507.
6. Park, T. et al. (2022). Deep Learning Segmentation in 2D X-ray Images. *Diagnostics*, 12(4), 778.
7. Iyer, K. et al. (2021). AngioNet: a convolutional neural network for vessel segmentation. *Scientific Reports*, 11, 18066.

---

**Code Repository**: https://github.com/eigenchip/xca_image_segmentation  
**License**: MIT
