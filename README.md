# XCA Image Segmentation

## Theory
Goal: segment coronary vessels in 2D X-ray coronary angiography (XCA) using a shallow MLP aided by morphology-based preprocessing.

**Hessian & Frangi vesselness**
- Hessian (second-order spatial derivatives):
```math
\begin{bmatrix}X\\Y\end{bmatrix}
```
$H = \begin{bmatrix} I_{xx} & I_{xy} \\\ I_{yx} & I_{yy} \end{bmatrix}$
- Let $|\lambda_1| < |\lambda_2|$. Define $R = \frac{|\lambda_1|}{|\lambda_2|}$ and $S = \sqrt{\lambda_1^2 + \lambda_2^2}$.
- Vesselness at scale $\sigma$:
  $$V(\sigma) =
  \begin{cases}
  0 & \lambda_2 > 0 \\\
  \exp\!\left(-\frac{R^2}{2\alpha^2}\right)\!\left(1 - \exp\!\left(-\frac{S^2}{2\beta^2}\right)\right) & \text{otherwise}
  \end{cases}$$
- Multiscale selection:
  $$V_\sigma(x,y) = \max_{\sigma} V(x,y,\sigma)$$

**Adaptive thresholding (Otsu)**
$$\tau^* = \arg\max_{\tau}\; w_1(\tau)\,w_2(\tau)\,[\mu_1(\tau) - \mu_2(\tau)]^2$$

**MLP objective (weighted BCE)**
$$L = -\frac{1}{N}\sum_{i=1}^N \Big(w_f\,y_i\log p_i + w_b\,(1-y_i)\log(1-p_i)\Big), \quad
w_f = \tfrac{1}{f},\; w_b = \tfrac{1}{b}$$

---

## Methodology
1. **Augment**: horizontal flips.  
2. **Preprocess**: multiscale Frangi ($\sigma \in [1.8,4.0]$), then small-object removal (min size ≈ 5000 px).  
3. **Classify (MLP)**: input = single pixel intensity; 2 hidden layers × 9 neurons (ReLU); softmax output; weighted BCE.  
4. **Segment**: apply Otsu on MLP foreground probabilities to produce the binary mask.

---

## Datasets
- **DCA1**: 130 grayscale XCA images, $300\times300$, expert ground truths.  
  Source: Cardiology Dept., Mexican Social Security Institute (UMAE T1-León).

---

## Benchmarking
Metrics: **AUROC**, **Dice** ($\displaystyle \frac{2|A\cap B|}{|A|+|B|}$), **IoU**, **Sensitivity**, **Specificity**, **Precision**, and foreground **SNR** ($\mu_{\text{fgd}}/\sigma_{\text{fgd}}$).

---

## Results
| Metric       | Filtered | Unfiltered |
|--------------|----------|------------|
| AUROC        | 0.948    | 0.760      |
| Dice         | 0.61     | 0.15       |
| Sensitivity  | 0.79     | 0.21       |
| Specificity  | 0.95     | 0.91       |
| Precision    | 0.51     | 0.15       |
| IoU          | 0.44     | 0.10       |

**Takeaways**
- Multiscale Frangi + small-object removal **substantially boosts** MLP segmentation quality.
- Otsu removes residual specks but cannot restore broken vessel connectivity (consider tree morphology / region growing in postprocessing).

---

## Tools
- Python 3.10 (Google Colab)
- NumPy, scikit-image, Matplotlib
- PyTorch (MLP)

---

## References
1. Frangi et al., *Multiscale Vessel Enhancement Filtering*, MICCAI (1998).  
2. Cervantes-Sanchez et al., *Automatic Segmentation of Coronary Arteries in X-ray Angiograms using Multiscale Analysis and Artificial Neural Networks*, Applied Sciences (2019).  
3. DCA1: *Angiogram Image Database* (Cruz-Aceves).  
