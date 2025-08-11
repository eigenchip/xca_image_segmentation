# XCA Image Segmentation

## Theory
We segment 2D X-ray Coronary Angiography (XCA) images using a shallow MLP, guided by morphology-based preprocessing.

**Frangi vesselness (Hessian-based):**  
The Hessian of image intensity is
$$
H=\begin{pmatrix} I_{xx} & I_{xy} \\ I_{yx} & I_{yy} \end{pmatrix},
$$
with eigenvalues ordered $|\lambda_1|<|\lambda_2|$. Vesselness at scale $\sigma$ is
$$
V(\sigma)=
\begin{cases}
0, & \lambda_2>0 \\
\exp\!\left(-\frac{R^2}{2\alpha^2}\right)\!\left(1-\exp\!\left(-\frac{S^2}{2\beta^2}\right)\right), & \text{otherwise}
\end{cases}
$$
where $R=\tfrac{|\lambda_1|}{|\lambda_2|}$ distinguishes tubes vs. blobs, and $S=\sqrt{\lambda_1^2+\lambda_2^2}$ penalizes textured background. Multiscale selection picks
$$
V_\sigma(x,y)=\max_{\sigma} V(x,y,\sigma).
$$

**Adaptive thresholding (Otsu):**  
Maximize inter-class variance to pick $\tau^\*$:
$$
\tau^\*=\arg\max_{\tau}\; w_1(\tau)w_2(\tau)\,[\mu_1(\tau)-\mu_2(\tau)]^2.
$$

---

## Methodology
1. **Data augmentation:** horizontal flips.  
2. **Preprocessing:** multiscale Frangi ($\sigma\in[1.8,4.0]$; tuned $\alpha=1,\beta=1,\gamma=0.3$) + small-object removal (min size $\approx 5000$ px).  
3. **MLP (pixelwise):** input = single pixel intensity → 2 hidden layers (9 ReLU units each) → softmax (bg/fg).  
   Weighted BCE handles class imbalance:
   $$
   \mathcal{L}=-\frac{1}{N}\sum_{i=1}^N \big[w_f\,y_i\log p_i + w_b(1-y_i)\log(1-p_i)\big],\quad w_f=\tfrac{1}{f},\; w_b=\tfrac{1}{b}.
   $$
4. **Postprocessing:** Otsu on foreground probabilities to obtain final mask.  
5. **Validation:** 5-fold CV; separate hold-out test set.

---

## Datasets
- **DCA1** (X-ray Coronary Angiograms): 130 grayscale images (300×300) with expert vessel masks.  
  http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html

---

## Benchmarking
Metrics: AUROC, Dice $=\tfrac{2|A\cap B|}{|A|+|B|}$, IoU, Sensitivity, Specificity, Precision, and SNR of predicted foreground.

---

## Results
**Effect of preprocessing (hold-out averages):**

| Metric      | With Frangi+cleanup | Raw (no preprocessing) |
|-------------|----------------------|-------------------------|
| AUROC       | 0.9478               | 0.7597                  |
| Dice        | 0.61                 | 0.15                    |
| Sensitivity | 0.79                 | 0.21                    |
| Specificity | 0.95                 | 0.91                    |
| Precision   | 0.51                 | 0.15                    |
| IoU         | 0.44                 | 0.10                    |

**Takeaways:** preprocessing is decisive—improves Dice ×4, IoU ×4.4, AUROC +0.2; Otsu removes much of the residual noise but cannot restore lost connectivity (consider tree morphology/region growing).

---

## Tools
- **Python 3.10** on Google Colab  
- **NumPy, scikit-image, Matplotlib, PyTorch**  

---

## References
- Frangi, A. F., et al. *Multiscale Vessel Enhancement Filtering*. MICCAI, 1998.  
- Cervantes-Sánchez, F., et al. *Automatic Segmentation of Coronary Arteries in X-ray Angiograms using Multiscale Analysis and Artificial Neural Networks*. *Applied Sciences*, 2019.  
- DCA1 dataset: http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html
