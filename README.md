# XCA Image Segmentation

## Theory
This project targets **2D X-ray Coronary Angiography (XCA) image segmentation** using a shallow Multi-Layer Perceptron (MLP) aided by morphological preprocessing.

**Frangi Vesselness Filter**  
- Relies on the **Hessian matrix** of second-order intensity derivatives:
  \[
  H =
  \begin{pmatrix}
  I_{xx} & I_{xy} \\
  I_{yx} & I_{yy}
  \end{pmatrix}
  \]
- Eigenvalues \(|\lambda_1| < |\lambda_2|\) characterize curvature; \(\lambda_2 > 0\) implies non-vessel pixels.
- Vesselness:
  \[
  V(\sigma) =
  \begin{cases}
  0 & \lambda_2 > 0 \\
  e^{-\frac{R^2}{2\alpha^2}}\left( 1 - e^{-\frac{S^2}{2\beta^2}} \right) & \text{otherwise}
  \end{cases}
  \]
  with \( R = |\lambda_1|/|\lambda_2| \) and \( S = \sqrt{\lambda_1^2 + \lambda_2^2} \).

**Multiscale approach**:  
\[
V_\sigma(x,y) = \max_{\sigma} V(x,y,\sigma)
\]
selects the optimal vessel width per pixel.

**Segmentation** uses **Otsu's adaptive threshold**:
\[
\tau^* = \arg\max_{\tau} \; w_1(\tau) w_2(\tau) \left[ \mu_1(\tau) - \mu_2(\tau) \right]^2
\]
maximizing inter-class variance between background and vessel classes.

---

## Methodology
1. **Data Augmentation** – Horizontal flips to improve robustness.
2. **Preprocessing**:
   - Frangi vesselness filter at multiple \(\sigma\) values (1.8–4.0).
   - Small-object removal (min size = 5000 px) to suppress artifacts.
3. **MLP Architecture**:
   - Input: single pixel intensity
   - 2 hidden layers, 9 neurons each (ReLU activation)
   - Output: softmax over background/foreground
   - Loss: Weighted Binary Cross-Entropy to address class imbalance
4. **Postprocessing**:
   - Otsu thresholding of MLP foreground probabilities for final segmentation.

---

## Datasets
- **DCA1 – Database of X-ray Coronary Angiograms**  
  - 130 grayscale images, 300×300 px  
  - Expert-annotated ground truths  
  - [Dataset link](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html)

---

## Benchmarking
Metrics:
- **AUROC** – Area under ROC curve
- **Dice coefficient** – Overlap measure robust to class imbalance:
  \[
  \text{Dice} = \frac{2|A \cap B|}{|A| + |B|}
  \]
- **IoU (Jaccard Index)**
- **Sensitivity**, **Specificity**, **Precision**
- **SNR** of predicted foreground

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

**Key findings**:
- Preprocessing improves Dice by ×4, IoU by ×4.4, and AUROC by +0.2.
- Frangi parameters critically affect vessel enhancement vs. noise amplification.
- Otsu effectively removes residual noise but cannot restore vessel connectivity.

---

## Tools
- **Language**: Python 3.10
- **Libraries**: NumPy, scikit-image, Matplotlib, PyTorch
- **Environment**: Google Colab

---

## References
1. Frangi, A. F., et al. "Multiscale Vessel Enhancement Filtering." MICCAI 1998.  
2. Cervantes-Sanchez, F., et al. "Automatic Segmentation of Coronary Arteries in X-ray Angiograms using Multiscale Analysis and Artificial Neural Networks." Applied Sciences, 2019.  
3. [DCA1 Dataset](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html)
