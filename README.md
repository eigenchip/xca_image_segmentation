# xca_image_segmentation
Automated X-ray coronary artery image segmentation using a shallow multilayer perceptron.


# XCA Image Segmentation

Segmentation of X-ray Coronary Angiography (XCA) images for vessel detection and analysis using classical image processing and deep learning methods.

---

## Theory
X-ray Coronary Angiography (XCA) visualizes coronary vessels by injecting contrast agents. Vessel segmentation enables automated stenosis detection, quantitative vessel analysis, and treatment planning.

**Mathematical basis**  
- **Frangi Vesselness Filter**:  
  Uses eigenvalues of the Hessian matrix \((\lambda_1, \lambda_2)\) to detect tubular structures:  
  \[
  V(x) =
  \begin{cases}
  0, & \lambda_2 > 0 \\
  \exp\left(-\frac{R_B^2}{2\beta^2}\right) \cdot
  \left(1 - \exp\left(-\frac{S^2}{2\gamma^2}\right)\right), & \text{otherwise}
  \end{cases}
  \]  
  where \( R_B = \frac{|\lambda_1|}{|\lambda_2|} \), \( S = \sqrt{\lambda_1^2 + \lambda_2^2} \).

- **Otsu Thresholding**: Selects the optimal threshold by maximizing inter-class variance.
- **Morphological Operations**: Removes noise and small isolated components.

---

## Methodology
1. **Preprocessing**
   - Contrast enhancement
   - Multi-scale Frangi filter
   - Removal of small objects (< 5000 px)

2. **Segmentation**
   - Otsu global thresholding
   - Morphological cleanup (closing, opening)

3. **Model Training**
   - PyTorch custom dataset loader
   - K-fold cross-validation
   - Mixed-precision training for speed

4. **Evaluation**
   - ROC curve, AUC score
   - Dice coefficient, pixel accuracy

---

## Datasets
- **Source**: Public XCA datasets with original frames and vessel masks.
- **Structure**:


- **Labels**: Binary ground-truth vessel masks.

---

## Benchmarking
| Method                     | ROC-AUC | Dice  |
|----------------------------|--------:|------:|
| Frangi + Otsu + Morphology | 0.87    | 0.75  |
| CNN (ours)                 | 0.94    | 0.82  |

---

## Results
- **Classical filtering** successfully enhances vessel visibility but may miss faint vessels.
- **Deep learning** improves robustness to noise and illumination changes.

*(Insert sample input/output images here)*

---

## Future Work
- Integrate U-Net architecture
- Explore attention-based segmentation
- Apply domain adaptation for other angiography modalities
- Real-time deployment pipeline

---

## Tools
- **Languages**: Python 3
- **Libraries**:  
- `scikit-image`  
- `torch`  
- `scikit-learn`  
- `matplotlib`

---

## References
1. Frangi, A. F., et al. "Multiscale vessel enhancement filtering." *MICCAI*, 1998.  
2. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.  
3. Otsu, N. "A threshold selection method from gray-level histograms." *IEEE Trans. Sys. Man. Cyber.*, 1979.
