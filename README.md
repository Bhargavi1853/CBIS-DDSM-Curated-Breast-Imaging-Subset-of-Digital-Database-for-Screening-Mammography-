# CBIS-DDSM-Curated-Breast-Imaging-Subset-of-Digital-Database-for-Screening-Mammography
# Dual-view deep learning for mammogram classification

Benign vs malignant mammogram classification on the CBIS-DDSM dataset. I compared single-view, dual-view and multi-task CNNs in PyTorch to test one question: does looking at both standard mammogram views together (CC and MLO) classify better than looking at each view on its own?

The best result came from an ensemble of all four cropped-image models: test AUC 0.818 and 75.8% accuracy on a patient-level held-out test set of 231 CC/MLO pairs.

Artificial Intelligence & Machine Vision, MSc Artificial Intelligence, University of East London (2026).


## Why two views

Radiologists check every finding on two images of the same breast: the craniocaudal (CC) view from above and the mediolateral oblique (MLO) view from the side. A lesion that shows up on both is real; one that shows on only one may be overlapping tissue. Most automated classifiers look at one image at a time and lose that check. This project builds models that see both.

## Results

### Cropped-image pipeline (lesion crops, patient-level split)

All models were trained with five-fold patient-grouped cross-validation and bagged, then scored once on the same held-out test set (231 pairs).

| Model | AUC | Accuracy | Precision | Sensitivity | Specificity | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| CC only (ResNet50) | 0.790 | 0.706 | 0.584 | 0.759 | 0.674 | 0.660 |
| MLO only (ResNet50) | 0.794 | 0.758 | 0.660 | 0.736 | 0.771 | 0.696 |
| Dual-view cross-attention | 0.796 | 0.732 | 0.634 | 0.678 | 0.764 | 0.656 |
| Dual-view multi-task | 0.785 | 0.710 | 0.647 | 0.506 | 0.833 | 0.568 |
| Weighted ensemble | **0.818** | **0.758** | **0.670** | 0.701 | 0.792 | **0.685** |

### Full-mammogram pipeline (whole images)

| Model | Validation accuracy | Test accuracy |
| --- | --- | --- |
| Shallow CNN (random init) | 0.607 | 0.591 |
| AlexNet (ImageNet-pretrained, SGD with momentum) | 0.657 | 0.649 |

With AlexNet fixed, I also compared four optimisers over 5 epochs. SGD with momentum did best (65.5% validation accuracy), then Adam (64.4%), SGD (63.3%) and RMSProp (60.9%).

<img width="590" height="1175" alt="Sample Images" src="https://github.com/user-attachments/assets/0292f61d-0ced-41cf-b229-b2844ae0f9f1" />


### What I learned

- Pretraining helps. ImageNet weights added nearly 6 points of test accuracy over training from scratch, even though mammograms look nothing like ImageNet photos.
- Dual-view fusion gave the best single-model AUC, but only just (0.796 vs 0.794 for MLO only). The multi-task heads didn't help at this data size.
- The ensemble beat every individual model. All four landed in a narrow 0.785 to 0.796 AUC band, so they were learning similar but not identical things, and averaging them gained more than any architecture change did.
- The models overfit. Training AUC kept rising while validation AUC stalled, which points to the small dataset (1,080 training pairs) rather than tuning as the limit.
- Grad-CAM showed the dual-view model focusing on the lesion rather than the breast edge or scanner labels.

## Data

[CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (Lee et al., 2017), the curated subset of the Digital Database for Screening Mammography. Benign and benign-without-callback cases were merged into one Benign class.

| Dataset | Benign | Malignant | Total |
| --- | --- | --- | --- |
| Full mammograms | 5,876 | 4,257 | 10,133 images |
| Paired CC/MLO lesion crops | 907 | 626 | 1,533 pairs |
| Single-view lesion crops (for the CC-only and MLO-only baselines) | | | 3,199 images |

Splits are 70/15/15. The full-mammogram pipeline splits at image level, stratified by class. The cropped pipeline splits at patient level, so no patient appears in more than one split (1,080 train, 222 validation, 231 test pairs). The test set was never used for model selection, threshold tuning or bagging.

## Models

Full-mammogram pipeline:

- Shallow CNN: three conv, batch norm, ReLU and max-pool blocks (16, 32, 64 channels), then a fully connected head with dropout 0.4. Trained from scratch as a lower bound.
- AlexNet: ImageNet-pretrained, fine-tuned end to end with a new two-class output layer.

Cropped-image pipeline:

- Single-view ResNet50: ImageNet-pretrained, one model for CC and one for MLO, with horizontal-flip test-time augmentation.
- Dual-view cross-attention: two weight-shared ResNet50 encoders, one per view. A bidirectional cross-attention block (scaled dot-product, query/key/value) lets each view's features attend to the other's. The original and attended features are concatenated and passed to a two-layer head (256 units, dropout 0.5).
- Dual-view multi-task: the same network with three extra heads predicting breast density, lesion subtlety and abnormality type (mass or calcification). Loss weights: density 0.20, subtlety 0.20, abnormality type 0.12.
- Weighted ensemble: averages the probabilities of all four cropped-image models, each weighted by its validation AUC. The decision threshold was tuned separately on validation data.

<img width="576" height="1134" alt="Mammography-Modeling-root" src="https://github.com/user-attachments/assets/405863da-8de1-4975-b747-a0f040b14d94" />

## Training

- Full-mammogram images resized to 224 x 224, ImageNet normalisation, augmentation with horizontal flips, rotation up to 10 degrees and mild brightness and contrast jitter.
- Lesion crops processed with CLAHE (clip limit 2.0, 8 x 8 grid) to bring out local contrast.
- Class imbalance handled with inverse-frequency class weights, plus focal loss (gamma 2.0, label smoothing 0.05) and a weighted random sampler on the cropped pipeline.
- Shallow CNN and AlexNet: SGD with momentum 0.9, cosine annealing, early stopping (patience 5).
- Dual-view models: AdamW with a two-phase schedule. 2 warm-up epochs with the encoder frozen, then up to 30 epochs with ResNet50 layers 2 to 4 unfrozen (learning rate 1e-5 for the encoder, 3e-4 for the head). Gradient clipping at 1.0, mixed precision, and a plateau scheduler on validation AUC.
- Five-fold patient-grouped stratified cross-validation (StratifiedGroupKFold), with the five fold models bagged at test time.
- Seed 42 throughout. Trained on a single Google Colab T4 GPU.

## What's in this repo

| File | What it is |
| --- | --- |
| `MammogramClassification_CBIS_DDSM.ipynb` | Full-mammogram pipeline: shallow CNN, fine-tuned AlexNet and the optimiser comparison |
| `CBIS_DDSM_DualView.ipynb` | Cropped-image pipeline: single-view ResNet50, dual-view cross-attention, multi-task model, ensemble and Grad-CAM |
| `MammographyReport.pdf` | Full write-up with methods, results and discussion |
| `Mammography_Presentation.pptx` | Presentation slides |
| `Mammography-Modeling-root.png` | Overall project pipeline |

## Running it

Both notebooks are built for Google Colab.

1. Open a notebook in Colab and set the runtime to a T4 GPU.
2. Get a Kaggle API token (Kaggle > Settings > Create New Token) and upload `kaggle.json` when the notebook asks for it.
3. Run the cells in order. The notebook downloads the dataset itself with:

```bash
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset
```

To run outside Colab, install the dependencies first:

```bash
pip install torch torchvision scikit-learn pandas numpy opencv-python matplotlib seaborn pillow grad-cam kaggle
```

## Limitations

- Small data. 1,080 training pairs is little for deep learning, and the train/validation gap shows it.
- ImageNet pretraining only. Weights pretrained on mammograms would likely do better.
- Two classes. Merging benign and benign-without-callback loses whether a benign finding needed follow-up.
- One dataset from a set of US institutions. Nothing here has been tested on external data.
- The cropped pipeline assumes the lesion has already been found. Real screening starts from the whole image, which is why the full-mammogram scores are lower.

This is a coursework research project. It is not a medical device and must not be used for diagnosis.

## Next steps

- Pretrain on a larger mammography dataset before fine-tuning.
- Test on an external dataset.
- Combine the two pipelines: detect candidate lesions on the full image, then classify each one with the dual-view model.

## References

- Lee, R. S. et al. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Krizhevsky, A., Sutskever, I. and Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. *ICCV*.
- Selvaraju, R. R. et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.

## Author

Bala Bhargavi Godari · [LinkedIn](https://www.linkedin.com/in/godaribalabhargavi) · [GitHub](https://github.com/Bhargavi1853)
