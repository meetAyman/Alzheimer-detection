# Alzheimer Detection with CNN and Attention MIL

A deep learning project for **patient-level Demented / Non Demented classification** from MRI images. A shared convolutional neural network learns image features, gated attention combines them, and an MLP produces one prediction per patient.

**Explore the project:** [Alzheimer-detection.ipynb](Alzheimer-detection.ipynb)

The notebook contains the complete implementation and executed results: preprocessing, model architecture, five-fold hyperparameter search, final training, ROC curves, confusion matrices and classification reports.

## Recorded results

| Metric | Result |
|---|---:|
| Mean five-fold validation ROC AUC | **0.9013** |
| Test ROC AUC | **0.9362** |
| Test accuracy | **88.57%** |
| Test balanced accuracy | **87.50%** |
| Demented-class precision | **66.67%** |
| Demented-class recall | **85.71%** |
| Demented-class F1 score | **0.7500** |

On **70 test patients**, the selected model correctly classifies **50 of 56 Non Demented patients** and detects **12 of 14 Demented patients**, using a classification threshold of **0.5**. The test ROC AUC has a **95% stratified bootstrap interval of [0.8686, 0.9847]**, calculated from 2,000 patient resamples.

The selected hyperparameters are **dropout 0.1** and **Adam learning rate 0.001**, chosen by comparing six configurations across five folds.

## From images to a patient prediction

Each patient contributes many MRI images, but has one class label. Multiple instance learning (MIL) treats these images as a **patient bag** and learns how to combine their information.

```text
Patient's MRI images
        |
Shared CNN: three convolution blocks (32, 64, 128 filters)
        |
Global average pooling: 128 features per image
        |
Gated attention: learned image weights
        |
MIL pooling: attention-weighted patient representation
        |
MLP: Dense 64 -> ReLU -> Dropout -> Dense 1 -> Sigmoid
        |
One patient-level classification score
```

Each convolution block uses a **3 x 3 convolution**, same padding, ReLU and **2 x 2 max pooling**. The same CNN processes every image through `TimeDistributed`.

Attention uses two **64-unit branches**, with tanh and sigmoid activations. Their outputs are multiplied and mapped to one score per image. Softmax converts these scores into weights that sum to one within each patient. A weighted sum then produces the **128-dimensional patient vector** used by the MLP.

The model accepts bags with different numbers of images and contains **117,570 parameters**.

## Dataset and preprocessing

The experiment uses **86,437 images from 347 patients**. Patients contribute between **122 and 488 images**, with a median of **244**.

| Split | Non Demented | Demented | Total patients |
|---|---:|---:|---:|
| Training | 210 | 67 | 277 |
| Test | 56 | 14 | 70 |

Patient identifiers are extracted from filename prefixes such as `OAS1_0001`. All images and acquisitions from the same patient remain together in training, validation and test splits. The loader also checks that each patient has a consistent label.

Preprocessing follows three steps:

1. **Verify dimensions:** check the original image headers, which are height **248 x width 496**.
2. **Convert and resize:** use grayscale images of **124 x 248**, preserving the aspect ratio while reducing the pixel count to one-quarter of the original.
3. **Standardize each image:** subtract its mean and divide by its adjusted standard deviation.

```text
standardized pixel = (pixel - image mean) / adjusted standard deviation
adjusted standard deviation = max(image standard deviation, 1 / sqrt(number of pixels))
```

The pipeline uses these standardized images directly, without data augmentation.

## Training and model selection

### 1. Learn image features

Within each cross-validation fold, the CNN is pretrained with a temporary image classifier using images labeled by their patient's class. Sample weights give each patient equal total influence before balancing the two classes.

Validation image predictions are averaged within each patient. Exact patient-level ROC AUC selects the best CNN checkpoint.

### 2. Learn patient aggregation

The CNN is frozen and its image features are cached. Gated attention and the MLP are then trained on patient bags using **class-weighted binary cross-entropy**.

| Hyperparameter | Candidates |
|---|---|
| Dropout | 0, 0.1, 0.3 |
| Adam learning rate | 0.0003, 0.001 |
| Validation | Five stratified patient folds |

All six configurations are evaluated on all five folds, giving **30 attention/MLP fits**. Cached features avoid repeating CNN inference for each candidate. Training uses best-checkpoint saving, early stopping and learning-rate reduction, guided by exact validation ROC AUC.

The configuration with the highest mean validation AUC is selected, with validation loss used to break ties.

### 3. Fit and evaluate the final model

The selected model is trained on all **277 training patients**, using the fold-median best epoch counts: **4 CNN pretraining epochs** and **2 attention/MLP epochs** in the recorded experiment.

The complete model is saved, then evaluated at patient level. The notebook displays training and test metrics, a test ROC curve, both confusion matrices and both classification reports.

## Run the notebook

To browse the code and recorded outputs, open [Alzheimer-detection.ipynb](Alzheimer-detection.ipynb) directly on GitHub.

To execute it locally, use **Python 3.10**. The recorded experiment uses TensorFlow **2.10.0**. Install the dependencies in your Python environment:

```powershell
python -m pip install tensorflow-cpu==2.10.0 numpy==1.23.5
python -m pip install pandas==2.3.3 scikit-learn==1.7.2 Pillow==12.3.0
python -m pip install matplotlib==3.10.9 h5py==3.16.0 ipykernel==7.3.0
python -m ipykernel install --user --name alzheimer-detection --display-name "Alzheimer Detection (Python 3.10)"
```

For the Windows DirectML GPU environment used in this experiment:

```powershell
python -m pip install tensorflow-directml-plugin==0.4.0.dev230202
```

1. Open the notebook in VS Code or Jupyter and select your Python environment's kernel.
2. Set `binary_path` in the settings cell to your dataset folder, containing the **Non Demented** and **Demented** directories.
3. Run the cells in order. The first code cell reports whether TensorFlow detects a GPU.

On a first run, the notebook trains the CNNs, performs the full hyperparameter search, fits the final model and saves its outputs locally.

### Reuse the saved experiment

Keep `save_dir = Path('alzheimer_saved')` to load the saved model, cross-validation scores and patient predictions on later runs. This preserves the recorded results without repeating training. Fixed seeds also control the patient split and training randomness.

To train a separate experiment, change this one setting, for example:

```python
save_dir = Path('alzheimer_saved_02')
```

## Repository contents

```text
Alzheimer-detection.ipynb    Complete pipeline with explanations and executed results
README.md                   Project overview and setup
```
