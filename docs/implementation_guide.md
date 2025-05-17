# Trash Classification Project: Implementation Guide

This document provides a detailed explanation of the trash classification project implementation. The code creates a comprehensive machine learning application that categorizes waste images into different classes such as can, glass, paper, plastic, and residual waste.

## Project Structure

The implementation follows a structured approach addressing all requirements:

1. **Data Preprocessing**
   - Data exploration and visualization
   - Data cleaning
   - Data augmentation
   - Train/validation/test split

2. **Model Selection and Training**
   - Custom CNN architecture with regularization
   - Transfer learning with MobileNetV2
   - Hyperparameter tuning

3. **Model Evaluation**
   - Accuracy and loss analysis
   - Confusion matrix visualization
   - Classification report
   - Model comparison

4. **Model Explanation**
   - GradCAM visualization to show important features

## Detailed Implementation

### 1. Data Preprocessing

**Loading the Dataset**

The code assumes a directory structure where data is organized into train, validation, and test sets, with subdirectories for each class:

```
waste_data/
  train/
    can/
    glass/
    paper/
    plastic/
    residual/
  valid/
    can/
    glass/
    paper/
    plastic/
    residual/
  test/
    can/
    glass/
    paper/
    plastic/
    residual/
```

**Data Exploration**

The script performs exploratory data analysis by:
- Counting samples per class to check class balance
- Visualizing sample images from each class
- Displaying class distribution

**Data Augmentation**

To increase dataset diversity and prevent overfitting, we apply multiple augmentation techniques:
- Rotation (±20°)
- Width/height shifts (±20%)
- Shear transformation (20%)
- Zoom (±20%)
- Horizontal flipping

The code visualizes the effects of these augmentations on sample images to demonstrate their impact.

### 2. Model Selection and Training

**Model 1: Custom CNN Architecture**

We implement a custom CNN with:
- 3 convolutional blocks (increasing filter counts: 32→64→128)
- Max pooling after each block
- Dropout layers (25%) for regularization
- L2 regularization on convolutional and dense layers
- Final dense layer with softmax activation

**Model 2: Transfer Learning (MobileNetV2)**

We leverage a pre-trained MobileNetV2 model:
- Trained on ImageNet dataset with weights frozen
- Global average pooling after base model
- Custom classifier head with 512 neurons
- Dropout (50%) for regularization
- Final layer with softmax activation

**Training Strategy**

Both models are trained with:
- Adam optimizer with learning rate 0.001
- Categorical cross-entropy loss
- Early stopping to prevent overfitting (patience=5)
- Learning rate reduction on plateau (patience=3)
- Model checkpointing to save best weights

### 3. Model Evaluation

**Performance Metrics**

We evaluate models using:
- Accuracy on test set
- Confusion matrix visualization
- Detailed classification report with precision, recall, and F1-score

**Overfitting Analysis**

The training history plots help identify overfitting by comparing:
- Training vs. validation accuracy
- Training vs. validation loss

If the training metrics significantly outperform validation metrics, this indicates overfitting.

**Regularization Techniques**

To combat overfitting, we implement:
- L2 regularization on weights
- Dropout layers
- Data augmentation
- Early stopping
- Learning rate reduction

### 4. Model Explanation with GradCAM

Gradient-weighted Class Activation Mapping (GradCAM) visualizes what the model "looks at" when making predictions:

1. We select the last convolutional layer of the model
2. Calculate gradients of the predicted class with respect to the feature maps
3. Generate a heatmap highlighting important regions
4. Overlay this heatmap on the original image

This visualization shows which parts of the image most influenced the classification decision, making the model more interpretable.

## Running the Project

To run this project:

1. Ensure you have all required dependencies installed:
   ```
   pip install tensorflow opencv-python matplotlib seaborn pandas sklearn
   ```

2. Organize your dataset according to the structure mentioned above

3. Update the `DATA_DIR` variable to point to your dataset location

4. Execute the script, which will:
   - Process and explore data
   - Train both models
   - Evaluate performance
   - Generate explanatory visualizations

## Expected Outputs

The code generates various visualizations saved in the `figures/` directory:
- Class distribution
- Sample images
- Data augmentation examples
- Training history for both models
- Confusion matrices
- Model comparison
- GradCAM visualizations

Model weights are saved in the `models/` directory for future use or deployment.

## Next Steps and Extensions

Possible improvements to this project:
- Fine-tuning the transfer learning model by unfreezing layers
- Testing additional architectures (ResNet, EfficientNet)
- Implementing cross-validation for more robust evaluation
- Developing a real-time classification application
- Deploying the model to a web or mobile application