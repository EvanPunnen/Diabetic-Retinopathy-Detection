# Diabetic Retinopathy Detection - Performance Report

## Key Performance Metrics

### Overall Model Performance
- Overall Accuracy: 53.0%
- Weighted Average F1-Score: 0.54
- Macro Average F1-Score: 0.45
- Total Test Cases: 103 images

### Detailed Accuracy Metrics
1. **Classification Accuracy**:
   - Raw Accuracy: 53.0% (55/103 cases correctly classified)
   - Balanced Accuracy: 45.0% (accounting for class imbalance)

2. **Global Performance Scores**:
   - Weighted Average Precision: 0.56
   - Weighted Average Recall: 0.53
   - Weighted Average F1-Score: 0.54

3. **Macro-Averaged Metrics** (averaged across classes):
   - Macro Precision: 0.45
   - Macro Recall: 0.45
   - Macro F1-Score: 0.45

4. **Quadratic Weighted Kappa Score**: 
   - Score: 0.48 (moderate agreement)
   - Range: -1 to 1 (1 being perfect agreement)

## Dataset Information

### Training Data (iDRiD Dataset)
- Total Training Images: 413
- Distribution:
  * No DR (Grade 0): 134 images (32.4%)
  * Mild DR (Grade 1): 20 images (4.8%)
  * Moderate DR (Grade 2): 136 images (32.9%)
  * Severe DR (Grade 3): 74 images (17.9%)
  * Proliferative DR (Grade 4): 49 images (11.9%)

### Testing Data
- Total Test Images: 103
- Distribution:
  * No DR (Grade 0): 34 images (33.0%)
  * Mild DR (Grade 1): 5 images (4.9%)
  * Moderate DR (Grade 2): 32 images (31.1%)
  * Severe DR (Grade 3): 19 images (18.4%)
  * Proliferative DR (Grade 4): 13 images (12.6%)

## Model Architecture
- Base Model: DenseNet121
- Pre-training: ImageNet
- Output Classes: 5 (No DR to Proliferative DR)
- Training Parameters:
  * Batch Size: 8
  * Learning Rate: 3e-4
  * Optimizer: Adam
  * Number of Epochs: 100

## Training Performance

### Best Model Performance (Epoch 11)
Overall accuracy on validation set: 53%

Class-wise Performance:
1. No DR (Grade 0):
   - Accuracy: 70.59%
   - Precision: 0.67
   - Recall: 0.71
   - F1-score: 0.69

2. Mild DR (Grade 1):
   - Accuracy: 0.00%
   - Precision: 0.00
   - Recall: 0.00
   - F1-score: 0.00
   Note: Poor performance likely due to class imbalance and limited samples

3. Moderate DR (Grade 2):
   - Accuracy: 40.62%
   - Precision: 0.59
   - Recall: 0.41
   - F1-score: 0.48

4. Severe DR (Grade 3):
   - Accuracy: 52.63%
   - Precision: 0.50
   - Recall: 0.53
   - F1-score: 0.51

5. Proliferative DR (Grade 4):
   - Accuracy: 61.54%
   - Precision: 0.50
   - Recall: 0.62
   - F1-score: 0.55

## Prediction Confidence Analysis

### Average Confidence by Class:
1. No DR: High confidence (>70%) when correct

## Model Challenges and Limitations

### Class Imbalance Impact
1. **Mild DR (Grade 1)**:
   - Severe underperformance (0% accuracy)
   - Likely due to significant class imbalance (only 4.8% of training data)
   - Limited training samples (20 images) affecting model's ability to learn patterns

2. **Performance Disparity**:
   - Best performance on Grade 0 (No DR): 70.59% accuracy
   - Worst performance on Grade 1 (Mild DR): 0% accuracy
   - Moderate performance on higher grades (40-62% accuracy)

### Technical Limitations
1. **Data Distribution**:
   - Uneven class distribution affecting model's learning
   - Limited samples in certain classes (particularly Grade 1)
   - Potential bias towards majority classes

2. **Model Behavior**:
   - Tendency to favor majority classes
   - Difficulty in distinguishing subtle differences between grades
   - Lower confidence in borderline cases

### Areas for Improvement
1. **Data Augmentation**:
   - Implement more aggressive augmentation for minority classes
   - Consider synthetic data generation for Grade 1
   - Balance class weights in loss function

2. **Model Architecture**:
   - Experiment with alternative architectures
   - Consider ensemble approaches
   - Implement attention mechanisms for better feature focus

3. **Training Strategy**:
   - Implement curriculum learning
   - Use stratified k-fold cross-validation
   - Explore transfer learning from similar medical imaging tasks
2. Mild DR: Very low confidence, model struggles with this class
3. Moderate DR: Moderate confidence (40-60%)
4. Severe DR: Moderate to high confidence (50-70%)
5. Proliferative DR: Very high confidence (>90%) in clear cases

### Notable Predictions:
1. Highest Confidence Predictions:
   - IDRiD_014.jpg: 99.5% confidence (Proliferative DR)
   - IDRiD_015.jpg: 99.3% confidence (Proliferative DR)
   - IDRiD_005.jpg: 95.6% confidence (Proliferative DR)

2. Lowest Confidence Predictions:
   - IDRiD_004.jpg: 36.3% confidence (Moderate DR)
   - Multiple cases showing confusion between Moderate and Severe DR

### Cross-Grade Confusion Patterns:
1. Most Common Confusions:
   - Between Moderate and Severe DR
   - Between Severe and Proliferative DR
   - Almost no confusion between No DR and Proliferative DR

2. Confidence Distribution:
   - Very High (>90%): 4 cases
   - High (70-90%): 3 cases
   - Moderate (50-70%): 4 cases
   - Low (<50%): 2 cases

## Model Strengths
1. Excellent at identifying clear cases of Proliferative DR (>90% confidence)
2. Good at distinguishing No DR cases (70.59% accuracy)
3. High confidence in extreme cases (No DR and Proliferative DR)

## Model Limitations
1. Poor performance on Mild DR cases (likely due to class imbalance)
2. Moderate confusion between adjacent grades
3. Variable confidence in borderline cases

## Recommendations for Improvement
1. Address class imbalance:
   - Data augmentation for Mild DR class
   - Class-weighted loss function
   - Additional data collection for underrepresented classes

2. Model Refinement:
   - Fine-tune learning rate
   - Implement gradual unfreezing of layers
   - Experiment with other architectures

3. Data Quality:
   - Additional preprocessing steps
   - Quality-based filtering
   - Expert review of uncertain cases

## Conclusion
The model shows promising performance on extreme cases (No DR and Proliferative DR) but struggles with intermediate grades, particularly Mild DR. The overall accuracy of 53% on the validation set indicates room for improvement, particularly in handling class imbalance and distinguishing between adjacent DR grades.