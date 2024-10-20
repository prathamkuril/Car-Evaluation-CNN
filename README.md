# Car Evaluation to predict safety of a car using Convolutional Neural Networks (CNN) with PyTorch and Skorch

## Project Overview

This project demonstrates the application of a Convolutional Neural Network (CNN) for car evaluation classification based on categorical attributes. The dataset includes features like `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`, and a target variable `class`. Key processes include data preprocessing, handling class imbalance using oversampling, and training the CNN model using PyTorch and Skorch.

### Key Components:
- **Data Preprocessing**: Includes categorical encoding using `category_encoders.OrdinalEncoder` and class balancing with oversampling.
- **Neural Network Model**: A CNN model built with PyTorch for classification.
- **Hyperparameter Tuning**: Grid search performed using Skorch and `GridSearchCV`.
- **Visualization**: Plotly is used for visualizing training metrics, confusion matrix, and classification reports.

## Dataset

The dataset (`car.csv`) contains 1728 rows and 7 columns, with the following features:
- `buying`: Buying price of the car.
- `maint`: Maintenance cost of the car.
- `doors`: Number of doors.
- `persons`: Number of persons the car can hold.
- `lug_boot`: Size of luggage boot.
- `safety`: Safety rating of the car.
- `class`: Target variable representing the car evaluation.

## Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Key Libraries:
- `pandas`, `numpy` for data manipulation.
- `category_encoders` for encoding categorical variables.
- `imblearn` for handling class imbalance.
- `torch`, `skorch` for neural network modeling and training.
- `matplotlib`, `seaborn`, `plotly` for data visualization.
- `scikit-learn` for model selection, preprocessing, and evaluation metrics.

### Installing Dependencies

The dependencies can be installed using:

```bash
pip install pandas numpy matplotlib seaborn category_encoders torch skorch plotly scikit-learn imbalanced-learn joblib
```

## Project Structure

- **`car.csv`**: Dataset file used for car evaluation classification.
- **`CarEvaluation_CNN.ipynb`**: Jupyter notebook containing the complete project code.
- **`best_trained_modelCNN.pkl`**: Saved model after training and hyperparameter tuning.
- **`X_test_processed.csv`** & **`y_test_processed.csv`**: Processed test data.
- **`label_encoder.pkl`**: Saved label encoder for target variable encoding.

## Data Preprocessing

1. **Encoding**: The categorical variables are encoded using `OrdinalEncoder` from `category_encoders`.
2. **Oversampling**: We address class imbalance using `RandomOverSampler` from `imbalanced-learn`.
3. **Tensor Conversion**: Data is converted to PyTorch tensors for training the CNN.

## CNN Model Architecture

The CNN model consists of:
- **2 Convolutional Layers**: Extracting features from the input data.
- **2 Fully Connected Layers**: Classifying the cars into 4 possible evaluation categories (`unacc`, `acc`, `good`, `vgood`).

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 1, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
```

## Hyperparameter Tuning

The model is optimized using GridSearchCV to find the best combination of learning rate (`lr`), batch size, and the number of epochs. Cross-validation with StratifiedKFold ensures that the model's performance is consistent across different folds.

```python
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'max_epochs': [10, 20, 30]
}
```

## Model Training

The CNN model is trained for multiple epochs, and the training accuracy, test accuracy, and training time per epoch are recorded and visualized using Plotly.

## Model Evaluation

### Evaluation Metrics:
- **Accuracy**: Accuracy on the test set.
- **Confusion Matrix**: To visualize the model’s classification performance.
- **ROC-AUC Score**: To evaluate the overall classification capability.
- **Classification Report**: Detailed precision, recall, and F1-score for each class.

### Example Results:
- **Accuracy**: `1.0`
- **Confusion Matrix**: A 4x4 matrix showing the classification performance for each class.
- **Average Training Time per Epoch**: ~7.9 seconds.

## Visualizations

The following visualizations are generated using Plotly:
- **Class Distribution Before and After Oversampling**.
- **Confusion Matrix**.
- **Training and Test Accuracy**.
- **Training Time per Epoch**.
- **Classification Report Heatmap**.

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/your-username/Car-Evaluation-CNN.git
```

2. Navigate to the project directory:
```bash
cd Car-Evaluation-CNN
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter notebook or Python script to train and evaluate the CNN model.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Skorch Documentation](https://skorch.readthedocs.io/en/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Plotly Documentation](https://plotly.com/python/)

---
