# Brain Tumor Detection – Multi-Algorithm ML/DL Classification

MRI brain tumor classification using 5 ML/DL algorithms with full comparative evaluation.

## Model Results
| Model | Accuracy |
|-------|----------|
| CNN2D (Custom Architecture) | 99% |
| VGG16 (Transfer Learning) | 83% |
| Random Forest | 81% |
| SVM (RBF Kernel) | 77% |
| KNN | 74% |

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset
- 3,000+ MRI images | 4 classes: Glioma, Meningioma, No Tumor, Pituitary
- Split: 80% training / 20% testing (stratified)

## Tech Stack
Python | TensorFlow | Keras | OpenCV | Scikit-learn | NumPy | Pandas | Matplotlib

## Pipeline
- Image loading, resizing to 128×128, pixel normalization
- Stratified 80/20 train-test split
- 5 model training and benchmarking
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Custom predict() function for inference on unseen MRI images

## How to Run
pip install tensorflow keras opencv-python scikit-learn numpy pandas matplotlib seaborn
python TumorDisease.ipynb
