# Lung Cancer Classification – Lightweight CNN & Transformers  

This repo contains **data and code** for our ICCIT 2024 paper:  
**“Custom Lightweight CNN and Data-Efficient Models for Efficient and Fast Convergence in Lung Cancer Classification.”**

## 🔑 Highlights  
- Custom **Lightweight CNN** achieved **99.71% test accuracy**.  
- **DeiT Transformer** reached **99.89% test accuracy in only 7 epochs**.  
- Compared against ConvNext, InceptionV3, ResNet50, VGG16, DenseNet121, EfficientFormer.  

## 📊 Dataset  
We used the publicly available dataset: **IQ-OTH/NCCD**  
- Contains CT-scan images of **benign, malignant, and normal** lung nodules.  
- Preprocessed with normalization and augmentations.  
- Used for **training, validation, and testing**.  

## 📊 Results  

| Model             | Train Acc | Test Acc | Epochs |
|-------------------|-----------|----------|--------|
| **ConvNext**      | 0.99429   | 0.99543  | 50     |
| **InceptionV3**   | 0.99771   | 0.99885  | 50     |
| **ResNet50**      | 0.99201   | 0.93151  | 50     |
| **VGG16**         | 0.99657   | 0.99772  | 50     |
| **DenseNet121**   | 0.99543   | 0.99657  | 50     |
| **Lightweight CNN** | 0.99901 | 0.99712  | 50     |
| **EfficientFormer** | 0.97830 | 0.96430  | 50     |
| **DeiT**          | 0.99999   | **0.99890** | **7** |

## ⚙️ Usage  
```bash
git clone https://github.com/yourusername/lung-cancer-classification.git
cd lung-cancer-classification
pip install -r requirements.txt

# Train models
python scripts/train_cnn.py
python scripts/train_transformer.py

# Evaluate
python scripts/evaluate.py --model custom_cnn

## 📄 Citation

@INPROCEEDINGS{11022575,
  author={Bashir, Mariam Binte and Rahi, Abu Mukaddim and Jahan, Md. Khurshid and Al Shafi, Abdullah},
  booktitle={2024 27th International Conference on Computer and Information Technology (ICCIT)}, 
  title={Custom Lightweight CNN and Data-Efficient Models for Efficient and Fast Convergence in Lung Cancer Classification}, 
  year={2024},
  pages={477-481},
  keywords={Deep learning;Analytical models;Accuracy;Sensitivity;Lung cancer;Transformers;Convolutional neural networks;Convergence;Testing;Residual neural networks;Lung Cancer;DeiT;CNN;Machine Learning;Medical Imaging},
  doi={10.1109/ICCIT64611.2024.11022575}}
