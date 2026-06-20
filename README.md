
# Plant Leaf Disease Multilevel Classification Using Few-Shot Learning

### Final Year Project 

##  **Project Overview**

Plant diseases cause major crop losses every year, especially in rural areas where farmers may not have access to experts. Early and accurate disease detection can help prevent these losses. However:

* Most ML/DL models need **large datasets**,
* Rare diseases have **very few images**,
* Collecting and labeling thousands of samples is **not practical**.

To solve this, we built a **Few-Shot Learning–based multilevel plant disease classifier** that works even with **very limited training images per class**.

Our system first identifies the **plant type**, then predicts the **specific disease**, making the classification more accurate and practical.

We also created a **Flask-based web app** where users can upload a leaf image to get instant predictions.

##  **Why We Chose This Project**

* Agriculture is vital to India’s economy.
* Farmers struggle to identify plant diseases early.
* Expert help is not always available in remote areas.
* Existing deep learning models are not data-efficient.
* Few-Shot Learning works extremely well when **data is scarce**.

Our goal was to build a **real, workable solution** using modern machine learning techniques that can help farmers and agricultural communities.

##  **What Is Few-Shot Learning?**

Few-Shot Learning allows models to learn from **very few examples** (1–10 images per class) while still maintaining good accuracy.
This is perfect for plant diseases where large datasets are not available.

We used a **Hierarchical Prototypical Network**, which performs classification at two levels:

1. **Plant Type Classification**
2. **Disease Classification**

This multilevel structure reduces error propagation and boosts final accuracy.

##  **Architecture Overview**

###  **EfficientNet-B0 Backbone**

EfficientNet-B0 is a fast, lightweight, and highly accurate CNN model developed by Google.
We used it for:

* Feature extraction
* Embedding generation
* Similarity-based dataset cleaning

Its compound scaling method improves performance without increasing compute cost.

#  **Project Pipeline**

Below is a clear step-by-step explanation of the full workflow.

## **1. Data Collection & Image Acquisition**

* Images were collected from public datasets (PlantVillage) and our custom dataset.
* All images were organized class-wise for further processing.

## **2. Dataset Cleaning**

To ensure high-quality inputs:

* Removed duplicates
* Fixed mislabels
* Cleaned noisy images
* Verified class consistency

EfficientNet-B0 embeddings were used for similarity-based cleaning.

## **3. Few-Shot Split Creation**

We simulated Few-Shot tasks such as:

10 images per folder respectively

This prepares the model to classify diseases using **very few images** per class.

## **4. Preprocessing & Augmentation**

Applied transformations to improve generalization:

* Resizing
* Normalization
* Horizontal/vertical flips
* Random rotation
* Color jitter

This increases dataset diversity.

## **5. Hierarchical Prototypical Network Training**

Main model performs:

###  Level 1: Plant Type Prediction

(e.g., Raspberry)

###  Level 2: Disease Classification

(e.g., Leaf Spot, Healthy, Rust)

Model learns prototype vectors for each class and uses distance-based classification in embedding space.




## **6. EfficientNet-B0 + Projection Head**

* EfficientNet extracts features.
* Projection Head converts them into compact embeddings.
* Prototypes are generated for each class.

Distance → determines final prediction.


## **7. Model Deployment (Flask Web App)**

A user-friendly web interface was built using **Flask**:

* Users upload a leaf image
* The system returns:

  * **Plant Name**
  * **Disease Type**
* Includes **Login & Signup** system
* Fully deployable, real-world ready


## **Tech Stack**

### **Machine Learning**

* Python
* PyTorch / TensorFlow
* EfficientNet-B0
* Prototypical Networks
* Few-Shot Learning
* Numpy, Pandas, Scikit-Learn

### **Web Development**

* Flask
* HTML, CSS, JS
* SQLite / MySQL



#  **Key Features**

✔ Multilevel plant & disease classification
✔ Works with very small datasets
✔ Lightweight and fast
✔ High accuracy using Few-Shot training
✔ Clean, simple Flask web app
✔ Real-time predictions
✔ Secure login/signup feature



#  **Conclusion**

This project provides an end-to-end solution for plant disease detection using Few-Shot Learning. It combines:

* Machine learning innovation
* Real-world agricultural relevance
* A complete deployable system

The pipeline—from raw images to final predictions—shows how deep learning can solve practical problems even when data is limited.

