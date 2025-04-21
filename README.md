---

##  Project Objective

To predict the **grade classification** of students based on multiple academic and socio-demographic factors using various supervised learning models.

---

## Notebooks & Methods

### `modeling.ipynb`
This notebook contains:

#### 1. **Problem Statement Definition**
- Predict student grades (classification problem).
- Based on features like study time, tutoring, parental support.

#### 2. **Hypothesis Identification**
- Students receiving additional tutoring tend to perform better.
- Parental support positively influences academic success.
- Students who study more hours per week achieve higher classifications.
- Participation in extracurricular activities may correlate with academic performance.
- Students with fewer disciplinary records perform better.

#### 3. **Data Preprocessing**
- Handling missing values.
- Encoding categorical features.
- Feature scaling and standardization.
- Outlier detection and treatment.

#### 4. **Exploratory Data Analysis**
- Univariate and bivariate visualizations.
- Correlation matrices.
- Class distribution and imbalance treatment (e.g., SMOTE).

#### 5. **Feature Engineering**
- Creation of new variables combining existing ones (e.g., "Study Support Score").

#### 6. **Modeling Techniques Used**
- **Logistic Regression** – baseline model.
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Deep Neural Networks (optional)** – for improved prediction performance.

#### 7. **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, ROC Curve

---

### `web_application.ipynb`
This notebook includes:

- Code for the Streamlit interface.
- UI for entering student data and predicting grade classification.
- Integration with the best-performing trained model.
- User-friendly results display.

---

##  Web Deployment

The app is hosted online using **Render**. It allows users (e.g., teachers or administrators) to input new student data and receive a predicted grade classification in real time.

---

##  Tools & Libraries Used

- Python: Pandas, Numpy, Scikit-learn, XGBoost, TensorFlow/Keras (optional)
- Visualization: Seaborn, Matplotlib, Plotly
- Web Framework: Streamlit
- Imbalanced Data Handling: SMOTE (Imbalanced-learn)

---

##  Documentation & Project Planning

This section outlines team roles and responsibilities as part of the assignment deliverables.


### Assignment Deliverables
-  **Modeling Notebook** (`modeling.ipynb`)
-  **Web Application Notebook** (`web_application.ipynb`)
-  **Source Code** (`src/`)
-  **Artifacts** (Trained Models, Predictions, Feature Importance)
-  **README File**
-  **Comprehensive PDF Report** with explanation of all findings
-  **Project Timeline & Task Distribution Plan**

---

##  Learnings

Through this project, we gained practical experience with the full machine learning workflow:
- Understanding and cleaning data
- Hypothesis testing
- Building and evaluating models
- Deploying a real-world application

---


