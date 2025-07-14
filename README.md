
# 💼 Salary Predictor – Machine Learning Web App

A machine learning-based web application that predicts employee salaries in USD based on job-related parameters using Random Forest Regression. Built with Streamlit and trained on a dataset of over 21,000 IT job roles.

🔗 **Live App**: [https://paypredictor.streamlit.app/](https://paypredictor.streamlit.app/)  
📦 **GitHub Repo**: [https://github.com/PrathmeshSose/salary-predictor](https://github.com/PrathmeshSose/salary-predictor)
   **Report** : [Salary_Prediction_Project.pdf](https://github.com/user-attachments/files/21220701/Salary_Prediction_Project.pdf)

---

## 📊 Project Overview

This project uses a machine learning model to predict salaries based on:

- Work year
- Experience level
- Employment type
- Job title
- Employee residence
- Remote work ratio
- Company location
- Company size

It uses a dataset sourced from a public GitHub repository and scholarly research, with no missing values and more than 21,000 job records.

---

## 📁 Dataset Information

- **Dataset Name:** IT Job Salary Overview  
- **Records:** 21,333  
- **Features:** 11  
- **Source:** [damalialutfiani/Salary-Dataset-Project](https://github.com/damalialutfiani/Salary-Dataset-Project)  
- **Top Job Title:** Data Scientist (4,996 records)  
- **Most Common Currency:** USD (19,912 records)  
- **Avg Salary (USD):** $151,037  
- **Range:** $15,000 to $800,000  
- **Missing Values:** None  

---

## ⚙️ Machine Learning Model

### ✅ Algorithms Explored

- Linear Regression
- Decision Tree Regressor
- **Random Forest Regressor** ✅
- Gradient Boosting (XGBoost/LightGBM)
- Support Vector Regression (SVR)
- Neural Networks (for large datasets)

### 📦 Libraries Used

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

## 🧠 Model Training Steps

1. Import necessary libraries  
2. Load and explore dataset (`final_salaries.csv`)  
3. Drop unnecessary columns (`salary`, `currency`)  
4. Encode categorical features with `OrdinalEncoder`  
5. Split data into training and test sets  
6. Train `RandomForestRegressor` model  
7. Evaluate using:
   - Mean Absolute Error (MAE)
   - R² Score
8. Save trained model using `joblib`

---

## 🖥️ Streamlit App Features

### 💡 Sections:
- **Dataset** – View raw job data and shapes
- **Graphs** – EDA: experience level, salary vs. remote, company size
- **Prediction** – Input parameters to get live salary prediction 💵

### 🌐 Deployment:
- Hosted on [Streamlit Cloud](https://streamlit.io/cloud)
- GitHub repo connected to auto-deploy on commit

---

## 🔧 How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/PrathmeshSose/salary-predictor
cd salary-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📂 Project Structure

```
salary-predictor/
├── app.py                # Streamlit application
├── train.py              # Model training script
├── final_salaries.csv    # Cleaned dataset
├── salary_model.pkl      # Trained model
├── requirements.txt      # Required Python packages
└── README.md             # Documentation file
```

---

## 🧪 Screenshots

### 📌 Dataset Page
_View of raw data and structure_

### 📈 Graphs Page
_EDA using seaborn and matplotlib_

### 🤖 Prediction Page
_Form input and real-time prediction result_

---

## 🌐 Deployment Instructions

1. Push files to GitHub repo  
2. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Connect GitHub and select your repo  
4. Choose `app.py` as the main file  
5. Click **Deploy**

---

## 📚 References

1. 📘 DELNET Research Portal – [https://delnet.in](https://delnet.in)  
2. 📄 Paper: *Salary Prediction Using Machine Learning*  
3. 📊 Dataset: [damalialutfiani/Salary-Dataset-Project](https://github.com/damalialutfiani/Salary-Dataset-Project)  
4. 🧰 Libraries: `pandas`, `sklearn`, `streamlit`, `joblib`, etc.  
5. 🌐 Streamlit Cloud: [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## ✅ Conclusion

This project built a Salary Prediction model using Random Forest Regression. It provides real-time salary estimates through an interactive web interface, demonstrating how ML can be applied to real-world business problems in HR and recruitment.

---

## 👤 Author

**Prathmesh Gajanan Sose**  
**Email:prathmeshsose93@gmail.com**

