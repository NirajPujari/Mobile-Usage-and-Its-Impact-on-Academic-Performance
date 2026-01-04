# 📊 Impact of Mobile Usage on Academic Performance

### A Data Science Research Project

## 🧾Abstract

This project is a **college research-oriented data science study** that analyzes the impact of mobile phone usage patterns on students’ academic performance. Using statistical analysis, clustering techniques, and machine learning models, the study aims to identify behavioral patterns, correlations, and predictive factors that influence academic outcomes.

The project follows a **complete data science pipeline**, including preprocessing, exploratory data analysis (EDA), unsupervised learning (clustering), and supervised learning (regression-based prediction).

## 🎯 Objectives

- To preprocess and convert raw survey data into a machine-readable numerical format
- To perform exploratory data analysis to understand feature distributions, correlations, and multicollinearity
- To identify student behavior groups using clustering techniques
- To predict academic performance using machine learning models
- To analyze which mobile usage factors most significantly impact academic results

## 🗂️ Project Structure

`    project/
    │── dataset/              # Raw and processed CSV files
    │── json/                 # Column mappings and configuration files
    │── src/
    │   ├── config.py         # Global constants and paths
    │   ├── util.py           # Utility functions (CSV/JSON loaders)
    │   ├── preprocessing.py # Data cleaning & encoding
    │   ├── eda.py            # Exploratory Data Analysis
    │   ├── clustering.py     # Unsupervised learning (KMeans, PCA)
    │   ├── prediction.py    # Supervised learning (Ridge Regression)
    │── main.py               # Pipeline execution entry point
    │── requirements.txt
    │── README.md
    │── run.bat`

## 🔬 Methodology

### 🧹 1. Data Preprocessing

- Column names standardized using a title mapping
- Categorical responses encoded numerically using predefined mappings
- Irrelevant fields (name, email, timestamp) removed
- Missing values handled using placeholder values

### 📊 2. Exploratory Data Analysis (EDA)

- Statistical summaries (mean, variance, skewness, kurtosis)
- Correlation analysis using heatmaps
- Multicollinearity detection using Variance Inflation Factor (VIF)
- Visualization using histograms and boxplots

### 🧩 3. Clustering (Unsupervised Learning)

- Feature scaling using StandardScaler
- Optimal number of clusters determined using Silhouette Score
- KMeans clustering applied
- PCA used for 2D visualization of clusters
- Hierarchical clustering visualized using dendrograms

### 🤖 4. Prediction (Supervised Learning)

- Ridge Regression used to predict academic performance
- Proper feature scaling applied via pipelines
- Model evaluated using:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Feature importance analyzed using regression coefficients

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **SciPy**
- **Statsmodels**

## ▶️ How to Run the Project

1. Use the git to clone the repo:

   ```bash
   git clone https://github.com/NirajPujari/Mobile-Usage-and-Its-Impact-on-Academic-Performance
   ```

2. Navigate to the project directory:

   ```bash
   cd Mobile-Usage-and-Its-Impact-on-Academic-Performance
   ```

3. Run the FastAPI application:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   or
   ```bash
   ./run
   ```

## 📈 Results & Observations

- The dataset consists of **92 student** responses with **14 numerical features**, representing demographic details, mobile usage behavior, and academic performance indicators.
- **Variance Inflation Factor (VIF)** analysis indicates no severe multicollinearity among the features.
  All VIF values remain below 4, suggesting that the selected variables contribute independently to the model and are suitable for regression analysis.
- **Clustering analysis using KMeans** identifies two distinct student groups (k = 2) based on mobile usage and study behavior.
  This suggests a clear behavioral separation, likely representing:
  - students with controlled and academic-focused mobile usage, and
  - students with higher or more disruptive mobile usage patterns.
- **Ridge Regression achieves moderate predictive performance**, with:
  - R² score ≈ 0.49, indicating that approximately 49% of the variance in academic performance is explained by mobile usage and study-related features,
  - RMSE ≈ 0.74, showing reasonable prediction accuracy given the limited dataset size.
- Features related to mobile usage duration, perceived impact of mobile phones on performance, and self-regulation behaviors (such as monitoring app usage and considering reduction of mobile usage) show stronger influence on academic outcomes.
- Academic performance is not driven by a single factor, but rather by a combination of usage intensity, behavioral awareness, and study habits, highlighting the complex relationship between mobile technology and student performance.

## 🎓 Academic Significance

This project was conducted as part of a college research initiative to:

- Apply theoretical data science concepts to real-world educational data
- Demonstrate practical usage of statistical analysis and machine learning
- Provide data-driven insights into student behavior and academic outcomes

## ⚠️ Limitations & 🔮 Future Scope

- Dataset size is limited to surveyed students
- Results are correlational, not causal
- Self-reported data may include bias
- Increase dataset size across multiple institutions
- Apply advanced models (Random Forest, XGBoost)
- Perform longitudinal analysis
- Integrate explainable AI techniques (SHAP, LIME)

## 📄License

MIT

## 👤Authors

- [@Niraj Pujari](https://github.com/NirajPujari)
