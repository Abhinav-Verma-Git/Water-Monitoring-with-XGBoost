<!-- README.md for Water Potability Prediction Project -->

<h1 align="center">💧 Water Potability Prediction using XGBoost</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue?logo=python">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg">
  <img src="https://img.shields.io/badge/XGBoost-%20v1.0+-orange">
</p>

---

## 📋 Project Overview

This project predicts the potability (drinkability) of water using several physical and chemical features, leveraging advanced machine learning techniques. Key steps include data exploration, visualization, and hyperparameter tuning of an XGBoost regression model.

---

## 🧰 Tools & Libraries Used

<ul>
  <li><b>Pandas</b> - Data manipulation and analysis</li>
  <li><b>Numpy</b> - Numerical computations</li>
  <li><b>Matplotlib</b> & <b>Seaborn</b> - Visualization</li>
  <li><b>scikit-learn</b> - Data splitting, metrics, and hyperparameter tuning</li>
  <li><b>XGBoost</b> - Gradient boosting regression</li>
</ul>

---

## 🚀 Workflow

<ol>
  <li><b>Data Loading & Inspection:</b> Load and display the <code>water_potability.csv</code> dataset, view descriptive statistics.</li>
  <li><b>Exploratory Analysis:</b> Plot histograms for each feature and a correlation heatmap for insight into feature relationships.</li>
  <li><b>Preprocessing:</b> Split the data into <code>features</code> and <code>target</code> (potability), then into training and test sets.</li>
  <li><b>Model Selection & Tuning:</b> Use <code>GridSearchCV</code> for hyperparameter tuning of <b>XGBoostRegressor</b> (testing n_estimators, max_depth, learning_rate).</li>
  <li><b>Evaluation:</b> Predict on the test set and evaluate results using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.</li>
</ol>

---


---

## 📈 Results

- **Best Hyperparameters:** Grid search finds the optimal set for n_estimators, max_depth, and learning_rate.
- **Evaluation Metrics:** The script reports MSE, RMSE, and R² on the test data.

---

## 🗃️ Dataset

Include `water_potability.csv` in your project directory. The dataset should have columns such as:
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability

---

## 🧑‍💻 How to Run

1. Clone this repository.
2. Place `water_potability.csv` in your project folder.
3. Install dependencies:
    ```
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```
4. Run the script in your preferred environment (e.g. Jupyter Notebook, VSCode, etc.)

---

## 📚 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.



