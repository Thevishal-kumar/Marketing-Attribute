# Marketing-Attribute
Marketing Campaign Response Prediction

This project analyzes and models customer responses to marketing campaigns using machine learning techniques. By leveraging classification algorithms, the system identifies key factors influencing customer behavior and helps optimize campaign targeting.

Features:
📈 Predictive Modeling: Logistic Regression, Decision Tree, and Random Forest classifiers.

🧹 Data Preprocessing: Feature encoding, normalization, missing value handling.

🧠 Model Evaluation: Confusion matrix, classification report, ROC AUC.

📊 Visualization: Feature importance, performance metrics, and comparison plots.

📤 Business Insight: Identifies high-value customer segments to enhance marketing ROI.

Tech Stack:
Programming Language: Python

Libraries & Tools:

pandas, numpy – data handling

scikit-learn – ML modeling

matplotlib, seaborn – visualizations

Jupyter Notebook – development environment

Project Structure:
marketing-ml/
│
├── marketingAttributes.py           # Core script for preprocessing and model building
├── 7.3. Code Example.ipynb          # Notebook with step-by-step implementation
├── data/
│   └── marketing_data.csv           # (Sample) Input dataset
├── outputs/
│   └── plots/                       # Model and metrics visualizations
└── README.md                        # Project documentation

Model Performance
| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | 82%      | 0.81      | 0.79   | 0.85    |
| Decision Tree       | 86%      | 0.84      | 0.83   | 0.88    |
| Random Forest       | 90%      | 0.89      | 0.87   | 0.92    |


How to Run:
1.Clone the repository:git clone https://github.com/Thevishal-kumar/Marketing-Attribute.git
                       cd marketing-ml

2.Install dependencies:
  pip install -r requirements.txt

3.Run the notebook or Python script:
  jupyter notebook 7.3. Code Example.ipynb
  # OR
  python marketingAttributes.py

Impact:
Improved customer targeting by identifying conversion-driving attributes.

Enhanced campaign effectiveness and reduced customer acquisition cost.

Demonstrated a full end-to-end ML pipeline for marketing decision support.





