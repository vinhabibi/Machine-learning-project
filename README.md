# Machine-learning-project
College Basketball Tournament Prediction Project
💡Overview.
This project uses machine learning to predict postseason outcomes for college basketball teams based on the cbb.csv dataset. The dataset includes features like team performance metrics (e.g., ADJOE, ADJDE, BARTHAG) from 2016–2019 seasons.
🔑Key components:

Data Exploration: Dataset information, descriptive statistics, missing values handling, and visualizations (histograms, boxplots, PCA).
Preprocessing: Label encoding for categorical columns, MinMax scaling for features, and handling NaN values.
⚙Model Training & Evaluation: Trains and evaluates multiple classifiers (K-Nearest Neighbors, Decision Tree, Support Vector Machine, Logistic Regression) using accuracy, confusion matrices, and classification reports.
📊Visualizations: Confusion matrix plots for each model, histograms, boxplots with swarm plots, and PCA 2D scatter plot.
Model Ranking: Ranks models based on 5-fold cross-validation accuracy.

The target variable is POSTSEASON, encoded numerically for classification.
Dataset

Source: cbb.csv (College Basketball Dataset, from Kaggle )
Columns: 24 features including TEAM, CONF, G (games), W (wins), ADJOE (adjusted offensive efficiency), ADJDE (adjusted defensive efficiency), BARTHAG (power rating), and more.
Rows: 1406 entries.
Target: POSTSEASON (e.g., Champions, Final Four, etc.), with many missing values filled or handled during preprocessing.

Sample data preview:



TEAM
CONF
G
W
ADJOE
ADJDE
BARTHAG
...
POSTSEASON
SEED
YEAR



North Carolina
ACC
40
33
123.3
94.9
0.9531
...
2ND
1.0
2016


Villanova
BE
40
35
123.1
90.9
0.9703
...
Champions
2.0
2016


Requirements

Python 3.x
Libraries:
pandas
scikit-learn (for models, preprocessing, metrics)
matplotlib
seaborn
numpy



Install dependencies:
pip install pandas scikit-learn matplotlib seaborn numpy

Usage

Clone the repository:
git clone https://github.com/vinhabibi/college-basketball-ml.git
cd college-basketball-ml


Place cbb.csv in the project root (download from Kaggle or similar).

Run the Jupyter Notebook:
jupyter notebook seaborn.ipynb


Execute cells sequentially for data loading, exploration, model training, visualizations, and ranking.



Models Used

K-Nearest Neighbors (KNN): n_neighbors=5
Decision Tree: random_state=42
Support Vector Machine (SVM): kernel='rbf', random_state=42
Logistic Regression: max_iter=1000, random_state=42

Evaluation Metrics:

Accuracy
Confusion Matrix (visualized)
Classification Report (precision, recall, F1-score)

Results
Sample Model Ranking (based on cross-validation):

Rank 1: Decision Tree - Accuracy: 0.8830
Rank 2: SVM - Accuracy: 0.8582
Rank 3: Logistic Regression - Accuracy: 0.8440
Rank 4: K-Nearest Neighbors - Accuracy: 0.8191

Actual results may vary slightly due to random splits.
Visualizations

Histograms: Distribution of features post-preprocessing.
Boxplots with Swarm: Outlier detection and data spread.
PCA Scatter: 2D projection of dataset for clustering insights.
Confusion Matrices: Heatmaps for each model's predictions.

Example Confusion Matrix (from notebook): 
Contributing
Feel free to fork and submit pull requests for improvements, such as hyperparameter tuning or additional models.
License
This project is licensed under the MIT License.
