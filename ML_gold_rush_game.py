import streamlit as st
import random
from collections import defaultdict
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Machine Learning - Gold Rush Study Guide",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Question bank (MCQs; correct answer is always first) ---
question_bank = [
    {
        "text": "Which the most common plot used for initial exploration numerical variables distribution and outliers?",
        "options": ["Boxplot", "Scatter plot", "Bar chart", "Line chart"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "It is how spread or dispersed a set of data points are from their mean...",
        "options": ["Variance", "Standard deviation", "Mean absolute deviation", "Range"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which Python library is commonly used for data manipulation and EDA?",
        "options": ["pandas", "matplotlib", "numpy", "scikit-learn"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "What is the primary goal of Exploratory Data Analysis (EDA)?",
        "options": ["Uncover patterns and anomalies", "Train a predictive model", "Deploy a web app", "Optimize hyperparameters"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which pandas' function shows basic descriptive statistics of numeric columns?",
        "options": ["df.describe()", "df.info()", "df.head()", "df.tail()"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },

    {
        "text": "This learning type uses labeled data to train models",
        "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Evolutionary Learning"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "This learning type discovers hidden structures in unlabeled data",
        "options": ["Unsupervised Learning", "Supervised Learning", "Reinforcement Learning", "Semi-supervised Learning"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "This type of learning involves an agent interacting with an environment to maximize reward",
        "options": ["Reinforcement Learning", "Supervised Learning", "Unsupervised Learning", "Transfer Learning"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which algorithm is an example of unsupervised dimensionality reduction?",
        "options": ["PCA", "Logistic Regression", "KNN", "Decision Tree"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which of the following algorithms is not supervised?",
        "options": ["Clustering", "Classification", "Regression", "Time-series forecasting"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which of the following algorithms is used for binary classification by estimating P(Y|X)?",
        "options": ["Logistic Regression", "K-means", "PCA", "Naive Bayes"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which classic classifier uses the nearest training examples in feature space?",
        "options": ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Naive Bayes"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "This classifier builds a series of rules based on feature splits:",
        "options": ["Decision Tree", "Logistic Regression", "KNN", "SVM"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "This model combines multiple intentional weak learners into a strong learner via aggregation:",
        "options": ["Random Forest", "Decision Tree", "Linear Regression", "KNN"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which algorithm seeks a hyperplane maximizing class separation (margin separation)?",
        "options": ["Support Vector Machine", "Logistic Regression", "Decision Tree", "Naive Bayes"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which splitting metric in decision trees measures class purity?",
        "options": ["Gini Impurity", "Euclidean Distance", "Manhattan Distance", "Cosine Similarity"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method in Random Forest reduces correlation between trees?",
        "options": ["Feature subsetting", "Pruning", "Gradient boosting", "Normalization"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which process removes branches that have little predictive power and prevents overfitting?",
        "options": ["Pruning", "Bagging", "Boosting", "Ensembling"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which algorithm suffers less from overfitting thanks to averaging?",
        "options": ["Random Forest", "Single Decision Tree", "K-means", "PCA"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which criterion is used by CART (Classification and Regression Tree) to split nodes?",
        "options": ["Gini Impurity", "Information Gain Ratio", "Chi-Square", "Gain Ratio"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which hyperparameter in SVM controls trade-off between margin width and classification error?",
        "options": ["C", "Gamma", "Kernel", "Degree"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which kernel implicitly maps data into infinite-dimensional space?",
        "options": ["RBF", "Linear", "Polynomial", "Sigmoid"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which concept in SVM defines the distance between parallel hyperplanes?",
        "options": ["Margin", "Support Vector", "Kernel", "Bias"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which SVM variant is used for regression tasks?",
        "options": ["SVR", "SVC", "One-Class SVM", "KNN"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which kernel parameter in RBF (Radial Basis Function) SVM influences influence radius of data points?",
        "options": ["Gamma", "C", "Degree", "Coef0"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },

    {
        "text": "Which algorithm predicts based on the majority vote of nearest neighbors?",
        "options": ["KNN", "Decision Tree", "SVM", "Naive Bayes"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which K (in terms of magnitude) in KNN tends to reduce noise but may oversmooth boundaries?",
        "options": ["Large K", "Small K", "K=1", "K=0"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which distance metric is commonly used in KNN?",
        "options": ["Euclidean Distance", "Cosine Similarity", "Jaccard Index", "Mahalanobis Distance"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which KNN variant weights neighbors by distance rather than uniform voting?",
        "options": ["Distance-weighted KNN", "Weighted Decision Tree", "Bagged KNN", "Random Forest"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which method can reduce computational cost in KNN searches?",
        "options": ["KD-Tree", "Pruning", "PCA", "Grid Search"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which scenario indicates high bias in a model?",
        "options": ["Underfitting", "Overfitting", "Perfect fit", "High variance"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which scenario indicates high variance in a model?",
        "options": ["Overfitting", "Underfitting", "High bias", "Low complexity"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which technique can help reduce overfitting by combining models?",
        "options": ["Ensembling", "PCA", "Standardization", "Feature selection"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which regularization adds a penalty on the absolute value of coefficients?",
        "options": ["L1 Regularization", "L2 Regularization", "Dropout", "Early Stopping"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which graph helps visualize the bias–variance trade-off as model complexity increases?",
        "options": ["Error vs. Complexity Curve", "ROC Curve", "Precision-Recall Curve", "Confusion Matrix"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },

    {
        "text": "Which term describes strong linear relationships between features?",
        "options": ["Multicollinearity", "Heteroscedasticity", "Sample Bias", "Stationarity"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which problem can cause unstable coefficient estimates in regression?",
        "options": ["Multicollinearity", "Overfitting", "Underfitting", "Imbalanced classes"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which diagnostics are used to detect multicollinearity?",
        "options": ["Variance Inflation Factor", "AIC", "BIC", "Cross-Validation Score"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which action can mitigate multicollinearity?",
        "options": ["Remove or combine features", "Increase tree depth", "Increase K in KNN", "Use more epochs"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which regression algorithm is most sensitive to multicollinearity?",
        "options": ["Linear Regression", "KNN", "Decision Tree", "Naive Bayes"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which algorithm groups data by minimizing within-cluster variance?",
        "options": ["K-means", "Hierarchical Clustering", "DBSCAN", "PCA"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which clustering method does not require specifying the number of clusters a priori?",
        "options": ["Hierarchical Clustering", "K-means", "PCA", "Logistic Regression"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "In K-means, which statistic measures cluster compactness?",
        "options": ["Within-Cluster Sum of Squares", "Silhouette Score", "AUC", "Cross-Entropy"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method visually helps choose K by looking for a sharp bend?",
        "options": ["Elbow Method", "Silhouette Analysis", "Gap Statistic", "Dendrogram"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which distance measure is used in the standard elbow method plot?",
        "options": ["Sum of squared distances", "Manhattan distance", "Cosine similarity", "Mahalanobis distance"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which technique reduces features by projecting onto orthogonal components?",
        "options": ["PCA", "LDA", "K-means", "Hierarchical Clustering"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which concept does PCA use to measure the information captured by each component?",
        "options": ["Variance", "Entropy", "Gini", "Covariance"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which visual shows data projected onto the first two principal components?",
        "options": ["2D PCA scatter plot", "ROC curve", "Decision boundary plot", "Boxplot"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Before PCA, why must you standardize your features?",
        "options": ["To ensure equal variance contribution", "To remove outliers", "To encode categorical data", "To reduce dimensionality"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which metric indicates how much total variance is explained by selected components?",
        "options": ["Explained Variance Ratio", "Silhouette Score", "AUC", "Rand Index"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },

    {
        "text": "Which metric counts correctly classified instances over all instances?",
        "options": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which metric is most informative on imbalanced datasets?",
        "options": ["F1 Score", "Accuracy", "Mean Squared Error", "Log Loss"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which metric measures the proportion of true positives among predicted positives?",
        "options": ["Precision", "Recall", "Accuracy", "AUC"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which metric measures the proportion of actual positives correctly identified?",
        "options": ["Recall", "Precision", "Accuracy", "F1 Score"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which score is the harmonic mean of precision and recall?",
        "options": ["F1 Score", "F2 Score", "ROC AUC", "Log Loss"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },

    {
        "text": "Which curve plots TPR vs FPR across thresholds?",
        "options": ["ROC Curve", "Precision-Recall Curve", "Calibration Curve", "Lift Chart"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which statistic is the area under the ROC curve?",
        "options": ["AUC", "Accuracy", "F1 Score", "Log Loss"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "On an ROC plot, what does the diagonal line represent?",
        "options": ["Random performance", "Perfect classifier", "No false positives", "Max precision"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which threshold selection method visualizes cost vs threshold on the ROC?",
        "options": ["Operating point analysis", "Elbow method", "Silhouette analysis", "Grid search"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which concept describes the probability the model ranks a positive above a negative?",
        "options": ["AUC interpretation", "Precision", "Recall", "Accuracy"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which matrix shows TP, FP, FN, TN counts?",
        "options": ["Confusion Matrix", "Covariance Matrix", "Correlation Matrix", "Distance Matrix"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "In a confusion matrix, FP refers to:",
        "options": ["False Positive", "False Negative", "True Positive", "True Negative"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "In a binary confusion matrix, which cell counts missed positives?",
        "options": ["False Negatives", "False Positives", "True Negatives", "True Positives"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which metric can be derived directly from a confusion matrix?",
        "options": ["Recall", "PCA", "Elbow Value", "Kernel Function"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which plot visualizes a confusion matrix as colored cells?",
        "options": ["Heatmap", "Scatter", "Boxplot", "Bar chart"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },

    {
        "text": "Which method splits data into k folds to estimate performance?",
        "options": ["k-Fold Cross-Validation", "Train-Test Split", "Bootstrap", "Grid Search"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which CV variant preserves class distribution in each fold?",
        "options": ["Stratified k-Fold", "Leave-One-Out", "K-means", "Bootstrapping"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which cross-validation uses each sample once as a test set?",
        "options": ["Leave-One-Out", "5-Fold", "Hold-Out", "Bootstrap"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which practice prevents data leakage in cross-validation?",
        "options": ["Perform preprocessing inside each fold", "Shuffle after splitting", "Impute before splitting", "Combine train/test first"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which metric should you report across folds to show variability?",
        "options": ["Mean ± Standard Deviation", "Median", "Max Score", "Min Score"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which Python library is primarily used for creating interactive visualizations in notebooks?",
        "options": ["Plotly", "NumPy", "Pandas", "scikit-learn"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method would you use to detect and remove duplicate rows in a DataFrame?",
        "options": ["df.drop_duplicates()", "df.dropna()", "df.duplicated()", "df.remove()"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which supervised algorithm can be used for multi-class classification with a one-vs-rest approach?",
        "options": ["Logistic Regression", "K-means", "PCA", "Apriori"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which unsupervised method can identify anomalous data points in high-dimensional data?",
        "options": ["DBSCAN", "Linear Regression", "AdaBoost", "Naive Bayes"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which technique reduces high-dimensional text data into vector features?",
        "options": ["TF-IDF vectorization", "One-hot encoding", "Standardization", "PCA"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which classifier assumes feature independence given the class label?",
        "options": ["Naive Bayes", "Random Forest", "SVM", "KNN"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which algorithm builds new trees sequentially to correct errors of previous trees?",
        "options": ["Gradient Boosting", "Random Forest", "K-means", "PCA"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which decision tree parameter controls the minimum samples required to split an internal node?",
        "options": ["min_samples_split", "max_depth", "n_estimators", "learning_rate"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which forest variant uses extremely randomized splits at each node?",
        "options": ["Extra Trees", "Bagging", "AdaBoost", "Gradient Boosting"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which hyperparameter in decision trees prevents overfitting by limiting depth?",
        "options": ["max_depth", "min_samples_leaf", "criterion", "splitter"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which SVM parameter adds a constant term in the kernel function?",
        "options": ["coef0", "C", "gamma", "degree"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which technique transforms non-linear data to a higher dimension for SVM separation?",
        "options": ["Kernel Trick", "Bagging", "Pruning", "Ensembling"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which regression uses a logistic function to model binary outcomes?",
        "options": ["Logistic Regression", "Linear Regression", "Ridge Regression", "Lasso Regression"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which neural network layer type connects every input neuron to every output neuron?",
        "options": ["Dense (Fully Connected)", "Convolutional", "Pooling", "Recurrent"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which regularization technique randomly zeroes out neurons during training?",
        "options": ["Dropout", "L1", "L2", "Batch Normalization"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which concept describes when adding features decreases test error but increases training error?",
        "options": ["Underfitting", "Overfitting", "Normalizing", "Standardizing"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which metric will not detect overfitting if both train and test accuracies are high?",
        "options": ["Log Loss", "Accuracy", "Recall", "Precision"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which transformation can mitigate skewness in feature distributions?",
        "options": ["Log transformation", "Min-Max scaling", "Standardization", "PCA"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method can detect multicollinearity by examining eigenvalues of X'X?",
        "options": ["Condition Number", "VIF", "AUC", "MSE"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which clustering algorithm can discover clusters of arbitrary shape?",
        "options": ["DBSCAN", "K-means", "PCA", "Hierarchical (complete)"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method orders clusters into a hierarchy without pre-specifying K?",
        "options": ["Agglomerative Clustering", "K-means", "PCA", "DBSCAN"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which approach uses silhouette coefficient to choose the number of clusters?",
        "options": ["Silhouette Analysis", "Elbow Method", "Gap Statistic", "AIC/BIC"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which PCA variant incorporates supervised class labels?",
        "options": ["Linear Discriminant Analysis", "Standard PCA", "Kernel PCA", "Sparse PCA"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which PCA component captures the second-highest variance orthogonal to the first?",
        "options": ["Second principal component", "First principal component", "Third principal component", "Eigenvector"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which metric penalizes confident but wrong predictions more heavily?",
        "options": ["Log Loss", "Accuracy", "Precision", "Recall"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which curve shows trade-off between precision and recall at various thresholds?",
        "options": ["Precision-Recall Curve", "ROC Curve", "Lift Chart", "Calibration Curve"],
        "difficulty": "easy",
        "reward_range": [100, 300]
    },
    {
        "text": "Which area metric is preferred when positive class is rare?",
        "options": ["PR AUC", "ROC AUC", "Accuracy", "F1 Score"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which matrix is normalized by dividing each cell by its row total?",
        "options": ["Normalized Confusion Matrix", "Covariance Matrix", "Distance Matrix", "Correlation Matrix"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    },
    {
        "text": "Which normalization makes each confusion matrix row sum to one?",
        "options": ["Recall normalization", "Precision normalization", "Accuracy normalization", "F1 normalization"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which method randomly samples with replacement to estimate model error?",
        "options": ["Bootstrap", "k-Fold CV", "LOO CV", "Hold-out Split"],
        "difficulty": "medium",
        "reward_range": [200, 400]
    },
    {
        "text": "Which cross-validation is best when time order matters?",
        "options": ["Time Series Split", "Stratified k-Fold", "Leave-One-Out", "Bootstrap"],
        "difficulty": "hard",
        "reward_range": [300, 500]
    }
    ]

# Color coding for difficulty levels
DIFFICULTY_COLORS = {
    "easy": "#4CAF50",    # Green
    "medium": "#FF9800",  # Orange
    "hard": "#F44336"     # Red
}

# --- Initialize session state ---
def init_session_state():
    if "teams" not in st.session_state:
        st.session_state.teams = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    if "winner" not in st.session_state:
        st.session_state.winner = None
    if "game_active" not in st.session_state:
        st.session_state.game_active = False
    if "remaining_questions" not in st.session_state:
        st.session_state.remaining_questions = len(question_bank)
    if "selected_questions" not in st.session_state:
        st.session_state.selected_questions = []

def reset_game():
    st.session_state.teams = []
    st.session_state.current_question = 0
    st.session_state.question_history = []
    st.session_state.winner = None
    st.session_state.game_active = False
    st.session_state.remaining_questions = len(question_bank)
    st.session_state.selected_questions = []

# Initialize state
init_session_state()

# --- Setup Teams & Game Control ---
st.sidebar.title("🎮 Game Control")

# Start a new game
if not st.session_state.game_active:
    st.sidebar.header("Setup New Game")
    num_teams = st.sidebar.number_input("Number of teams", min_value=2, max_value=6, value=3)
    initial_budget = st.sidebar.number_input("Initial budget ($)", min_value=100, value=1000)
    
    # Custom team names input
    use_custom_names = st.sidebar.checkbox("Use custom team names")
    team_names = []
    
    if use_custom_names:
        for i in range(num_teams):
            team_names.append(st.sidebar.text_input(f"Team {i+1} name", value=f"Team {i+1}"))
    else:
        team_names = [f"Team {i+1}" for i in range(num_teams)]
    
    questions_to_use = st.sidebar.slider("Number of questions", min_value=5, max_value=len(question_bank), value=len(question_bank))
    
    if st.sidebar.button("Start Game"):
        # Randomly select questions from the question bank
        # This ensures we get a different set of questions each time
        st.session_state.selected_questions = random.sample(question_bank, questions_to_use)
        
        st.session_state.teams = [
            {"name": name, "budget": initial_budget, "correct": 0, "incorrect": 0} 
            for name in team_names
        ]
        st.session_state.remaining_questions = questions_to_use
        st.session_state.game_active = True
        st.rerun()
else:
    # Reset game button when a game is active
    if st.sidebar.button("Reset Game"):
        reset_game()
        st.rerun()

    # Display game stats
    st.sidebar.header("Game Stats")
    st.sidebar.markdown(f"**Questions**: {st.session_state.current_question}/{st.session_state.remaining_questions}")
    
    # Display team stats in sidebar
    st.sidebar.header("Team Stats")
    
    # Create dataframe for team statistics
    stats_data = []
    for team in st.session_state.teams:
        stats_data.append({
            "Team": team["name"],
            "Budget": team["budget"],
            "Correct": team.get("correct", 0),
            "Incorrect": team.get("incorrect", 0)
        })
    
    stats_df = pd.DataFrame(stats_data)
    # Handle older versions of Streamlit that don't support use_container_width
    try:
        st.sidebar.dataframe(stats_df, use_container_width=True)
    except TypeError:
        st.sidebar.dataframe(stats_df, width=300)

# --- Main Game Area ---
# Title with remaining questions
if st.session_state.game_active:
    # If the game has ended
    if st.session_state.current_question >= st.session_state.remaining_questions:
        st.title("🏆 Final Standings")
        
        # Create a dataframe for final rankings
        ranked = sorted(st.session_state.teams, key=lambda x: x["budget"], reverse=True)
        
        final_data = []
        for i, team in enumerate(ranked, 1):
            final_data.append({
                "Rank": i,
                "Team": team["name"],
                "Final Budget": team["budget"],
                "Questions Answered": team.get("correct", 0) + team.get("incorrect", 0),
                "Correct Answers": team.get("correct", 0)
            })
        
        final_df = pd.DataFrame(final_data)
        
        # Display winner announcement
        winner = ranked[0]
        st.header(f"🥇 Winner: {winner['name']} with ${winner['budget']}")
        
        # Display final standings table
        try:
            st.dataframe(final_df, use_container_width=True)
        except TypeError:
            st.dataframe(final_df, width=800)
        
        # Create budget comparison chart
        budget_chart = alt.Chart(final_df).mark_bar().encode(
            x=alt.X('Team:N', sort='-y'),
            y=alt.Y('Final Budget:Q'),
            color=alt.Color('Team:N', legend=None),
            tooltip=['Team', 'Final Budget', 'Questions Answered', 'Correct Answers']
        ).properties(
            title='Final Budget by Team',
            height=300
        )
        
        st.altair_chart(budget_chart, use_container_width=True)
        
        # Game history
        st.header("Game History")
        history_df = pd.DataFrame(st.session_state.question_history)
        if not history_df.empty:
            try:
                st.dataframe(history_df, use_container_width=True)
            except TypeError:
                st.dataframe(history_df, width=800)
        
    else:  # Game in progress
        question = st.session_state.selected_questions[st.session_state.current_question]
        
        # Question header with difficulty indicator
        difficulty_color = DIFFICULTY_COLORS.get(question["difficulty"], "#000000")
        st.markdown(f"""
        # Question {st.session_state.current_question+1} of {st.session_state.remaining_questions}
        <span style="background-color: {difficulty_color}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 0.8em">
            {question['difficulty'].upper()}
        </span>
        """, unsafe_allow_html=True)
        
        # Show bid phase if no winner yet
        if "winner" not in st.session_state or st.session_state.winner is None:
            with st.form("bid_form", clear_on_submit=False):
                st.subheader("Place Your Secret Bids")
                
                # Create columns for team bids
                cols = st.columns(len(st.session_state.teams))
                bids = {}
                
                for i, team in enumerate(st.session_state.teams):
                    with cols[i]:
                        bids[team["name"]] = st.number_input(
                            f"{team['name']} (${team['budget']})",
                            min_value=0, max_value=team['budget'],
                            key=f"bid_{team['name']}"
                        )
                
                submitted = st.form_submit_button("Submit Bids")
                
                if submitted:
                    # Check if all bids are 0
                    if all(bid == 0 for bid in bids.values()):
                        st.warning("All bids were $0. Someone must bid to continue!")
                    else:
                        # Get all max bidders
                        max_bid = max(bids.values())
                        max_bidders = [name for name, bid in bids.items() if bid == max_bid]
                        
                        # Resolve ties
                        if len(max_bidders) > 1:
                            winner = random.choice(max_bidders)
                            st.info(f"Tie between {', '.join(max_bidders)}! Random selection chose {winner}.")
                        else:
                            winner = max_bidders[0]
                        
                        st.session_state.winner = winner
                        
                        # Deduct bid from winner's budget
                        bid_amount = bids[winner]
                        for t in st.session_state.teams:
                            if t["name"] == winner:
                                t["budget"] -= bid_amount
                        
                        st.rerun()
        
        # Question answering phase
        else:
            st.success(f"{st.session_state.winner} won the bid!")
            
            with st.form("answer_form", clear_on_submit=False):
                st.markdown(f"### {question['text']}")
                
                # First option is always the correct answer in the original data
                correct_answer = question["options"][0]
                
                # Randomize answer order for display
                display_options = []
                # Make a copy to avoid modifying the original
                for option in question["options"]:
                    display_options.append(option)
                    
                # Shuffle the display options
                if "shuffled_options" not in st.session_state:
                    random.shuffle(display_options)
                    st.session_state.shuffled_options = display_options
                else:
                    display_options = st.session_state.shuffled_options
                
                # Let user select from shuffled options
                choice = st.radio("Select your answer:", display_options, key="ans")
                answered = st.form_submit_button("Submit Answer")
                
                if answered:
                    # Check if selected answer matches the correct answer
                    is_correct = (choice == correct_answer)
                    reward = 0
                    
                    # Clear the shuffled options for next question
                    if "shuffled_options" in st.session_state:
                        del st.session_state.shuffled_options
                    
                    # Update team stats
                    for t in st.session_state.teams:
                        if t["name"] == st.session_state.winner:
                            if is_correct:
                                t["correct"] = t.get("correct", 0) + 1
                                reward = random.randint(*question["reward_range"])
                                t["budget"] += reward
                            else:
                                t["incorrect"] = t.get("incorrect", 0) + 1
                    
                    # Handle correct answers
                    if is_correct:
                        st.balloons()
                        st.success(f"✅ Correct! {st.session_state.winner} earns ${reward}")
                    else:
                        st.error(f"❌ Incorrect! The correct answer was: {correct_answer}")
                    
                    # Add to question history
                    st.session_state.question_history.append({
                        "Question #": st.session_state.current_question + 1,
                        "Question": question["text"],
                        "Difficulty": question["difficulty"],
                        "Answering Team": st.session_state.winner,
                        "Correct": is_correct,
                        "Reward": reward if is_correct else 0
                    })
                    
                    # Prepare for next question
                    st.session_state.current_question += 1
                    st.session_state.winner = None
                    
                    # Check if we're at the end
                    if st.session_state.current_question >= st.session_state.remaining_questions:
                        st.info("This was the last question! View final results.")
                    
                    st.rerun()
        
        # Display budget visualization
        st.subheader("Current Budgets")
        
        # Create budget dataframe for visualization
        budget_data = [{"Team": team["name"], "Budget": team["budget"]} for team in st.session_state.teams]
        budget_df = pd.DataFrame(budget_data)
        
        # Create bar chart
        budget_chart = alt.Chart(budget_df).mark_bar().encode(
            x=alt.X('Team:N', sort='-y'),
            y=alt.Y('Budget:Q'),
            color=alt.Color('Team:N', legend=None),
            tooltip=['Team', 'Budget']
        ).properties(
            height=200
        )
        
        st.altair_chart(budget_chart, use_container_width=True)
else:
    # Welcome screen
    st.title("🏆 Machine Learning Gold Rush - Study Guide")
    st.markdown("""
    ## Welcome to the Machine Learning Bid Quiz Game!
    
    ### How to Play:
    1. Teams bid on the chance to answer data science questions
    2. The highest bidder gets to answer the question
    3. If correct, they win additional money based on question difficulty
    4. The team with the most money at the end wins!
    
    **Set up your game in the sidebar to begin.**
    """)
    
    # Display question difficulty levels
    st.subheader("Question Difficulty Levels:")
    levels_data = [
        {"Difficulty": "Easy", "Reward Range": "$100-300"},
        {"Difficulty": "Medium", "Reward Range": "$200-400"},
        {"Difficulty": "Hard", "Reward Range": "$300-500"}
    ]
    
    levels_df = pd.DataFrame(levels_data)
    try:
        st.dataframe(levels_df, use_container_width=True)
    except TypeError:
        st.dataframe(levels_df, width=400)
