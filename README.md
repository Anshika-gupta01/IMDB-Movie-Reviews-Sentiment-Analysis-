# IMDB-Movie-Reviews-Sentiment-Analysis-
This project performs sentiment analysis on IMDB movie reviews using the Kaggle IMDB 50000 dataset. The goal is to classify reviews as positive or negative using machine learning techniques. The project includes data preprocessing, vectorization, model training, evaluation, and visualization of results using a confusion matrix.

📁 Dataset
Source: Kaggle IMDB 50000 dataset
Description: The dataset contains 50,000 movie reviews from IMDB, labeled as positive or negative.

🚀 Features and Workflow
1. Data Cleaning and Preprocessing
Converted text to lowercase.

Removed punctuation and special characters.
Encoded labels (positive → 1, negative → 0).

2. Vectorization
Used TF-IDF (Term Frequency-Inverse Document Frequency) for converting text into numerical format.

3. Model Training
Split data into training and test sets (80% training, 20% test).
Trained a Multinomial Naive Bayes classifier.

4. Evaluation
Checked model accuracy.
Generated a confusion matrix and classification report for performance evaluation.

🏆 Results
Model Accuracy: Achieved a high accuracy using the Naive Bayes model.
Confusion Matrix: Created to visualize the classification results.

📊 Libraries Used
pandas
numpy
sklearn
joblib
re

💡 How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt

Run the Jupyter Notebook:
bash
Copy
Edit
jupyter notebook Sentiment_new.ipynb

✅ To-Do
Try other models like Logistic Regression and SVM.
Improve preprocessing to handle sarcasm and complex sentences.

🤝 Contributing
Feel free to contribute by creating a pull request or reporting issues.

🏅 Acknowledgments
Kaggle for the dataset.

Scikit-learn for providing machine learning tools.
