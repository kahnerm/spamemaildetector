# Spam Email Detector

This project is a Streamlit web app that predicts whether an email is spam using Natural Language Processing and a Multinomial Naive Bayes classifier.

## Project Scope

The detector cleans human language with NLTK, converts the cleaned text into numeric word-frequency features with Scikit-learn's `CountVectorizer`, adds spelling-quality features, and classifies the message as spam or not spam with `MultinomialNB`.

## Team

- Kahner Moreno: backend prediction logic, saved model artifacts, and app integration
- Noel Vazquez: data cleaning, preprocessing, and model training
- Quojosalyn Duck: Streamlit interface, user experience, and documentation

## Files

- `app.py`: Streamlit dashboard for checking emails
- `train_model.py`: trains the model, evaluates it, and saves artifacts
- `predict.py`: loads the saved artifacts and predicts new email text
- `preprocessing.py`: shared NLTK cleaning and stemming logic
- `feature_engineering.py`: spelling and text feature helpers used by training and prediction
- `model/spam_model.pkl`: saved Naive Bayes model
- `model/vectorizer.pkl`: saved CountVectorizer
- `model/metrics.json`: latest training metrics
- `requirements.txt`: Python dependencies

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Using a Kaggle Dataset

The app works immediately with a built-in fallback dataset for class demos. To train on a larger Kaggle spam dataset, place the CSV in one of these paths:

- `data/spam.csv`
- `data/emails.csv`
- `data/email_spam.csv`

Then run:

```bash
python train_model.py
```

You can also pass a custom path:

```bash
python train_model.py --data path/to/spam.csv
```

The CSV needs one text column such as `text`, `message`, `email`, `body`, or `content`, and one label column such as `label`, `category`, `class`, `spam`, or `target`. Labels can be `spam`/`ham`, `1`/`0`, or similar values.

## Model Notes

This is a baseline supervised learning model. It is useful for demonstrating NLP preprocessing, vectorization, and Bayesian classification, but it should not be treated as a production-grade security filter. Accuracy depends heavily on the quality and size of the training dataset.
