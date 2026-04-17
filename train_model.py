import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from feature_engineering import combine_text_and_spelling_features
from preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_DATA_PATHS = [
    PROJECT_ROOT / "data" / "spam.csv",
    PROJECT_ROOT / "data" / "emails.csv",
    PROJECT_ROOT / "data" / "email_spam.csv",
]

TEXT_COLUMN_CANDIDATES = [
    "text",
    "message",
    "email",
    "email_text",
    "body",
    "content",
    "Message",
    "Email Text",
]
LABEL_COLUMN_CANDIDATES = [
    "label",
    "category",
    "class",
    "spam",
    "target",
    "Label",
    "Category",
]


FALLBACK_TRAINING_DATA = [
    ("Win a free iPhone now click here to claim your prize", 1),
    ("Urgent you have won 1000 dollars claim now", 1),
    ("Buy cheap meds online with free shipping today", 1),
    ("Congratulations you are selected as our weekly winner", 1),
    ("Free entry in our cash competition click here", 1),
    ("Your account is suspended verify your password immediately", 1),
    ("Limited time offer act now and receive a bonus", 1),
    ("You have been pre approved for a fast cash loan", 1),
    ("Claim your reward before midnight urgent response required", 1),
    ("Click this link to unlock your exclusive gift card", 1),
    ("Lowest price pills available without prescription", 1),
    ("Final notice your payment failed update billing now", 1),
    ("Winner collect your vacation voucher today", 1),
    ("Make money from home no experience needed", 1),
    ("Exclusive deal buy one get one free click now", 1),
    ("Immediate review required for pending account resolution confirm your profile before access is interrupted", 1),
    ("Your account information requires confirmation to avoid temporary restrictions and delayed transactions", 1),
    ("Use the secure verification page to confirm details and keep your account eligible for continued access", 1),
    ("Automated protection notice complete the account review to prevent service interruption", 1),
    ("We tried to deliver your package but the address information was incomplete", 1),
    ("Please confirm your delivery details so we can schedule another attempt", 1),
    ("Update your shipping information within 24 hours or your package may be returned", 1),
    ("Delivery failed click the tracking update portal to pay a small redelivery fee", 1),
    ("Your mailbox storage is almost full sign in to keep receiving messages", 1),
    ("Unusual login attempt detected verify your identity to restore normal access", 1),
    ("Payment method declined update your billing profile to prevent cancellation", 1),
    ("Security department requires account validation before your case closes automatically", 1),
    ("Confirm your bank details through the secure portal to release the pending transfer", 1),
    ("Your subscription renewal failed update card information immediately to avoid suspension", 1),
    ("A document has been shared with you sign in below to view the confidential file", 1),
    ("Urgentt notifcation your acount has been suspnded verfy your pasword imediately", 1),
    ("We could not complete delivry because your adress detials are missing confirmm shiping now", 1),
    ("Final paymant warning recieve your benifit after verfication through the securty portal", 1),
    ("Cliam your aproved cash transfer by updating bankingg informaton before suspention", 1),
    ("Documment waiting clickk below to finish authetication and restore account access", 1),
    ("Automated notice complete your account protection review before service access is disabled", 1),
    ("A confidential document is waiting sign in through this link to keep access active", 1),
    ("Your account review is incomplete verify details now or the case will close", 1),
    ("Shared file requires login verification confirm identity before viewing the document", 1),
    ("Security review pending restore your mailbox by confirming the requested information", 1),
    ("Hey are we still meeting tomorrow after class", 0),
    ("Can you send me the notes from today's lecture", 0),
    ("Your invoice is attached please review when you can", 0),
    ("Lunch today at noon works for me", 0),
    ("Please review the project report by Friday", 0),
    ("The team meeting moved to 3 pm in room 204", 0),
    ("I finished my part of the assignment and uploaded it", 0),
    ("Could you proofread the introduction before we submit", 0),
    ("Reminder that the presentation practice is tomorrow", 0),
    ("Thanks for sending the schedule for next week", 0),
    ("Here are the documents you requested for the project", 0),
    ("Professor Lee posted the study guide online", 0),
    ("I will call you after work about dinner plans", 0),
    ("Please confirm that you received the updated slides", 0),
    ("The appointment was rescheduled for Monday morning", 0),
    ("The delivery driver left the package at the front desk this afternoon", 0),
    ("Your order from the campus bookstore is ready for pickup", 0),
    ("I updated my address in the school portal yesterday", 0),
    ("The bank statement for our project budget is attached for review", 0),
    ("Please sign in to the class website before the quiz opens", 0),
    ("The IT department confirmed that the password reset was completed", 0),
    ("Your subscription receipt is attached for your records", 0),
    ("The shared document is in our class folder with the final notes", 0),
    ("We can review the account section of the report during the meeting", 0),
    ("I received the package and will bring it to class tomorrow", 0),
    ("I think I spelled tomorrow wrong in the first draft of my paragraph", 0),
    ("Can you resend the package notes because I typed the file name wrong", 0),
    ("The report has a few spelling mistakes but the class content is finished", 0),
    ("I corrected the password example in our cybersecurity slides", 0),
    ("My address line in the project sample data has a typo", 0),
    ("The appointment was rescheduled for Monday morning after the office called", 0),
    ("The IT department confirmed that my password reset was completed successfully", 0),
    ("Could you proofread the introduction before we submit the assignment", 0),
    ("I finished my part of the assignment and uploaded it to the class folder", 0),
    ("Please sign in to the class website before the quiz opens tomorrow", 0),
    ("The shared file for our group project is available on the school website", 0),
    ("I submitted the document through the college portal before the deadline", 0),
    ("The review meeting was moved because the instructor had an appointment", 0),
]


def find_column(columns, candidates):
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate.lower().strip() in normalized:
            return normalized[candidate.lower().strip()]
    return None


def normalize_label(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"spam", "1", "true", "yes", "junk"}:
            return 1
        if normalized in {"ham", "not spam", "not_spam", "0", "false", "no", "safe"}:
            return 0
    return int(value)


def load_dataset(path=None):
    dataset_path = Path(path).expanduser() if path else next(
        (candidate for candidate in DEFAULT_DATA_PATHS if candidate.exists()),
        None,
    )

    if dataset_path and dataset_path.exists():
        df = pd.read_csv(dataset_path, encoding="latin-1")
        text_column = find_column(df.columns, TEXT_COLUMN_CANDIDATES)
        label_column = find_column(df.columns, LABEL_COLUMN_CANDIDATES)

        if text_column is None or label_column is None:
            raise ValueError(
                "Could not find text/label columns in "
                f"{dataset_path}. Expected text column like {TEXT_COLUMN_CANDIDATES} "
                f"and label column like {LABEL_COLUMN_CANDIDATES}."
            )

        df = df[[text_column, label_column]].rename(
            columns={text_column: "text", label_column: "label"}
        )
        source = str(dataset_path.relative_to(PROJECT_ROOT))
    else:
        df = pd.DataFrame(FALLBACK_TRAINING_DATA, columns=["text", "label"])
        source = "built-in fallback dataset"

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].apply(normalize_label).astype(int)
    df = df[df["label"].isin([0, 1])]

    if df["label"].nunique() != 2:
        raise ValueError("Training data must include both spam and not-spam examples.")

    return df, source


def train_model(data_path=None):
    df, source = load_dataset(data_path)
    df["clean_text"] = df["text"].apply(clean_text)

    labels = df["label"]

    stratify = labels if labels.value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    eval_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    eval_vectorizer.fit(train_df["clean_text"])
    X_train = combine_text_and_spelling_features(
        eval_vectorizer,
        train_df["text"],
        train_df["clean_text"],
    )
    X_test = combine_text_and_spelling_features(
        eval_vectorizer,
        test_df["text"],
        test_df["clean_text"],
    )

    eval_model = MultinomialNB()
    eval_model.fit(X_train, y_train)

    predictions = eval_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["not_spam", "spam"],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    MODEL_DIR.mkdir(exist_ok=True)

    final_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    final_vectorizer.fit(df["clean_text"])
    final_features = combine_text_and_spelling_features(
        final_vectorizer,
        df["text"],
        df["clean_text"],
    )
    final_model = MultinomialNB()
    final_model.fit(final_features, labels)

    joblib.dump(final_model, MODEL_DIR / "spam_model.pkl")
    joblib.dump(final_vectorizer, MODEL_DIR / "vectorizer.pkl")

    metrics = {
        "data_source": source,
        "total_examples": int(len(df)),
        "spam_examples": int((df["label"] == 1).sum()),
        "not_spam_examples": int((df["label"] == 0).sum()),
        "test_examples": int(len(y_test)),
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": {
            "true_not_spam_pred_not_spam": int(matrix[0][0]),
            "true_not_spam_pred_spam": int(matrix[0][1]),
            "true_spam_pred_not_spam": int(matrix[1][0]),
            "true_spam_pred_spam": int(matrix[1][1]),
        },
        "classification_report": report,
        "vectorizer": "CountVectorizer(ngram_range=(1, 2)) plus spelling features",
        "model": "MultinomialNB",
    }

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the spam email detector.")
    parser.add_argument(
        "--data",
        help="Optional CSV path. The CSV needs one text column and one spam/ham label column.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training_metrics = train_model(args.data)
    print("Model trained and saved.")
    print(f"Data source: {training_metrics['data_source']}")
    print(f"Examples: {training_metrics['total_examples']}")
    print(f"Accuracy: {training_metrics['accuracy']:.2%}")
