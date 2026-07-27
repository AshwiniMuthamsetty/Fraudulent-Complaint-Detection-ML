import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

VALID_LABELS = {"genuine", "spam"}


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase and remove extra symbols/spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load and validate the training dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"complaint_text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain 'complaint_text' and 'label' columns.")

    df = df.dropna(subset=["complaint_text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)]

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both 'genuine' and 'spam' labels.")

    df["cleaned_text"] = df["complaint_text"].apply(clean_text)
    return df


def train_classifier(dataset_path: str | Path):
    """Train TF-IDF + Multinomial Naive Bayes classifier."""
    df = _load_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)

    predictions = classifier.predict(x_test_vec)
    accuracy = accuracy_score(y_test, predictions) * 100

    metrics = {
        "accuracy": accuracy,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }

    return vectorizer, classifier, metrics


def classify_complaint(vectorizer: TfidfVectorizer, classifier: MultinomialNB, complaint_text: str) -> str:
    """Predict complaint category as 'genuine' or 'spam'."""
    cleaned = clean_text(complaint_text)
    vector = vectorizer.transform([cleaned])
    return str(classifier.predict(vector)[0])


def get_existing_complaints(file_path: str | Path) -> list[str]:
    """Load previously submitted complaints for duplicate checking."""
    path = Path(file_path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if "complaint_text" not in df.columns:
        return []

    complaints = (
        df["complaint_text"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    return [text for text in complaints if text]


def detect_duplicate(new_complaint: str, existing_complaints: list[str], threshold: float = 0.75):
    """Detect duplicates separately using cosine similarity on TF-IDF vectors."""
    if not existing_complaints:
        return False, 0.0, ""

    cleaned_existing = [clean_text(text) for text in existing_complaints]
    cleaned_new = clean_text(new_complaint)

    corpus = cleaned_existing + [cleaned_new]
    duplicate_vectorizer = TfidfVectorizer(stop_words="english")
    matrix = duplicate_vectorizer.fit_transform(corpus)

    new_vector = matrix[-1]
    existing_vectors = matrix[:-1]

    scores = cosine_similarity(new_vector, existing_vectors).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    is_duplicate = best_score >= threshold
    closest_match = existing_complaints[best_index]

    return is_duplicate, best_score, closest_match


def save_complaint(complaint_text: str, label: str, file_path: str | Path) -> None:
    """Store new complaint so future submissions can be checked for duplicates."""
    path = Path(file_path)

    new_row = pd.DataFrame(
        [
            {
                "complaint_text": complaint_text,
                "label": label,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )

    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(path, index=False)
