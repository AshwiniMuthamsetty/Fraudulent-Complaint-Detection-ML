import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from model import classify_complaint, clean_text, train_classifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
USERS_DB = BASE_DIR / "users.db"
COMPLAINTS_DB = BASE_DIR / "complaints.db"

app = Flask(__name__)
app.secret_key = "college-mini-project-secret-key"

VECTORIZER, CLASSIFIER, METRICS = train_classifier(DATASET_PATH)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def init_db() -> None:
    with get_db_connection(USERS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT NOT NULL,
                location TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student'
            )
            """
        )
        user_columns = get_table_columns(conn, "users")
        if "role" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'student'")
        conn.commit()

    with get_db_connection(COMPLAINTS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                complaint_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        complaint_columns = get_table_columns(conn, "complaints")
        if "username" not in complaint_columns:
            conn.execute("ALTER TABLE complaints ADD COLUMN username TEXT NOT NULL DEFAULT 'unknown'")
        if "prediction" not in complaint_columns:
            conn.execute("ALTER TABLE complaints ADD COLUMN prediction TEXT NOT NULL DEFAULT 'Unknown'")
        if "created_at" not in complaint_columns:
            conn.execute("ALTER TABLE complaints ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
        conn.commit()

    with get_db_connection(USERS_DB) as conn:
        admin = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if admin is None:
            conn.execute(
                """
                INSERT INTO users (full_name, phone, email, location, username, password_hash, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Administrator",
                    "9999999999",
                    "admin@college.local",
                    "Campus Office",
                    "admin",
                    generate_password_hash("admin123"),
                    "admin",
                ),
            )
            conn.commit()


def current_user() -> dict | None:
    username = session.get("username")
    role = session.get("role")
    if not username:
        return None
    return {"username": username, "role": role}


def insert_complaint_record(username: str, complaint_text: str, raw_prediction: str, prediction: str) -> None:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cleaned = clean_text(complaint_text)
    features = VECTORIZER.transform([cleaned])

    probabilities = CLASSIFIER.predict_proba(features)[0]
    class_prob = dict(zip(CLASSIFIER.classes_, probabilities))
    probability_fake = float(class_prob.get("spam", 0.0))
    probability_genuine = float(class_prob.get("genuine", 0.0))
    confidence = max(probability_fake, probability_genuine)

    values: dict[str, object] = {
        "complaint_text": complaint_text,
        "username": username,
        "prediction": prediction,
        "created_at": created_at,
        "category": "General",
        "channel": "web",
        "predicted_label": raw_prediction,
        "confidence": confidence,
        "probability_fake": probability_fake,
        "probability_genuine": probability_genuine,
        "is_duplicate": 0,
        "spam_score": int(round(probability_fake * 100)),
        "credibility_score": round(probability_genuine * 100, 2),
        "duplicate_of_id": None,
        "created_by_id": None,
    }

    with get_db_connection(COMPLAINTS_DB) as conn:
        complaint_columns = get_table_columns(conn, "complaints")
        insert_data = {k: v for k, v in values.items() if k in complaint_columns}
        columns = ", ".join(insert_data.keys())
        placeholders = ", ".join("?" for _ in insert_data)
        conn.execute(
            f"INSERT INTO complaints ({columns}) VALUES ({placeholders})",
            tuple(insert_data.values()),
        )
        conn.commit()


@app.route("/")
def home():
    if not current_user():
        return redirect(url_for("login"))
    return redirect(url_for("complaint_page"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", error=None)

    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    email = request.form.get("email", "").strip()
    location = request.form.get("location", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not all([full_name, phone, email, location, username, password]):
        return render_template("register.html", error="All fields are required.")

    if not (phone.isdigit() and len(phone) == 10):
        return render_template("register.html", error="Phone must be 10 digits.")

    password_hash = generate_password_hash(password)
    try:
        with get_db_connection(USERS_DB) as conn:
            conn.execute(
                """
                INSERT INTO users (full_name, phone, email, location, username, password_hash, role)
                VALUES (?, ?, ?, ?, ?, ?, 'student')
                """,
                (full_name, phone, email, location, username, password_hash),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return render_template("register.html", error="Username already exists.")

    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    with get_db_connection(USERS_DB) as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid username or password.")

    session["username"] = user["username"]
    session["role"] = user["role"]
    return redirect(url_for("complaint_page"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/complaint")
def complaint_page():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    return render_template(
        "index.html",
        user_name=user["username"],
        accuracy=f"{METRICS['accuracy']:.2f}%",
        prediction=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    complaint_text = request.form.get("complaint_text", "").strip()
    if not complaint_text:
        return render_template(
            "index.html",
            user_name=user["username"],
            accuracy=f"{METRICS['accuracy']:.2f}%",
            prediction=None,
            error="Complaint text is required.",
        )

    raw_prediction = classify_complaint(VECTORIZER, CLASSIFIER, complaint_text)
    prediction = "Genuine" if raw_prediction == "genuine" else "Fraudulent"

    insert_complaint_record(user["username"], complaint_text, raw_prediction, prediction)

    return render_template(
        "index.html",
        user_name=user["username"],
        accuracy=f"{METRICS['accuracy']:.2f}%",
        prediction=prediction,
        error=None,
    )


@app.route("/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    if user["role"] != "admin":
        return render_template(
            "index.html",
            user_name=user["username"],
            accuracy=f"{METRICS['accuracy']:.2f}%",
            prediction=None,
            error="Only admin can access the dashboard.",
        )

    with get_db_connection(COMPLAINTS_DB) as conn:
        complaints = conn.execute(
            """
            SELECT id, username, complaint_text, prediction, created_at
            FROM complaints
            ORDER BY id DESC
            """
        ).fetchall()

    total = len(complaints)
    genuine_count = sum(1 for row in complaints if row["prediction"] == "Genuine")
    fraud_count = total - genuine_count

    return render_template(
        "dashboard.html",
        user_name=user["username"],
        total=total,
        genuine_count=genuine_count,
        fraud_count=fraud_count,
        complaints=complaints,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
