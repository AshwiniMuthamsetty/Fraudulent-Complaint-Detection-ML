# Detection of Fraudulent Complaints Using Machine Learning in a College Grievance System

This is a simple college mini project that classifies student complaint text as **Genuine** or **Fraudulent** using basic machine learning.

## Project Features
- Text-only complaint analysis (no image detection)
- ML model: **TF-IDF + Multinomial Naive Bayes**
- Student login/register and complaint submission
- Admin panel to view all complaints and detection results
- Simple Flask web interface

## Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- SQLite

## Folder Structure
- `app.py` - Flask web app and routes
- `model.py` - ML training and text classification logic
- `dataset.csv` - sample labeled complaint dataset
- `templates/` - HTML pages (login, register, complaint form, admin dashboard)
- `static/style.css` - styling
- `users.db` - user database
- `complaints.db` - submitted complaints and predictions

## Run the Project
```bash
pip install -r requirements.txt
python app.py
```

Then open:
- `http://127.0.0.1:5000`

## Default Admin Account
- Username: `admin`
- Password: `admin123`

You can use the admin account to open the dashboard and view complaint records with fraud detection results.
