# Detection of Fraudulent Complaints Using Machine Learning in a College Grievance System.
 
## Project Description
This project aims to detect whether a student complaint is genuine or fraudulent using Machine Learning.  
The application is built using Flask and allows users to submit complaints while administrators can view prediction results.

---

##  Project Objectives
- Perform text-based complaint analysis using Machine Learning  
- Detect fraudulent complaints using Multinomial Naive Bayes classifier  
- Enable user registration and secure login  
- Allow complaint submission with automated fraud prediction  
- Provide an admin dashboard to manage complaints and results  
- Build a simple and user-friendly interface using Flask  

---

##  Tech Stack

### Backend
- Python 3.x  
- Flask  

### Machine Learning
- Scikit-learn  
- TF-IDF Vectorizer  
- Multinomial Naive Bayes  

### Data Processing
- Pandas  
- NumPy  

### Frontend
- HTML5  
- CSS3  

### Database
- SQLite  

### Tools
- Git & GitHub  
- VS Code  

---

##  Machine Learning Model
- Algorithm: Multinomial Naive Bayes  
- Vectorizer: TF-IDF  
- Output: Fraud / Genuine  

---

## Project Structure

```text
Fraudulent-Complaint-Detection-ML/
│
├── app.py
├── model.py
├── dataset.csv
├── submitted_complaints.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│   └── style.css
│
└── templates/
    ├── dashboard.html
    ├── index.html
    ├── login.html
    ├── register.html
```
---
## Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

### 3. Open in browser

```text
http://127.0.0.1:5000
```
---
## Default Admin Account
```text
Username: admin***
Password: admin***
```
 Change default credentials before deploying.

---
##  Dataset
The dataset contains sample student grievance complaints labeled as **Fraud** or **Genuine**. It is used to train the Multinomial Naive Bayes classifier for complaint prediction. 

---
##  Features
- User registration & login  
- Complaint submission  
- Real-time fraud detection  
- Admin dashboard  
- Clean user interface  
---
## Future Improvements
- Improve prediction accuracy using advanced NLP models
- Add email notifications
- Role-based access control
- Complaint status tracking
- Dashboard analytics and visualizations
- Cloud deployment
 ---
##  Author

**Ashwini Muthamsetty**

Final Year B.Tech – Computer Science & Engineering

GitHub: https://github.com/AshwiniMuthamsetty
