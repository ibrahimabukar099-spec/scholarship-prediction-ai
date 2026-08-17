# Scholarship Eligibility Prediction System

Machine Learning web application that predicts student scholarship eligibility using an XGBoost Model.

## Features

- Student signup and login
- Admin login
- Faculty selection
- Scholarship application form
- Machine Learning scholarship eligibility prediction
- SHAP explanation page
- Admin dashboard with applicant records
- Search, filter, verify, delete, and CSV export
- One application per student, faculty, and scholarship cycle
- Admin audit logging for applicant updates and deletions
- SQLite database storage

## Tech Stack

- Python
- Flask
- Flask-SQLAlchemy
- SQLite
- XGBoost
- Scikit-learn
- SHAP
- HTML, CSS, Bootstrap, Chart.js

## Important Files

```text
app1.py
templates/
static/
.env
.env.example
requirements.txt
scholarship_data.db
scholarship_dataset.csv
xgboost_scholarship_model.pkl
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure `.env`:

```env
ADMIN_EMAIL=admin@gmail.com
ADMIN_PASSWORD_HASH=your_hashed_admin_password
SECRET_KEY=your_secret_key_here
```

Create a new admin password hash:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_admin_password'))"
```

Copy the generated hash into `ADMIN_PASSWORD_HASH`.

## Run

```bash
python app1.py
```

Open:

```text
http://127.0.0.1:5000
```

## Notes

- `admin@gmail.com` is reserved for Admin login only.
- Admin email cannot be used as a student account.
- Keep `.env` private because it contains secret settings.
- The trained model is already included as `xgboost_scholarship_model.pkl`.
- Existing applicant records without a scholarship cycle remain legacy records; new applications are always linked to the active cycle.
- Development server runs locally at `127.0.0.1:5000`.
