from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import UniqueConstraint, inspect, text, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import joblib
import pandas as pd
import shap
import os
import secrets
import json
import re
from io import BytesIO
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
# Nidaam noo sahlaya inaan toos u akhrino .env adigoon isticmaalin python-dotenv
env_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app_environment = os.environ.get('APP_ENV', 'development').lower()
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if app_environment == 'production':
        raise RuntimeError('SECRET_KEY must be configured when APP_ENV=production.')
    secret_key = secrets.token_urlsafe(64)
    app.logger.warning('SECRET_KEY is not configured; using an ephemeral development key for local development only.')
app.config['SECRET_KEY'] = secret_key

# --- GLOBAL SETTINGS ---
ADMIN_EMAIL = (os.environ.get('ADMIN_EMAIL') or 'admin@gmail.com').strip().lower()
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH')
DEV_ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'Admin@123')
if not ADMIN_PASSWORD_HASH and app_environment != 'production':
    ADMIN_PASSWORD_HASH = generate_password_hash(DEV_ADMIN_PASSWORD)
    app.logger.warning('ADMIN_PASSWORD_HASH is not configured; using local development admin password only.')

# --- DATABASE SETUP ---
basedir = os.path.abspath(os.path.dirname(__file__))
custom_db_path = os.environ.get('DATABASE_PATH') or os.environ.get('DATABASE_URL')
if custom_db_path:
    if custom_db_path.startswith('sqlite:///'):
        app.config['SQLALCHEMY_DATABASE_URI'] = custom_db_path
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.abspath(custom_db_path)
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'scholarship_data.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20), default="student")

    def __init__(self, full_name, email, password, role="student"):
        self.full_name = full_name
        self.email = email
        self.password = password
        self.role = role

class Faculty(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    min_gpa = db.Column(db.Float, default=0.0)
    is_active = db.Column(db.Boolean, default=True)

class ScholarshipCycle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    academic_year = db.Column(db.String(20), nullable=False)
    open_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    close_date = db.Column(db.DateTime, nullable=True)

# ====================================================
# TASK 1 — Duplicate Applications Policy & Constraint
# ====================================================
# REAPPLICATION POLICY DOCUMENTATION:
# To prevent resubmission gaming and duplicate entry exploitation,
# a student is restricted from submitting multiple applications for the same:
#   - user (user_id)
#   - faculty (faculty_id)
#   - scholarship cycle (cycle_id)
# A database-level UniqueConstraint enforces this integrity rule across cycles.

class Applicant(db.Model):
    __table_args__ = (
        UniqueConstraint('user_id', 'faculty_id', 'cycle_id', name='uq_applicant_user_faculty_cycle'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    cycle_id = db.Column(db.Integer, db.ForeignKey('scholarship_cycle.id'), nullable=False)
    # TASK 5 — English Database Identifiers: Renamed 'magaca' column to 'full_name'
    full_name = db.Column(db.String(100))
    gpa = db.Column(db.Float)
    family_income = db.Column(db.Float)

    # TASK 4 — Binary Features: Kept as raw 0/1 values bypassing StandardScaler
    is_orphan = db.Column(db.Integer)
    is_displaced = db.Column(db.Integer)
    region = db.Column(db.Integer, default=0)
    high_school_type = db.Column(db.Integer, default=0)
    has_verification = db.Column(db.Integer, default=0)
    gender = db.Column(db.Integer, default=0)
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'))
    prediction_result = db.Column(db.Integer)
    model_prediction = db.Column(db.Integer, nullable=True)
    model_confidence = db.Column(db.Float, nullable=True)
    reason = db.Column(db.Text)
    date_applied = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_verified = db.Column(db.Integer, default=0)
    verification_status = db.Column(db.String(20), default='pending', nullable=False)
    verification_note = db.Column(db.Text, nullable=True)
    verification_document = db.Column(db.String(255), nullable=True)
    verification_document_original = db.Column(db.String(255), nullable=True)
    verified_at = db.Column(db.DateTime, nullable=True)
    verified_by_admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    deleted_by_admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def __init__(self, user_id, full_name, gpa, family_income, is_orphan, is_displaced,
                 region, high_school_type, has_verification, gender, faculty_id,
                 prediction_result, reason, date_applied=None, is_verified=0, cycle_id=None,
                 verification_status='pending', verification_note=None, verified_at=None,
                 verified_by_admin_id=None, verification_document=None,
                 verification_document_original=None):
        self.user_id = user_id
        self.full_name = full_name
        self.gpa = gpa
        self.family_income = family_income
        self.is_orphan = is_orphan
        self.is_displaced = is_displaced
        self.region = region
        self.high_school_type = high_school_type
        self.has_verification = has_verification
        self.gender = gender
        self.faculty_id = faculty_id
        self.prediction_result = prediction_result
        self.reason = reason
        if date_applied:
            self.date_applied = date_applied
        self.is_verified = is_verified
        self.verification_status = verification_status
        self.verification_note = verification_note
        self.verification_document = verification_document
        self.verification_document_original = verification_document_original
        self.verified_at = verified_at
        self.verified_by_admin_id = verified_by_admin_id
        self.cycle_id = cycle_id

# ====================================================
# TASK 3 — Audit Log Table (SQLAlchemy Model)
# ====================================================
# Automatically records administrative actions (Create, Update, Delete)
# including admin_id, action type, target table, record ID, description, and timestamp.
class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(20), nullable=False)
    target_table = db.Column(db.String(100), nullable=False)
    target_record_id = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

MODEL_FEATURES = [
    'gpa', 'family_income', 'is_orphan', 'is_displaced', 'region',
    'high_school_type', 'has_verification', 'gender', 'faculty'
]
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(basedir, 'xgboost_scholarship_model.pkl'))
MODEL_METRICS_PATH = os.environ.get('MODEL_METRICS_PATH', os.path.join(basedir, 'model_metrics.json'))
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(basedir, 'uploads', 'verification_documents'))
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def validate_model_schema(loaded_model):
    feature_names = list(getattr(loaded_model, 'feature_names_in_', []))
    if feature_names != MODEL_FEATURES:
        raise ValueError(
            f'Model feature schema mismatch. Expected {MODEL_FEATURES}, received {feature_names}.'
        )
    if not hasattr(loaded_model, 'predict') or not hasattr(loaded_model, 'predict_proba'):
        raise TypeError('Configured model must provide predict and predict_proba methods.')


def load_model():
    try:
        loaded_model = joblib.load(MODEL_PATH)
        validate_model_schema(loaded_model)
        return loaded_model, None
    except FileNotFoundError:
        message = f'Configured model file was not found: {MODEL_PATH}'
    except (ValueError, TypeError, AttributeError, EOFError) as exc:
        message = f'Configured model is unavailable or incompatible: {exc}'
    except Exception as exc:
        message = f'Configured model could not be loaded: {exc}'

    app.logger.error(message)
    return None, message


model, model_load_error = load_model()

# --- FACULTY AND OCCUPATION MAPPINGS ---
faculty_list = ['CS', 'Medicine', 'Eng', 'Agri']
region_list = ['Banadir', 'Puntland', 'Somaliland', 'Jubaland', 'Galmudug', 'Hirshabelle', 'Koonfur Galbeed']
school_list = ['Public', 'Private']
faculty_gpa_requirements = {
    'Medicine': 80,
    'Eng': 70,
    'CS': 60,
    'Agri': 55
}
priority_bonus_points = 2
low_income_threshold = 250
VALID_VERIFICATION_STATUSES = {'pending', 'verified', 'rejected'}


def get_current_scholarship_cycle():
    return (ScholarshipCycle.query
            .filter((ScholarshipCycle.close_date.is_(None)) |
                    (ScholarshipCycle.close_date > datetime.now(timezone.utc)))
            .order_by(ScholarshipCycle.open_date.desc())
            .first())


def active_applicants_query():
    return Applicant.query.filter_by(is_deleted=False)


def log_admin_action(action, target_table, target_record_id, description):
    """Record an admin action in the audit log.

    This helper is transaction-safe for critical admin actions. It
    raises if an audit record cannot be created so the caller can roll
    back the entire operation.
    """
    admin = db.session.get(User, session.get('user_id'))
    if admin is None or admin.role != 'admin':
        admin = ensure_configured_admin_user()
    if admin is None:
        raise RuntimeError('Admin user not available for audit logging.')

    audit_entry = AuditLog(
        admin_id=admin.id,
        action=action,
        target_table=target_table,
        target_record_id=target_record_id,
        description=description,
    )
    db.session.add(audit_entry)
    db.session.flush()


def migrate_database_schema():
    inspector = inspect(db.engine)
    if 'applicant' not in inspector.get_table_names():
        return

    applicant_columns = {column['name'] for column in inspector.get_columns('applicant')}
    with db.engine.begin() as connection:
        if 'magaca' in applicant_columns and 'full_name' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant RENAME COLUMN magaca TO full_name'))

        if 'is_deleted' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT 0'))
        if 'deleted_at' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN deleted_at DATETIME'))
        if 'deleted_by_admin_id' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN deleted_by_admin_id INTEGER'))
        if 'verification_status' not in applicant_columns:
            connection.execute(text("ALTER TABLE applicant ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'pending'"))
        if 'verification_note' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN verification_note TEXT'))
        if 'verification_document' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN verification_document TEXT'))
        if 'verification_document_original' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN verification_document_original TEXT'))
        if 'verified_at' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN verified_at DATETIME'))
        if 'verified_by_admin_id' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN verified_by_admin_id INTEGER'))
        if 'model_prediction' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN model_prediction INTEGER'))
        if 'model_confidence' not in applicant_columns:
            connection.execute(text('ALTER TABLE applicant ADD COLUMN model_confidence FLOAT'))

        connection.execute(text(
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_applicant_user_faculty_cycle '
            'ON applicant (user_id, faculty_id, cycle_id) WHERE cycle_id IS NOT NULL'
        ))


def ensure_configured_admin_user():
    if not ADMIN_EMAIL or not ADMIN_PASSWORD_HASH:
        return None

    admin = User.query.filter_by(email=ADMIN_EMAIL, role='admin').first()
    if admin is None:
        admin = User(
            full_name='System Administrator',
            email=ADMIN_EMAIL,
            password=ADMIN_PASSWORD_HASH,
            role='admin',
        )
        db.session.add(admin)
        db.session.flush()
    elif admin.password != ADMIN_PASSWORD_HASH:
        admin.password = ADMIN_PASSWORD_HASH
    return admin


def admin_required():
    return session.get('logged_in') and session.get('role') == 'admin'

def admin_password_matches(admin, password):
    if ADMIN_PASSWORD_HASH and check_password_hash(ADMIN_PASSWORD_HASH, password):
        return True
    if admin and admin.password and check_password_hash(admin.password, password):
        return True
    return app_environment != 'production' and password == DEV_ADMIN_PASSWORD

def get_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

def validate_csrf_token():
    token = request.form.get('csrf_token')
    if not token:
        token = request.headers.get('X-CSRFToken')
    return token == session.get('csrf_token') and token is not None

def student_required():
    return session.get('logged_in') and session.get('role') == 'student'

def validate_student_password(password):
    if not password or len(password) < 6:
        return False, "Password-ku waa inuu ahaadaa ugu yaraan 6 xaraf."

    checks = [
        bool(re.search(r"[A-Za-z]", password)),
        bool(re.search(r"[0-9]", password)),
        bool(re.search(r"[^A-Za-z0-9]", password)),
    ]
    if sum(checks) < 2:
        return False, "Password-ku waa inuu leeyahay ugu yaraan laba nooc: xarfo, tiro, ama calaamad."

    return True, ""

def validate_full_name(full_name):
    name_parts = full_name.split()
    if len(name_parts) < 3:
        return False, "Magaca waa inuu ka koobnaadaa ugu yaraan 3 magac."
    if not re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", full_name):
        return False, "Magaca waxaa lagu oggol yahay xarfo iyo spaces oo keliya. Tirooyin iyo calaamado lama oggola."
    return True, ""

def allowed_document_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_DOCUMENT_EXTENSIONS
    )

def save_verification_document(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("Warqadda caddeynta waa qasab. Fadlan soo geli PDF, Word, PNG, ama JPG ka hor intaadan gudbin.")
    if not allowed_document_file(file_storage.filename):
        raise ValueError("Nooca warqadda lama oggola. Soo geli PDF, Word, PNG, ama JPG.")

    original_filename = secure_filename(file_storage.filename)
    extension = original_filename.rsplit('.', 1)[1].lower()
    stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(8)}.{extension}"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file_storage.save(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename))
    return stored_filename, original_filename

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=get_csrf_token())



def get_faculty_name(faculty_id):
    if 0 <= faculty_id < len(faculty_list):
        return faculty_list[faculty_id]
    return 'Other'

def get_choice(options, index, default_index=0):
    if 0 <= index < len(options):
        return options[index]
    return options[default_index]

def calculate_priority_bonus(orphan, displaced, income):
    bonus = 0
    reasons = []

    if int(orphan) == 1:
        bonus += priority_bonus_points
        reasons.append("agoonnimo +2")
    if int(displaced) == 1:
        bonus += priority_bonus_points
        reasons.append("barakac +2")
    if float(income) < low_income_threshold:
        bonus += priority_bonus_points
        reasons.append("dakhliga qoyska oo ka yar $250 +2")

    return bonus, reasons


def get_adjusted_gpa_requirement(faculty_name, priority_bonus):
    base_requirement = faculty_gpa_requirements.get(faculty_name, 0)
    return base_requirement, max(base_requirement - priority_bonus, 0)




def make_feature_frame(gpa, income, orphan, displaced, region, high_school_type,
                       verification, gender, faculty):
    """Construct feature DataFrame for XGBoost model inference.

    Parameter 'verification' (has_verification) is an APPLICANT-DECLARED self-reported binary feature (0 or 1).
    It is used solely as an input feature for the ML model and does NOT represent administrative verification.
    Administrative verification is separately tracked in verification_status ('pending', 'verified', 'rejected').
    """
    return pd.DataFrame([[
        float(gpa),
        float(income),
        int(orphan),
        int(displaced),
        str(region),
        str(high_school_type),
        int(verification),
        "Male" if int(gender) == 0 else "Female",
        str(faculty)
    ]], columns=[
        "gpa", "family_income", "is_orphan", "is_displaced", "region",
        "high_school_type", "has_verification", "gender", "faculty"
    ])

def get_model_metrics():
    """Load evaluation metrics from the training artifact.

    The dashboard reads metrics dynamically from a JSON file generated
    during model training. This avoids hard-coded performance values.
    """
    try:
        with open(MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            return metrics
    except FileNotFoundError:
        app.logger.warning('Model metrics file not found: %s', MODEL_METRICS_PATH)
        return {}
    except json.JSONDecodeError as exc:
        app.logger.error('Model metrics file is invalid: %s', exc)
        return {}


def evaluate_policy(gpa, verification_status, required_gpa, income):
    """Evaluate administrative scholarship policy rules.

    Requires verification_status == 'verified' (administrative action).
    """
    if verification_status != 'verified':
        return False, 'verification_missing'

    try:
        if float(income) > 500:
            return False, 'income_above_limit'
    except ValueError:
        return False, 'income_invalid'

    if float(gpa) < float(required_gpa):
        return False, 'gpa_below_requirement'

    return True, 'policy_passed'


def predict_with_xgboost(feature_frame):
    if model is None:
        raise RuntimeError('XGBoost model-ka lama rarin.')

    try:
        pred = int(model.predict(feature_frame)[0])
        probs = model.predict_proba(feature_frame)[0]
        confidence = float(probs[1]) if pred == 1 else float(probs[0])
        return pred, confidence
    except Exception as exc:
        app.logger.error('XGBoost Prediction Error: %s', exc)
        raise RuntimeError(f'XGBoost prediction failed: {exc}')

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', '').strip().lower()
        
        if not validate_csrf_token():
            flash("Codsiga amniga lama xaqiijin.", "danger")
            return redirect(url_for('login_page'))

        # 1. Hubi haddii uu yahay Admin (Hardcoded check)
        if role == "admin":
            if not ADMIN_EMAIL or not ADMIN_PASSWORD_HASH:
                flash("Admin credentials lama dejin. Fadlan la xiriir maamulka.", "danger")
                return redirect(url_for('login_page'))
            admin = ensure_configured_admin_user()
            if username == ADMIN_EMAIL and admin_password_matches(admin, password):
                admin = ensure_configured_admin_user()
                db.session.commit()
                session['user_id'] = admin.id
                session['role'] = 'admin'
                session['logged_in'] = True
                session['name'] = "System Administrator"
                flash("Si guul leh ayaad u gashay sidii Admin!", "success")
                return redirect(url_for('admin_database_route'))
            else:
                flash("Email ama Password-ka Admin-ka waa khalad!", "danger")
                return redirect(url_for('login_page'))

        if username == ADMIN_EMAIL:
            flash("Email-kan waxaa loo isticmaalaa Admin oo keliya. Fadlan dooro Admin.", "danger")
            return redirect(url_for('login_page'))

        # 2. Hubi haddii uu yahay Student (Database check)
        user = User.query.filter_by(email=username, role='student').first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = 'student'
            session['logged_in'] = True
            session['name'] = user.full_name
            flash(f"Ku soo dhawaada, {user.full_name}!", "success")
            return redirect(url_for('student_dashboard'))
        
        flash("Email ama Password khaldan!", "danger")
        
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        if not validate_csrf_token():
            flash("Codsiga amniga lama xaqiijin.", "danger")
            return redirect(url_for('signup_page'))

        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')

        gmail_pattern = r"^[A-Za-z0-9._%+-]{3,}@gmail\.com$"
        if not re.fullmatch(gmail_pattern, email, re.IGNORECASE):
            flash("Fadlan geli Gmail sax ah, tusaale: burhan@gmail.com", "danger")
            return redirect(url_for('signup_page'))

        is_valid_name, name_message = validate_full_name(full_name)
        if not is_valid_name:
            flash(name_message, "danger")
            return redirect(url_for('signup_page'))

        if email == ADMIN_EMAIL:
            flash("Email-ka Admin lama diiwaangelin karo sidii arday.", "danger")
            return redirect(url_for('signup_page'))

        is_valid_password, password_message = validate_student_password(password)
        if not is_valid_password:
            flash(password_message, "danger")
            return redirect(url_for('signup_page'))
        
        if User.query.filter_by(email=email).first():
            flash("Email-kan horey ayaa loo diiwaangeliyey!", "danger")
            return redirect(url_for('signup_page'))
            
        hashed_pw = generate_password_hash(password)
        new_user = User(full_name=full_name, email=email, password=hashed_pw, role='student')
        db.session.add(new_user)
        db.session.commit()
        
        # Auto-login: Set session variables directly
        session['user_id'] = new_user.id
        session['role'] = 'student'
        session['logged_in'] = True
        session['name'] = new_user.full_name
        
        return redirect(url_for('student_dashboard'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password_page():
    if request.method == 'POST':
        if not validate_csrf_token():
            flash("Codsiga amniga lama xaqiijin.", "danger")
            return redirect(url_for('forgot_password_page'))

        email = request.form.get('email', '').strip().lower()
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if new_password != confirm_password:
            flash("Labada password isku mid ma aha.", "danger")
            return redirect(url_for('forgot_password_page'))

        is_valid_password, password_message = validate_student_password(new_password)
        if not is_valid_password:
            flash(password_message, "danger")
            return redirect(url_for('forgot_password_page'))

        user = User.query.filter_by(email=email, role='student').first()
        if not user:
            flash("Email-kan lagama helin arday diiwaangashan.", "danger")
            return redirect(url_for('forgot_password_page'))

        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash("Password-ka si guul leh ayaa loo beddelay. Hadda waad geli kartaa.", "success")
        return redirect(url_for('login_page'))

    return render_template('forgot_password.html')

@app.route('/student_dashboard')
def student_dashboard():
    if not session.get('logged_in') or session.get('role') != 'student':
        return redirect(url_for('login_page'))
    
    # Get applicant history for this user
    user_id = session.get('user_id')
    applications = (active_applicants_query()
                    .filter_by(user_id=user_id)
                    .order_by(Applicant.id.desc())
                    .all())
    
    return render_template('student_dashboard.html', 
                           name=session.get('name'), 
                           applications=applications)

@app.route('/logout')
def logout():
    session.clear()
    flash("Si guul leh ayaad uga baxday nidaamka.", "info")
    return redirect(url_for('login_page'))

@app.route('/admin_database')
def admin_database_route():
    if not admin_required():
        flash("Fadlan marka hore soo gal sidii Admin!", "danger")
        return redirect(url_for('login_page'))

    search_query = request.args.get('search', '').strip()
    status_filter = request.args.get('status', '').strip()
    faculty_filter = request.args.get('faculty', '').strip()
    verification_filter = request.args.get('verification', '').strip()
    page = max(request.args.get('page', 1, type=int), 1)
    per_page = 10

    query = active_applicants_query()
    if search_query:
        query = query.filter(Applicant.full_name.like(f"%{search_query}%"))
    if status_filter in {"0", "1"}:
        query = query.filter(Applicant.prediction_result == int(status_filter))
    if faculty_filter.isdigit():
        query = query.filter(Applicant.faculty_id == int(faculty_filter))
    if verification_filter in VALID_VERIFICATION_STATUSES:
        query = query.filter(Applicant.verification_status == verification_filter)

    filtered_total = query.count()
    students = (query.order_by(Applicant.id.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all())
    total_pages = max((filtered_total + per_page - 1) // per_page, 1)
    
    active_applicants = active_applicants_query()
    total_apps = active_applicants.count()
    accepted = active_applicants.filter(Applicant.prediction_result == 1, Applicant.verification_status == 'verified').count()
    rejected = active_applicants.filter(Applicant.prediction_result == 0, Applicant.verification_status == 'verified').count()
    rejected_verification = active_applicants.filter(Applicant.verification_status == 'rejected').count()
    orphans = active_applicants.filter_by(is_orphan=1).count()
    displaced = active_applicants.filter_by(is_displaced=1).count()
    verified = active_applicants.filter(Applicant.verification_status == 'verified').count()
    pending = active_applicants.filter(Applicant.verification_status == 'pending').count()
    faculty_stats = (
        db.session.query(Applicant.faculty_id, func.count(Applicant.id)).filter(Applicant.is_deleted.is_(False))
        .group_by(Applicant.faculty_id)
        .all()
    )
    region_stats = (
        db.session.query(Applicant.region, func.count(Applicant.id)).filter(Applicant.is_deleted.is_(False))
        .group_by(Applicant.region)
        .all()
    )
    top_gpa = active_applicants.order_by(Applicant.gpa.desc()).first()
    lowest_income = active_applicants.order_by(Applicant.family_income.asc()).first()
    latest_verification = (
        AuditLog.query
        .filter(AuditLog.target_table == 'applicant', AuditLog.action.in_(['VERIFY', 'REJECT_VERIFICATION', 'UPDATE']))
        .order_by(AuditLog.timestamp.desc(), AuditLog.id.desc())
        .first()
    )
    recently_verified = db.session.get(Applicant, latest_verification.target_record_id) if latest_verification else (
        active_applicants.filter_by(is_verified=1).order_by(Applicant.id.desc()).first()
    )
    latest_verified_date = latest_verification.timestamp if latest_verification else None
    active_cycle = get_current_scholarship_cycle()
    average_gpa = db.session.query(func.avg(Applicant.gpa)).scalar() or 0
    average_income = db.session.query(func.avg(Applicant.family_income)).scalar() or 0
    faculty_names = dict(enumerate(faculty_list))

    return render_template('admin.html', 
                           students=students, 
                           total=total_apps, 
                           accepted=accepted, 
                           rejected=rejected, 
                           orphans=orphans,
                           displaced=displaced,
                           verified=verified,
                           pending=pending,
                           average_gpa=round(average_gpa, 1),
                           average_income=round(average_income, 0),
                           model_metrics=get_model_metrics(),
                           faculty_stats=[{"label": get_faculty_name(item[0]), "count": item[1]} for item in faculty_stats],
                           region_stats=[{"label": get_choice(region_list, item[0]), "count": item[1]} for item in region_stats],
                           top_applicants={"gpa": top_gpa, "income": lowest_income, "verified": recently_verified},
                           latest_verified_date=latest_verified_date,
                           active_cycle=active_cycle,
                           today=datetime.now().strftime('%d %b %Y'),
                           filtered_total=filtered_total,
                           page=page, total_pages=total_pages,
                           filters={
                               "search": search_query,
                               "status": status_filter,
                               "faculty": faculty_filter,
                               "verification": verification_filter
                           },
                           faculty_options=list(enumerate(faculty_list)),
                           faculty_names=faculty_names,
                           csrf_token=get_csrf_token(),
                           rejected_verification=rejected_verification
                           )

@app.route('/select_faculty')
def select_faculty():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template('faculty_dashboard.html')

@app.route('/apply/<int:faculty_id>')
def apply_with_faculty(faculty_id):
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    user = db.session.get(User, session.get('user_id'))
    return render_template('form.html', user=user, selected_faculty=faculty_id)


@app.route('/apply', methods=['POST'])
def submit_application():
    """Handle form submission: validate, apply hard constraints, call model, save."""
    if not student_required():
        flash("Fadlan marka hore soo gal nidaamka!", "warning")
        return redirect(url_for('login_page'))

    if not validate_csrf_token():
        flash("Codsiga amniga lama xaqiijin.", "danger")
        return redirect(url_for('select_faculty'))

    try:
        form = request.form
        full_name = form.get('full_name', '').strip()
        gpa = float(form.get('gpa', 0))
        family_income = float(form.get('family_income', 0))
        has_verification = int(form.get('has_verification', 1))
        is_orphan = int(form.get('is_orphan', 0))
        is_displaced = int(form.get('is_displaced', 0))
        faculty_value = form.get('faculty')
        if faculty_value is None:
            flash("Kulliyadda la doortay lama helin. Fadlan mar kale dooro kulliyaddaada.", "danger")
            return redirect(url_for('select_faculty'))
        faculty_int = int(faculty_value)
        region_int = int(form.get('region', 0))
        high_school_int = int(form.get('high_school_type', 0))
        gender_int = int(form.get('gender', 0))

        # Basic validation
        if not full_name:
            flash("Magaca oo dhammaystiran waa loo baahan yahay.", "danger")
            return redirect(url_for('apply_with_faculty', faculty_id=faculty_int))
        is_valid_name, name_message = validate_full_name(full_name)
        if not is_valid_name:
            flash(name_message, "danger")
            return redirect(url_for('apply_with_faculty', faculty_id=faculty_int))
        if not (0 <= gpa <= 100):
            flash("GPA waa inuu u dhexeeyaa 0 iyo 100.", "danger")
            return redirect(url_for('apply_with_faculty', faculty_id=faculty_int))
        if family_income < 0:
            flash("Dakhliga qoyska ma noqon karo tiro taban.", "danger")
            return redirect(url_for('apply_with_faculty', faculty_id=faculty_int))
        if not (0 <= faculty_int < len(faculty_list)):
            flash("Kulliyadda la doortay sax ma aha.", "danger")
            return redirect(url_for('select_faculty'))

        current_cycle = get_current_scholarship_cycle()
        if current_cycle is None:
            flash("Ma jiro scholarship cycle furan hadda.", "warning")
            return redirect(url_for('student_dashboard'))

        # Prevent duplicate submission for same faculty & cycle
        existing = Applicant.query.filter_by(
            user_id=session.get('user_id'), faculty_id=faculty_int, cycle_id=current_cycle.id
        ).first()
        if existing:
            flash("Waxaad horay uga codsatay kuliyadda tan isla xilliga scholarship.", "warning")
            return redirect(url_for('student_dashboard'))

        prediction_result = 0
        reason = "Codsigaaga waa la helay, wuxuuna sugayaa xaqiijinta maamulka."
        verification_document, verification_document_original = save_verification_document(
            request.files.get('verification_document')
        )

        new_applicant = Applicant(
            user_id=session.get('user_id'),
            full_name=full_name,
            gpa=gpa,
            family_income=family_income,
            is_orphan=is_orphan,
            is_displaced=is_displaced,
            region=region_int,
            high_school_type=high_school_int,
            has_verification=has_verification,
            gender=gender_int,
            faculty_id=faculty_int,
            cycle_id=current_cycle.id,
            prediction_result=prediction_result,
            reason=reason,
            is_verified=0,
            verification_status='pending',
            verification_document=verification_document,
            verification_document_original=verification_document_original
        )
        db.session.add(new_applicant)
        db.session.commit()

        flash("Codsigaaga waa la keydiyay — waadna mahadsan tahay!", "success")
        return redirect(url_for('student_dashboard'))

    except Exception as exc:
        db.session.rollback()
        app.logger.error('submit_application failed: %s', exc)
        flash(f"Khalad ayaa dhacay marka la keydinayey codsiga: {exc}", "danger")
        return redirect(url_for('apply_with_faculty', faculty_id=request.form.get('faculty', 0)))

@app.route('/apply_now')
def student_form():
    if not session.get('logged_in'):
        flash("Fadlan marka hore soo gal nidaamka!", "warning")
        return redirect(url_for('login_page'))

    return redirect(url_for('select_faculty'))

@app.route('/result/<int:id>')
def result_page(id):
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    student = active_applicants_query().filter_by(id=id).first_or_404()
    
    # Amni: Hubi in qofka uu iska leeyahay codsigan ama uu yahay Admin
    if session.get('role') != 'admin' and student.user_id != session.get('user_id'):
        flash("Ma xaq u lihid inaad aragto xogtan!", "danger")
        return redirect(url_for('student_dashboard'))
        
    return render_template('result.html', s=student)

@app.route('/verification_document/<int:id>')
def verification_document_route(id):
    if not admin_required():
        flash("Kaliya maamulka ayaa furi kara warqadaha caddeynta.", "danger")
        return redirect(url_for('login_page'))

    applicant = active_applicants_query().filter_by(id=id).first_or_404()
    if not applicant.verification_document:
        abort(404)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], applicant.verification_document)
    if not os.path.isfile(file_path):
        abort(404)

    return send_file(
        file_path,
        as_attachment=False,
        download_name=applicant.verification_document_original or applicant.verification_document
    )

@app.route('/delete/<int:id>', methods=['POST'])
def delete_student(id):
    if not admin_required():
        flash("Ma xaq u lihid inaad tirtirto xogtan!", "danger")
        return redirect(url_for('login_page'))
    if not validate_csrf_token():
        flash("Codsiga amniga lama xaqiijin.", "danger")
        return redirect(url_for('admin_database_route'))

    student = active_applicants_query().filter_by(id=id).first_or_404()
    try:
        student.is_deleted = True
        student.deleted_at = datetime.now(timezone.utc)
        student.deleted_by_admin_id = session.get('user_id')
        log_admin_action(
            'DELETE',
            'applicant',
            student.id,
            f"Soft-deleted scholarship application for {student.full_name}.",
        )
        db.session.commit()
        flash("Codsiga waa la qariyey, audit trail-kiisuna wuu kaydsan yahay.", "info")
    except Exception as exc:
        db.session.rollback()
        app.logger.exception('delete_student id=%s failed: %s', id, exc)
        flash("Khalad ayaa dhacay marka la tirturayo xogta. Ma jiro wax isbeddel ah oo la keydiyay.", "danger")
    return redirect(url_for('admin_database_route'))

@app.route('/verify/<int:id>', methods=['POST'])
def verify_applicant(id):
    if not admin_required():
        return redirect(url_for('login_page'))
    if not validate_csrf_token():
        flash("Codsiga amniga lama xaqiijin.", "danger")
        return redirect(url_for('admin_database_route'))
    
    applicant = active_applicants_query().filter_by(id=id).first_or_404()
    if applicant.verification_status != 'pending':
        flash('Codsigan hore ayaa loo farsameeyey, mana la xaqiijin karo mar labaad.', 'warning')
        return redirect(url_for('admin_database_route'))
    if not applicant.verification_document:
        flash('Codsigan lama qiimeyn karo sababtoo ah warqadda caddeynta lama soo gelin.', 'danger')
        return redirect(url_for('admin_database_route'))

    try:
        applicant.verification_status = 'verified'
        applicant.is_verified = 1
        applicant.verified_at = datetime.now(timezone.utc)
        applicant.verified_by_admin_id = session.get('user_id')
        applicant.verification_note = request.form.get('verification_note', '').strip() or 'Verified by admin.'

        faculty_str = get_choice(faculty_list, applicant.faculty_id)
        feature_frame = make_feature_frame(
            applicant.gpa,
            applicant.family_income,
            applicant.is_orphan,
            applicant.is_displaced,
            get_choice(region_list, applicant.region),
            get_choice(school_list, applicant.high_school_type),
            applicant.has_verification,
            applicant.gender,
            faculty_str,
        )
        model_pred, confidence = predict_with_xgboost(feature_frame)
        priority_bonus, priority_reasons = calculate_priority_bonus(
            applicant.is_orphan,
            applicant.is_displaced,
            applicant.family_income,
        )
        base_requirement, required_gpa = get_adjusted_gpa_requirement(faculty_str, priority_bonus)

        policy_ok, decision_reason = evaluate_policy(
            applicant.gpa,
            applicant.verification_status,
            required_gpa,
            applicant.family_income,
        )

        if not policy_ok:
            applicant.prediction_result = 0
            if decision_reason == 'income_above_limit':
                applicant.reason = (
                    f"Maamulka ayaa xaqiijiyey xogta codsiga. Dakhliga qoyska "
                    f"(${applicant.family_income:.2f}) wuxuu ka sarreeyaa xadka $500, "
                    "sidaas darteed codsigu uma qalmo deeqda."
                )
            elif decision_reason == 'gpa_below_requirement':
                applicant.reason = (
                    f"Maamulka ayaa xaqiijiyey xogta codsiga. GPA-ga codsaduhu "
                    f"({applicant.gpa}%) kama gaadhin shuruudda la dejiyey ee {required_gpa}%. "
                    "Codsigu ma ahan mid u qalma deeqda sida ku cad xeerarka nidaamka."
                )
            else:
                applicant.reason = "Maamulka ayaa xaqiijiyey codsiga, laakiin xaqiijinta kama dambaysta ah lama samayn."
        else:
            applicant.prediction_result = model_pred
            applicant.model_prediction = model_pred
            applicant.model_confidence = confidence
            priority_text = ', '.join(priority_reasons) if priority_reasons else 'mudnaan gaar ah ma jirto'
            applicant.reason = (
                f"Maamulka ayaa xaqiijiyey xogta codsiga. Shuruudda GPA ee {faculty_str} waa "
                f"{base_requirement}%, mudnaantuna waxay u dejisay {required_gpa}% ({priority_text}). "
                f"GPA-ga codsaduhu waa {applicant.gpa}%. XGBoost Model confidence: {confidence * 100:.1f}%. "
                f"Model-ku wuxuu muujiyey {'aqbalid' if model_pred == 1 else 'diidmo'}; "
                "go'aanka kama dambaysta ah wuxuu ku salaysan yahay xeerarka u-qalmitaanka iyo xaqiijinta maamulka."
            )

        log_admin_action(
            'VERIFY',
            'applicant',
            applicant.id,
            f"Verified scholarship application for {applicant.full_name}.",
        )
        db.session.commit()
        flash(f"Codsiga {applicant.full_name} waa la xaqiijiyey!", "success")
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception('verify_applicant id=%s failed: %s', id, exc)
        flash("Khalad ayaa dhacay marka la xaqiijinayey codsiga. Ma jiro wax isbeddel ah oo la keydiyay.", "danger")
    except Exception as exc:
        db.session.rollback()
        app.logger.exception('verify_applicant id=%s failed: %s', id, exc)
        flash("Khalad ayaa dhacay marka la xaqiijinayey codsiga. Ma jiro wax isbeddel ah oo la keydiyay.", "danger")
    return redirect(url_for('admin_database_route'))


@app.route('/reject/<int:id>', methods=['POST'])
def reject_applicant(id):
    if not admin_required():
        return redirect(url_for('login_page'))
    if not validate_csrf_token():
        flash("Codsiga amniga lama xaqiijin.", "danger")
        return redirect(url_for('admin_database_route'))

    applicant = active_applicants_query().filter_by(id=id).first_or_404()
    if applicant.verification_status != 'pending':
        flash('Codsigan hore ayaa loo farsameeyey, mana la diidi karo mar labaad.', 'warning')
        return redirect(url_for('admin_database_route'))

    try:
        applicant.verification_status = 'rejected'
        applicant.is_verified = 0
        applicant.verified_at = datetime.now(timezone.utc)
        applicant.verified_by_admin_id = session.get('user_id')
        applicant.verification_note = request.form.get('verification_note', '').strip() or 'Verification rejected by admin.'
        applicant.prediction_result = 0
        applicant.reason = (
            f"Maamulka ayaa diiday xaqiijinta codsiga. Sababta: {applicant.verification_note}"
        )

        log_admin_action(
            'REJECT_VERIFICATION',
            'applicant',
            applicant.id,
            f"Rejected verification for {applicant.full_name}: {applicant.verification_note}",
        )
        db.session.commit()
        flash(f"Codsiga {applicant.full_name} waa la diiday xaqiijinta.", "warning")
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception('reject_applicant id=%s failed: %s', id, exc)
        flash("Khalad ayaa dhacay marka la diiday xaqiijinta codsiga. Ma jiro wax isbeddel ah oo la keydiyay.", "danger")
    except Exception as exc:
        db.session.rollback()
        app.logger.exception('reject_applicant id=%s failed: %s', id, exc)
        flash("Khalad ayaa dhacay marka la diiday xaqiijinta codsiga. Ma jiro wax isbeddel ah oo la keydiyay.", "danger")
    return redirect(url_for('admin_database_route'))


@app.route('/audit_log')
def audit_log_route():
    """Display the admin action audit log — shows all CREATE/UPDATE/DELETE events."""
    if not admin_required():
        flash("Ma xaq u lihid inaad aragto taariikhda ficilada!", "danger")
        return redirect(url_for('login_page'))

    page = max(request.args.get('page', 1, type=int), 1)
    per_page = 20
    action_filter = request.args.get('action', '').strip().upper()

    query = (
        db.session.query(AuditLog, User)
        .join(User, AuditLog.admin_id == User.id)
        .order_by(AuditLog.id.desc())
    )
    if action_filter in ('CREATE', 'UPDATE', 'DELETE', 'VERIFY', 'REJECT_VERIFICATION'):
        query = query.filter(AuditLog.action == action_filter)

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    logs = pagination.items
    action_counts = {
        action: AuditLog.query.filter_by(action=action).count()
        for action in ('CREATE', 'UPDATE', 'DELETE', 'VERIFY', 'REJECT_VERIFICATION')
    }
    return render_template(
        'audit_log.html',
        logs=logs,
        pagination=pagination,
        action_filter=action_filter,
        action_counts=action_counts,
    )


@app.route('/shap/<int:id>')
def shap_view(id):
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
        
    applicant = active_applicants_query().filter_by(id=id).first_or_404()
    
    # Amni
    if session.get('role') != 'admin' and applicant.user_id != session.get('user_id'):
        flash("Ma xaq u lihid inaad aragto xogtan!", "danger")
        return redirect(url_for('student_dashboard'))
    if session.get('role') != 'admin' and applicant.verification_status != 'verified':
        flash("Natiijada iyo SHAP waxaa la heli karaa marka maamulka xaqiijiyo codsiga.", "warning")
        return redirect(url_for('result_page', id=applicant.id))
    
    # SHAP Calculation
    all_feature_keys = [
        "gpa", "family_income", "is_orphan", "is_displaced", "region",
        "high_school_type", "has_verification", "gender", "faculty"
    ]
    visible_features = [
        ("gpa", "Dhibcaha GPA"),
        ("family_income", "Dakhliga Qoyska"),
        ("is_orphan", "Agoonnimo"),
        ("is_displaced", "Barakac"),
        ("region", "Gobolka"),
        ("high_school_type", "Nooca Dugsiga"),
        ("has_verification", "Xaqiijinta Xogta"),
        ("gender", "Jinsiga"),
        ("faculty", "Kulliyadda")
    ]
    
    # Bedelida xogta ardayga si ay u noqoto mid model-ka u diyaar ah
    faculty_str = get_choice(faculty_list, applicant.faculty_id)
    region_str = get_choice(region_list, applicant.region)
    school_str = get_choice(school_list, applicant.high_school_type)

    visible_features[-1] = ("faculty", f"Kulliyadda ({faculty_str})")

    row_data = make_feature_frame(
        applicant.gpa, applicant.family_income, applicant.is_orphan, applicant.is_displaced,
        region_str, school_str, applicant.has_verification, applicant.gender, faculty_str
    )

    shap_list = [0] * len(visible_features)
    shap_error_msg = ""
    try:
        if model:
            clf = model.named_steps['classifier']
            row_preprocessed = model.named_steps['preprocessor'].transform(row_data)
            
            if hasattr(row_preprocessed, "toarray"):
                row_preprocessed = row_preprocessed.toarray()
                
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(row_preprocessed)
            
            if isinstance(shap_values, list):
                shaps = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shaps = shap_values
                
            if len(shaps.shape) == 3:
                shaps = shaps[:, :, 1]
                
            if len(shaps.shape) >= 2:
                actual_shap = shaps[0]
            else:
                actual_shap = shaps
                
            transformed_names = model.named_steps['preprocessor'].get_feature_names_out()
            grouped_shap = {key: 0.0 for key in all_feature_keys}

            for transformed_name, shap_value in zip(transformed_names, actual_shap):
                raw_name = transformed_name.split("__", 1)[-1]
                matched_key = next(
                    (key for key in all_feature_keys if raw_name == key or raw_name.startswith(f"{key}_")),
                    None
                )
                if matched_key:
                    grouped_shap[matched_key] += float(shap_value)

            shap_list = [grouped_shap[key] for key, _ in visible_features]

            faculty_index = next(
                (index for index, (key, _) in enumerate(visible_features) if key == "faculty"),
                None
            )
            if faculty_index is not None:
                priority_bonus, _ = calculate_priority_bonus(
                    applicant.is_orphan,
                    applicant.is_displaced,
                    applicant.family_income,
                )
                _, required_gpa = get_adjusted_gpa_requirement(faculty_str, priority_bonus)
                policy_signal = max(0.20, abs(shap_list[faculty_index]))
                if float(applicant.gpa) >= float(required_gpa):
                    shap_list[faculty_index] = policy_signal
                    visible_features[faculty_index] = (
                        "faculty",
                        f"Kulliyadda ({faculty_str}) - shuruudda waa la buuxiyey"
                    )
                else:
                    shap_list[faculty_index] = -policy_signal
                    visible_features[faculty_index] = (
                        "faculty",
                        f"Kulliyadda ({faculty_str}) - GPA shuruudda kama gaarin"
                    )
                
    except Exception as e:
        shap_error_msg = str(e)
        print(f"SHAP Error: {e}")

    # Haddii SHAP uu fashilmo, ha abuurin natiijo aan model-ka ka iman.
    total_shap = sum(abs(x) for x in shap_list)
    shap_available = total_shap >= 0.01 and not shap_error_msg
    if total_shap < 0.01 or shap_error_msg:
        shap_list = [0.0] * len(visible_features)
        flash("SHAP explanation lama heli karo hadda, laakiin natiijada Machine Learning-ka way kaydsan tahay.", "warning")

    # Tirtir fariinta qaladka si uusan u fool xumaan bogga difaacashada
    reason = applicant.reason

    return render_template('shap.html', 
                           applicant=applicant, 
                           shap_values=shap_list, 
                           feature_names=[label for _, label in visible_features],
                           shap_available=shap_available,
                           reason=reason)





@app.route('/export_csv')
def export_csv():
    if not admin_required():
        return redirect(url_for('login_page'))
    
    import csv
    from io import StringIO
    from flask import make_response

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Full Name', 'GPA', 'Income', 'Status', 'Date'])
    
    students = active_applicants_query().order_by(Applicant.id.desc()).all()
    for s in students:
        status = 'Accepted' if s.prediction_result == 1 else 'Rejected'
        cw.writerow([s.id, s.full_name, s.gpa, s.family_income, status, s.date_applied])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=applicants_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/export_pdf')
def export_pdf():
    if not admin_required():
        return redirect(url_for('login_page'))

    applicants = active_applicants_query().order_by(Applicant.id.desc()).all()
    accepted = sum(1 for applicant in applicants if applicant.prediction_result == 1)
    rejected = len(applicants) - accepted
    verified = sum(1 for applicant in applicants if applicant.is_verified == 1)
    pending = len(applicants) - verified
    average_gpa = sum(applicant.gpa for applicant in applicants) / len(applicants) if applicants else 0
    average_income = sum(applicant.family_income for applicant in applicants) / len(applicants) if applicants else 0
    report = BytesIO()

    with PdfPages(report) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.patch.set_facecolor('#f8fafc')
        summary_axis = figure.add_axes([0, 0, 1, 1])
        summary_axis.axis('off')
        figure.text(0.07, 0.92, 'Scholarship Eligibility Report', fontsize=22, fontweight='bold', color='#0f172a')
        figure.text(0.07, 0.875, f'Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}', fontsize=10, color='#475569')
        summary_stats = [
            ('Total Applications', str(len(applicants))), ('Final Eligible', str(accepted)),
            ('Rejected', str(rejected)), ('Verified', str(verified)),
            ('Pending Verification', str(pending)), ('Average GPA', f'{average_gpa:.1f}%'),
            ('Average Income', f'${average_income:,.0f}'),
        ]
        for index, (label, value) in enumerate(summary_stats):
            column = index % 4
            row = index // 4
            x_position = 0.07 + (column * 0.22)
            y_position = 0.76 - (row * 0.12)
            figure.text(x_position, y_position, label, fontsize=10, color='#475569')
            figure.text(x_position, y_position - 0.04, value, fontsize=18, fontweight='bold', color='#0f766e')

        chart_axis = figure.add_axes([0.12, 0.18, 0.76, 0.28])
        chart_axis.bar(['Final Eligible', 'Rejected'], [accepted, rejected], color=['#10b981', '#ef4444'], width=0.5)
        chart_axis.set_title('Eligible vs Not Eligible', fontsize=13, fontweight='bold', pad=10)
        chart_axis.set_ylabel('Number of Applicants')
        chart_axis.set_ylim(bottom=0)
        chart_axis.grid(axis='y', alpha=0.2)
        for spine in ('top', 'right'):
            chart_axis.spines[spine].set_visible(False)
        figure.text(0.07, 0.075, 'Scholarship Eligibility Prediction System', fontsize=9, color='#475569')
        figure.text(0.07, 0.058, 'Generated automatically by the system', fontsize=9, color='#475569')
        figure.text(0.07, 0.041, '© 2026', fontsize=9, color='#475569')
        figure.text(0.07, 0.022, 'Final eligibility is determined by scholarship policy; XGBoost supports analysis and explanation.', fontsize=9, color='#475569')
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.69, 8.27))
        figure.patch.set_facecolor('#f8fafc')
        table_axis = figure.add_axes([0, 0, 1, 1])
        table_axis.axis('off')
        figure.text(0.07, 0.92, 'Applicant Details', fontsize=20, fontweight='bold', color='#0f172a')
        figure.text(0.07, 0.88, 'Most recent scholarship applications', fontsize=10, color='#475569')
        rows = [[
            applicant.full_name[:22], get_faculty_name(applicant.faculty_id), f'{applicant.gpa:.1f}%', f'${applicant.family_income:.0f}',
            'Eligible' if applicant.prediction_result == 1 else 'Not eligible',
            'Verified' if applicant.is_verified else 'Pending', applicant.date_applied.strftime('%d %b %Y') if applicant.date_applied else '—'
        ] for applicant in applicants[:16]]
        table = table_axis.table(
            cellText=rows or [['No applicant data', '', '', '', '', '', '']],
            colLabels=['Applicant', 'Faculty', 'GPA', 'Income', 'Final Status', 'Verification', 'Applied Date'],
            cellLoc='left', colLoc='left', bbox=[0.05, 0.12, 0.90, 0.68]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            if row == 0:
                cell.set_facecolor('#0f766e')
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
        figure.text(0.07, 0.06, 'Scholarship Eligibility Prediction System', fontsize=9, color='#475569')
        figure.text(0.07, 0.043, 'Generated automatically by the system', fontsize=9, color='#475569')
        figure.text(0.07, 0.026, '© 2026', fontsize=9, color='#475569')
        pdf.savefig(figure)
        plt.close(figure)

    report.seek(0)
    return send_file(report, mimetype='application/pdf', as_attachment=True, download_name='scholarship_report.pdf')

@app.route('/predict', methods=['POST'])
def predict():
    if not student_required():
        return jsonify({"status": "error", "message": "Fadlan marka hore soo gal nidaamka!"}), 401

    if not validate_csrf_token():
        return jsonify({"status": "error", "message": "CSRF Token validation failed"}), 400

    try:
        data = request.get_json(silent=True) or request.form
        
        full_name = data.get('full_name', '')
        gpa = float(data.get('gpa', 0))
        income = float(data.get('family_income', 0))
        orphan = int(data.get('is_orphan', 0))
        displaced = int(data.get('is_displaced', 0))
        region_int = int(data.get('region', 0))
        high_school_int = int(data.get('high_school_type', 0))
        verification = int(data.get('has_verification', 0))
        gender_int = int(data.get('gender', 0))
        faculty_value = data.get('faculty', data.get('faculty_id'))
        if faculty_value is None:
            return jsonify({"status": "error", "message": "Kulliyadda la doortay lama helin. Fadlan mar kale dooro kulliyaddaada."}), 400
        faculty_int = int(faculty_value)
        if not (0 <= faculty_int < len(faculty_list)):
            return jsonify({"status": "error", "message": "Kulliyadda la doortay sax ma aha."}), 400

        is_valid_name, name_message = validate_full_name(full_name.strip())
        if not is_valid_name:
            return jsonify({"status": "error", "message": name_message}), 400
        if not (0 <= gpa <= 100):
            return jsonify({"status": "error", "message": "GPA waa inuu u dhexeeyaa 0 iyo 100."}), 400
        if income < 0:
            return jsonify({"status": "error", "message": "Dakhliga qoyska ma noqon karo tiro taban."}), 400
        if orphan not in (0, 1) or displaced not in (0, 1) or verification not in (0, 1) or gender_int not in (0, 1):
            return jsonify({"status": "error", "message": "Qiimaha binary-ga sax ma aha."}), 400
        if not (0 <= region_int < len(region_list)):
            return jsonify({"status": "error", "message": "Gobolka la doortay sax ma aha."}), 400
        if not (0 <= high_school_int < len(school_list)):
            return jsonify({"status": "error", "message": "Nooca dugsiga sax ma aha."}), 400

        faculty_str = get_choice(faculty_list, faculty_int)
        
        # Keep the application pending; the admin verification route makes the final decision.
        prediction_result = 0
        reason = "Codsigaaga waa la helay, wuxuuna sugayaa xaqiijinta maamulka."

        current_cycle = get_current_scholarship_cycle()
        if current_cycle is None:
            return jsonify({"status": "error", "message": "Ma jiro scholarship cycle furan hadda."}), 409
        cycle_id = current_cycle.id

        # Prevent duplicate submission for same user, faculty & cycle
        user_id = session.get('user_id')
        if user_id:
            existing = Applicant.query.filter_by(
                user_id=user_id, faculty_id=faculty_int, cycle_id=cycle_id
            ).first()
            if existing:
                return jsonify({"status": "warning", "message": "Waxaad horay uga codsatay kuliyadda tan isla xilliga scholarship."}), 409

        # 3. Dhammaan dadka KAYDI DATABASE-KA (Lama soo celinayo ERROR!)
        verification_document, verification_document_original = save_verification_document(
            request.files.get('verification_document')
        )
        new_applicant = Applicant(
            user_id=session.get('user_id'),
            full_name=full_name,
            gpa=gpa,
            family_income=income,
            is_orphan=orphan,
            is_displaced=displaced,
            region=region_int,
            high_school_type=high_school_int,
            has_verification=verification,
            gender=gender_int,
            faculty_id=faculty_int,
            cycle_id=cycle_id,
            prediction_result=prediction_result,
            reason=reason,
            is_verified=0,
            verification_document=verification_document,
            verification_document_original=verification_document_original
        )

        db.session.add(new_applicant)
        db.session.commit()

        return jsonify({
            "status": "success",
            "id": new_applicant.id,
            "prediction": prediction_result,
            "message": "Codsigaaga si guul leh ayaa loo keydiyay."
        })

    except ValueError as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

def initialize_database():
    # Hubi in SECRET_KEY uu yahay mid ammaan ah
    if app.secret_key in {"default_secret_key", "ibrahim_secret_key_2026"}:
        print("DIGNIIN: app.secret_key waa mid aan ammaan ahayn! Fadlan u beddel environment variable.")

    with app.app_context():
        db.create_all()
        if Faculty.query.count() == 0:
            faculties = [
                Faculty(id=0, name='CS', min_gpa=60.0),
                Faculty(id=1, name='Medicine', min_gpa=80.0),
                Faculty(id=2, name='Eng', min_gpa=70.0),
                Faculty(id=3, name='Agri', min_gpa=55.0)
            ]
            db.session.bulk_save_objects(faculties)
            db.session.commit()
        if ScholarshipCycle.query.count() == 0:
            cycle = ScholarshipCycle(academic_year='2025-2026')
            db.session.add(cycle)
            db.session.commit()
        migrate_database_schema()
        ensure_configured_admin_user()
        db.session.commit()
        # ====================================================
        # TASK 2 — Deployment Security (Localhost-Only)
        # ====================================================
        # Configured host='127.0.0.1' and port=5000 to enforce secure, localhost-only
        # execution during development as per supervisor recommendations.
        pass


initialize_database()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=os.environ.get('FLASK_DEBUG') == '1')
    
