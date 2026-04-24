from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "ibrahim_secret_key_2026" 
CORS(app)

# --- DATABASE SETUP ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'scholarship_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Applicant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    magaca = db.Column(db.String(100))
    gpa = db.Column(db.Float)
    family_income = db.Column(db.Float)
    is_orphan = db.Column(db.Integer)
    is_displaced = db.Column(db.Integer)
    prediction_result = db.Column(db.Integer)
    reason = db.Column(db.String(1000))

# --- LOAD AI MODEL ---
try:
    model = joblib.load('xgboost_scholarship_final.pkl')
except Exception as e:
    print(f"Model-ka lama rari karin: {e}")
    model = None

# --- ROUTES ---

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    role = request.form.get('role', '').strip().lower()
    username = request.form.get('username', '').strip().lower()
    password = request.form.get('password', '').strip()

    if role == "admin":
        # Admin Login Data
        if username == "admin" and password == "12345":
            session['logged_in'] = True
            return redirect(url_for('admin_database_route'))
        else:
            flash("Username ama Password-ka Admin-ka waa khalad!", "danger")
            return redirect(url_for('login_page'))

    return redirect(url_for('student_form'))

@app.route('/admin_database')
def admin_database_route():
    search_query = request.args.get('search', '').strip()
    if search_query:
        students = Applicant.query.filter(Applicant.magaca.like(f"%{search_query}%")).all()
    else:
        students = Applicant.query.all()
    
    total_apps = Applicant.query.count()
    accepted = Applicant.query.filter_by(prediction_result=1).count()
    rejected = Applicant.query.filter_by(prediction_result=0).count()
    orphans = Applicant.query.filter_by(is_orphan=1).count()

    return render_template('admin.html', 
                           students=students, 
                           total=total_apps, 
                           accepted=accepted, 
                           rejected=rejected, 
                           orphans=orphans)

@app.route('/apply_now')
def student_form():
    return render_template('form.html')

@app.route('/result/<int:id>')
def result_page(id):
    student = Applicant.query.get_or_404(id)
    return render_template('result.html', s=student)

@app.route('/delete/<int:id>')
def delete_student(id):
    student = Applicant.query.get_or_404(id)
    db.session.delete(student)
    db.session.commit()
    flash("Xogta ardayga waa la tirturay!", "info")
    return redirect(url_for('admin_database_route'))

@app.route('/shap/<int:id>')
def shap_view(id):
    applicant = Applicant.query.get_or_404(id)
    return render_template('shap.html', applicant=applicant)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Soo qabashada xogta
        gpa = float(data.get('gpa', 0))
        income = float(data.get('family_income', 0))
        orphan = int(data.get('is_orphan', 0))
        displaced = int(data.get('is_displaced', 0))
        verification = int(data.get('has_verification', 1))

        features = [
            gpa, income,
            int(data.get('parents_education', 0)), orphan,
            displaced, int(data.get('region', 0)),
            int(data.get('parent_occupation', 0)), int(data.get('gap_years', 0)),
            int(data.get('high_school_type', 0)), verification,
            int(data.get('gender', 0)), int(data.get('faculty', 0))
        ]

        # AI Prediction
        if model:
            pred = int(model.predict(np.array([features]))[0])
        else:
            pred = 0 

        # --- DYNAMIC REASONING LOGIC (XALKA KAMA DAMBAYSTA AH) ---
        reasons_list = []

        if pred == 1:
            reason_text = "Hambalyo! Analysis-ka AI wuxuu muujiyey inaad si buuxda u buuxisay shuruudihii deeqda waxbarasho ee loo dejiyey."
        else:
            # 1. Hubinta GPA (Kaliya sheeg haddii uu ka yaryahay 50%)
            if gpa < 50:
                reasons_list.append(f"GPA-gaaga ({gpa}%) aad ayuu u hooseeyaa, taas oo saameyn weyn ku yeelatay go'aanka AI-da.")
            
            # 2. Hubinta Dakhliga
            if income > 500:
                reasons_list.append(f"Dakhliga qoyskaaga (${income}) ayaa loo arkaa mid sareeya oo caqabad ku noqon kara helitaanka deeqda.")
            
            # 3. Hubinta Xaaladda Bulshada
            if orphan == 0 and displaced == 0:
                reasons_list.append("Ma muujin baahi bulsho oo gaar ah (Agoon ama Barakac) oo mudnaan dheeraad ah ku siin karta.")
            
            # 4. Hubinta Caddaynta
            if verification == 0:
                reasons_list.append("Waxaad soo gudbisay codsi aan wadan caddaymihii rasmiga ahaa (Verification: No).")

            # Isku xirka sababaha
            if reasons_list:
                reason_text = "Sababaha laguu diiday: " + " ".join(reasons_list)
            else:
                reason_text = "Analysis-ka AI wuxuu muujiyey in natiijadaada iyo tartanka guud aysan is waafaqsanayn sanadkan."

        new_entry = Applicant(
            magaca=data.get('magaca'), 
            gpa=gpa,
            family_income=income,
            is_orphan=orphan, 
            is_displaced=displaced,
            prediction_result=pred, 
            reason=reason_text
        )
        db.session.add(new_entry)
        db.session.commit()
        return jsonify({"status": "success", "id": new_entry.id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)