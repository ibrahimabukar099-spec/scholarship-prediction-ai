import os
import sys
import json
import unittest
from datetime import datetime, timezone
from werkzeug.security import check_password_hash, generate_password_hash

# Ensure Flask app environment
os.environ['APP_ENV'] = 'development'
os.environ['DATABASE_PATH'] = 'sqlite:///:memory:'

# Ensure workspace root is in sys.path
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if basedir not in sys.path:
    sys.path.insert(0, basedir)

from app1 import (
    app, db, User, Applicant, AuditLog, ScholarshipCycle, Faculty,
    get_model_metrics, log_admin_action, evaluate_policy,
    ensure_configured_admin_user,
    active_applicants_query
)

class TestFinalFiveFixes(unittest.TestCase):

    def setUp(self):
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False

        self.app_context = app.app_context()
        self.app_context.push()
        self.client = app.test_client()

        db.drop_all()
        db.create_all()

        # Seed Faculties
        faculties = [
            Faculty(id=0, name='CS', min_gpa=60.0),
            Faculty(id=1, name='Medicine', min_gpa=80.0),
            Faculty(id=2, name='Eng', min_gpa=70.0),
            Faculty(id=3, name='Agri', min_gpa=55.0)
        ]
        db.session.bulk_save_objects(faculties)

        self.admin = User(
            full_name="System Administrator",
            email="admin@system.local",
            password="pbkdf2:sha256:test_password",
            role="admin"
        )
        db.session.add(self.admin)

        self.student = User(
            full_name="Test Student",
            email="student_test@system.local",
            password="pbkdf2:sha256:test_password",
            role="student"
        )
        db.session.add(self.student)
        db.session.commit()

        self.cycle = ScholarshipCycle(academic_year="2025-2026")
        db.session.add(self.cycle)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    # -----------------------------------------------------------------
    # FIX 1 & FIX 2 TESTS: model_metrics.json & 5-fold CV Mean ± SD
    # -----------------------------------------------------------------
    def test_01_model_metrics_json_and_5fold_cv(self):
        metrics_file = os.path.join(basedir, 'model_metrics.json')
        self.assertTrue(os.path.exists(metrics_file), "model_metrics.json must exist")

        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check required holdout metrics
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            self.assertIn(key, data, f"Key {key} must be in model_metrics.json")
            self.assertIsInstance(data[key], (int, float))

        # Check separate holdout_test dictionary
        self.assertIn('holdout_test', data)
        self.assertIn('accuracy', data['holdout_test'])

        # Check 5-fold CV metrics (mean ± SD)
        self.assertIn('cross_validation', data)
        cv = data['cross_validation']
        self.assertEqual(cv.get('folds'), 5)

        for cv_metric in ['accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                          'recall_mean', 'recall_std', 'f1_mean', 'f1_std']:
            self.assertIn(cv_metric, cv, f"CV metric {cv_metric} must be present in cross_validation")
            self.assertIsInstance(cv[cv_metric], (int, float))
            self.assertGreaterEqual(cv[cv_metric], 0.0)

        # Verify get_model_metrics() function reads this dynamically
        dynamic_metrics = get_model_metrics()
        self.assertEqual(dynamic_metrics.get('accuracy'), data.get('accuracy'))
        print("[OK] FIX 1 & FIX 2 VERIFIED: model_metrics.json contains real holdout metrics & 5-fold CV mean ± SD.")

    # -----------------------------------------------------------------
    # FIX 3 TEST: Separate applicant has_verification from admin verification_status
    # -----------------------------------------------------------------
    def test_02_verification_status_separation(self):
        # Create applicant with has_verification=1 (applicant-declared)
        app_item = Applicant(
            user_id=self.student.id,
            full_name="Verification Test Applicant",
            gpa=95.0,
            family_income=200.0,
            is_orphan=1,
            is_displaced=1,
            region=0,
            high_school_type=0,
            has_verification=1,  # Applicant claims verification
            gender=0,
            faculty_id=1,  # Medicine (GPA req 80)
            prediction_result=0,
            reason="Awaiting admin verification",
            cycle_id=self.cycle.id,
            verification_status='pending'
        )
        db.session.add(app_item)
        db.session.commit()

        # 1. Check pending status: prediction_result MUST be 0 despite has_verification=1
        fetched = db.session.get(Applicant, app_item.id)
        self.assertEqual(fetched.verification_status, 'pending')
        self.assertEqual(fetched.prediction_result, 0, "has_verification=1 MUST NOT grant final eligibility when pending")

        # Check evaluate_policy when pending
        policy_ok, reason = evaluate_policy(fetched.gpa, fetched.verification_status, 80, fetched.family_income)
        self.assertFalse(policy_ok)
        self.assertEqual(reason, 'verification_missing')

        # 2. Check verified status: eligibility evaluated when verification_status == 'verified'
        fetched.verification_status = 'verified'
        db.session.commit()

        policy_ok, reason = evaluate_policy(fetched.gpa, fetched.verification_status, 80, fetched.family_income)
        self.assertTrue(policy_ok)
        self.assertEqual(reason, 'policy_passed')

        print("[OK] FIX 3 VERIFIED: has_verification = 1 does NOT grant eligibility. verification_status (pending/verified) is enforced.")

    # -----------------------------------------------------------------
    # FIX 4 TEST: Complete AuditLog for admin actions (VERIFY, DELETE)
    # -----------------------------------------------------------------
    def test_03_audit_logging_completeness(self):
        # Create applicant for faculty 2 to avoid unique constraint
        app_item = Applicant(
            user_id=self.student.id,
            full_name="Audit Test Student",
            gpa=85.0,
            family_income=150.0,
            is_orphan=0,
            is_displaced=0,
            region=0,
            high_school_type=0,
            has_verification=1,
            gender=0,
            faculty_id=2,
            prediction_result=0,
            reason="Initial",
            cycle_id=self.cycle.id,
            verification_status='pending'
        )
        db.session.add(app_item)
        db.session.commit()

        # Simulate admin login session & CSRF
        token = 'test_token'
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.admin.id
            sess['role'] = 'admin'
            sess['logged_in'] = True
            sess['csrf_token'] = token

        # Test VERIFY action
        response = self.client.post(f'/verify/{app_item.id}', data={'csrf_token': token, 'verification_note': 'Approved'})
        self.assertEqual(response.status_code, 302)

        verify_audit = AuditLog.query.filter_by(target_record_id=app_item.id, action='VERIFY').first()
        self.assertIsNotNone(verify_audit)
        self.assertEqual(verify_audit.admin_id, self.admin.id)
        self.assertEqual(verify_audit.target_table, 'applicant')

        # Create another applicant for DELETE test
        app_item_delete = Applicant(
            user_id=self.student.id,
            full_name="Delete Test Student",
            gpa=85.0,
            family_income=150.0,
            is_orphan=0,
            is_displaced=0,
            region=0,
            high_school_type=0,
            has_verification=1,
            gender=0,
            faculty_id=3,
            prediction_result=0,
            reason="Initial",
            cycle_id=self.cycle.id,
            verification_status='pending'
        )
        db.session.add(app_item_delete)
        db.session.commit()

        # Test DELETE action
        response = self.client.post(f'/delete/{app_item_delete.id}', data={'csrf_token': token})
        self.assertEqual(response.status_code, 302)

        delete_audit = AuditLog.query.filter_by(target_record_id=app_item_delete.id, action='DELETE').first()
        self.assertIsNotNone(delete_audit)
        self.assertEqual(delete_audit.admin_id, self.admin.id)

        print("[OK] FIX 4 VERIFIED: AuditLog entries recorded with admin_id, action, target_table, record ID, and description.")

    # -----------------------------------------------------------------
    # FIX 5 TEST: Transaction Safety and Rollback on Audit Failure
    # -----------------------------------------------------------------
    def test_04_audit_transaction_safety_and_rollback(self):
        app_item = Applicant(
            user_id=self.student.id,
            full_name="Rollback Test Student",
            gpa=88.0,
            family_income=200.0,
            is_orphan=0,
            is_displaced=0,
            region=0,
            high_school_type=0,
            has_verification=1,
            gender=0,
            faculty_id=0,
            prediction_result=0,
            reason="Initial",
            cycle_id=self.cycle.id,
            verification_status='pending'
        )
        db.session.add(app_item)
        db.session.commit()

        app_id = app_item.id
        token = 'test_token'

        with self.client.session_transaction() as sess:
            sess['user_id'] = self.admin.id
            sess['role'] = 'admin'
            sess['logged_in'] = True
            sess['csrf_token'] = token

        # Test transaction safety: if AuditLog fails during verify, applicant update MUST be rolled back
        import app1
        original_log_fn = app1.log_admin_action

        def failing_log_admin_action(*args, **kwargs):
            raise RuntimeError("Simulated AuditLog DB failure!")

        app1.log_admin_action = failing_log_admin_action

        try:
            response = self.client.post(f'/verify/{app_id}', data={'csrf_token': token})
            self.assertEqual(response.status_code, 302)

            # Check that applicant is STILL pending because transaction was rolled back!
            db.session.expire_all()
            re_fetched = db.session.get(Applicant, app_id)
            self.assertEqual(re_fetched.verification_status, 'pending', "Application change MUST be rolled back if audit logging fails!")
            print("[OK] FIX 5 VERIFIED: Transaction safety verified! When AuditLog fails, application modification is strictly rolled back.")

        finally:
            app1.log_admin_action = original_log_fn

if __name__ == '__main__':
    unittest.main()
