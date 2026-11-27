import os
import time, datetime, re
import json
import pandas as pd
from io import StringIO
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash, session, Response, stream_with_context
from app import db
from app.models import User, StudentFeedback

try:
    from app.nlp.triage_engine import TriagePipeline
    ml_engine = TriagePipeline()
except Exception as e:
    print(f"Failed to initialize TriagePipeline: {e}")
    ml_engine = None

main = Blueprint('main', __name__)

def map_csv_columns(df):
    """Cell 2: Ingestion mapper matching thesis_system final"""
    col_lower = {c: str(c).lower().strip() for c in df.columns}
    text_keys = ['feedback', 'text', 'comment', 'review', 'description', 'raw']
    office_keys = ['office', 'department', 'target', 'unit', 'assigned']
    
    text_col = next((c for c, lc in col_lower.items() if any(k in lc for k in text_keys)), None)
    office_col = next((c for c, lc in col_lower.items() if any(k in lc for k in office_keys)), None)
    
    if not text_col:
        str_cols = df.select_dtypes(include=['object']).columns
        text_col = str_cols[0] if len(str_cols) > 0 else df.columns[0]
        
    df = df.rename(columns={text_col: 'Feedback'})
    if office_col:
        df = df.rename(columns={office_col: 'Office'})
    else:
        df['Office'] = 'Unassigned'
    return df

@main.route('/', methods=['GET', 'POST'])
@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if action == 'register':
            first_name = request.form.get('first_name', '').strip()
            middle_name = request.form.get('middle_name', '').strip()
            last_name = request.form.get('last_name', '').strip()
            confirm_password = request.form.get('confirm_password', '')

            if not first_name or not last_name:
                flash('First name and last name are required.', 'error')
                return redirect(url_for('main.login'))

            if password != confirm_password:
                flash('Password and confirm password do not match.', 'error')
                return redirect(url_for('main.login'))

            if User.query.filter_by(email=email).first():
                flash('Email address is already registered!', 'error')
                return redirect(url_for('main.login'))
            
            new_user = User(
                first_name=first_name,
                middle_name=middle_name or '',
                last_name=last_name,
                email=email,
                role='Executioner',
                status='Pending'
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account registered! Please wait for Administrator approval.', 'success')
            return redirect(url_for('main.login'))

        if email == 'developer@isu.edu.ph' and password == 'developer123':
            session['user_id'] = 9999
            session['user_name'] = 'System Developer'
            session['role'] = 'Developer'
            return redirect(url_for('main.developer_panel'))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            if user.status == 'Pending':
                flash('Your account is still pending Administrator approval.', 'error')
                return redirect(url_for('main.login'))
            
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['role'] = user.role
            return redirect(url_for('main.dashboard'))
        
        flash('Invalid email or password credentials!', 'error')
        return redirect(url_for('main.login'))

    return render_template('login.html')

@main.route('/logout')
def logout():
    session.clear()
    flash('Logged out of active session context.', 'success')
    return redirect(url_for('main.login'))

@main.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))
    return render_template('dashboard.html', username=session.get('user_name'), role=session.get('role'))

@main.route('/api/comparative_evaluation')
def comparative_evaluation():
    if 'user_id' not in session:
        return jsonify({'error': 'Access session expired'}), 401
    if not ml_engine or not ml_engine.models_loaded:
        return jsonify({'error': 'ML Engine is offline.'}), 500

    try:
        evaluation_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testt.csv')
        df = map_csv_columns(pd.read_csv(evaluation_path))
        ground_truth = df['priority_level'].astype(str).str.strip().str.capitalize().replace({
            'Cirtical': 'Critical', 'Critcal': 'Critical'
        }).tolist()
        labels = ['Low', 'Moderate', 'Critical']
        label_to_id = {label: index for index, label in enumerate(labels)}
        y_true = [label_to_id.get(label, 1) for label in ground_truth]

        start_time = time.time()
        df_valid, proposed_classes, *_ = ml_engine.process_dataframe(df.copy())
        proposed_runtime = time.time() - start_time
        cleaned_texts = df_valid['Cleaned_Text'].tolist()
        baseline_start = time.time()
        baseline_classes = ml_engine.classify_unfine_tuned_priority(cleaned_texts)
        baseline_runtime = time.time() - baseline_start
        record_count = len(y_true)

        def evaluate(predictions):
            matrix = [[sum(actual == row_label and predicted == col_label for actual, predicted in zip(y_true, predictions)) for col_label in range(3)] for row_label in range(3)]
            accuracy = sum(actual == predicted for actual, predicted in zip(y_true, predictions)) / record_count
            precision_values = []
            recall_values = []
            table = []
            for index, label in enumerate(labels):
                tp = matrix[index][index]
                fn = sum(matrix[index]) - tp
                fp = sum(matrix[row][index] for row in range(3)) - tp
                tn = record_count - tp - fn - fp
                precision_values.append(tp / (tp + fp) if tp + fp else 0)
                recall_values.append(tp / (tp + fn) if tp + fn else 0)
                table.append({'label': label, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
            expected = sum(sum(matrix[index][column] for column in range(3)) * sum(matrix[row][index] for row in range(3)) for index in range(3)) / (record_count * record_count)
            kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0
            return {
                'accuracy': f'{accuracy * 100:.2f}%',
                'precision': f'{sum(precision_values) / 3 * 100:.2f}%',
                'recall': f'{sum(recall_values) / 3 * 100:.2f}%',
                'kappa': f'{kappa:.4f}',
                'runtime': f'{proposed_runtime if predictions is proposed_classes else baseline_runtime:.4f} s',
                'truth_table': table,
            }

        return jsonify({'records': record_count, 'proposed': evaluate(proposed_classes), 'baseline': evaluate(baseline_classes)})
    except Exception as error:
        return jsonify({'error': f'Comparative evaluation failed: {error}'}), 500

@main.route('/api/users')
def get_users():
    if session.get('role') != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])

@main.route('/api/users/<action>/<int:user_id>', methods=['POST'])
def manage_user(action, user_id):
    if session.get('role') != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    if action == 'approve':
        user.status = 'Approved'
    elif action in ['delete', 'reject']:
        db.session.delete(user)
        
    db.session.commit()
    return jsonify({'success': True})

@main.route('/api/feedbacks/resolve/<int:fb_id>', methods=['POST'])
def resolve_feedback(fb_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    feedback = StudentFeedback.query.get_or_404(fb_id)
    feedback.is_resolved = True
    feedback.resolved_at = datetime.datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True})

@main.route('/api/upload', methods=['POST'])
def upload_csv():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication session expired.'}), 401
        
    file = request.files.get('csv_file') or request.files.get('file')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid format. Please select a valid .csv file.'}), 400

    if not ml_engine or not ml_engine.models_loaded:
        return jsonify({'error': 'ML Engine is offline. Check model directories.'}), 500

    try:
        df = pd.read_csv(file)
        df = map_csv_columns(df)

        (df_valid, 
         prop_classes, 
         prop_probs, 
         sent_classes, 
         emot_classes, 
         sent_confs, 
         emot_confs) = ml_engine.process_dataframe(df)

        feedbacks_imported = 0
        for i in range(len(df_valid)):
            pred_class = prop_classes[i]
            actual_priority = ml_engine.priority_map.get(pred_class, 'Moderate')
            
            # Map visual priority score for sorting
            if actual_priority == 'Critical':
                actual_score = 0.95
            elif actual_priority == 'Moderate':
                actual_score = 0.50
            else:
                actual_score = 0.10
            
            actual_sentiment = ml_engine.sent_map.get(sent_classes[i], 'Neutral')
            actual_emotion = ml_engine.emot_map.get(emot_classes[i], 'Neutral')

            feedback_node = StudentFeedback(
                raw_text=str(df_valid.iloc[i]['Feedback']),
                cleaned_text=str(df_valid.iloc[i]['Cleaned_Text']),
                source='CSV Bulk Upload',
                target_office=str(df_valid.iloc[i]['Norm_Office']),
                sentiment=actual_sentiment,
                emotion=actual_emotion,
                priority_level=actual_priority,
                priority_score=actual_score,
                sentiment_conf=float(sent_confs[i]),
                emotion_conf=float(emot_confs[i]),
                priority_conf=float(prop_probs[i])
            )
            db.session.add(feedback_node)
            feedbacks_imported += 1
            
        db.session.commit()
        return jsonify({'message': f'Successfully triaged {feedbacks_imported} reports.'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Processing failure: {str(e)}'}), 500
    
@main.route('/api/feedbacks/<path:office_name>')
def get_office_feedbacks(office_name):
    if 'user_id' not in session:
        return jsonify({'error': 'Access session expired'}), 401
    
    canonical_office = ml_engine.normalize_office(office_name) if ml_engine else office_name
    feedbacks = StudentFeedback.query.filter_by(
        target_office=canonical_office, 
        is_resolved=False
    ).order_by(StudentFeedback.priority_score.desc()).all()
    return jsonify([fb.to_dict() for fb in feedbacks])

@main.route('/api/feedbacks/dashboard_unresolved')
def get_dashboard_unresolved():
    if 'user_id' not in session:
        return jsonify({'error': 'Access session expired'}), 401

    feedbacks = StudentFeedback.query.filter_by(is_resolved=False).all()

    # Rule 1: Priority Tier Rank (Critical > Moderate > Low)
    priority_rank = {'Critical': 1, 'Moderate': 2, 'Low': 3}

    # Rule 2: Office Tie-Breaker (Infirmary -> Library -> Admin -> OSAS)
    office_order = {'Infirmary': 1, 'Library': 2, 'Admin': 3, 'OSAS': 4}

    # Rule 3: First-Come, First-Served (Ascending timestamp or ID)
    feedbacks.sort(key=lambda x: (
        priority_rank.get(x.priority_level, 4),
        office_order.get(x.target_office, 5),
        x.date_submitted if x.date_submitted is not None else datetime.datetime.max,
        x.id or 0
    ))

    return jsonify([fb.to_dict() for fb in feedbacks])

@main.route('/api/feedbacks/all')
def get_all_feedbacks():
    if 'user_id' not in session:
        return jsonify({'error': 'Access session expired'}), 401
# My task 3 route
    feedbacks = StudentFeedback.query.all()
    priority_rank = {'Critical': 1, 'Moderate': 2, 'Low': 3}
    feedbacks.sort(key=lambda feedback: (
        1 if feedback.is_resolved else 0,
        priority_rank.get(feedback.priority_level, 4),
        feedback.date_submitted if feedback.date_submitted is not None else datetime.datetime.max,
        feedback.id or 0
    ))
    return jsonify([fb.to_dict() for fb in feedbacks])

@main.route('/developer')
def developer_panel():
    if session.get('role') != 'Developer':
        return redirect(url_for('main.login'))
    return render_template('developer.html')

@main.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    if session.get('role') != 'Developer':
        return jsonify({'error': 'Unauthorized'}), 403
        
    file = request.files.get('csv_file')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format.'}), 400

    if not ml_engine or not ml_engine.models_loaded:
        return jsonify({'error': 'ML Engine is offline.'}), 500

    file_content = file.read().decode('utf-8', errors='replace')

    def generate_pipeline_stream():
        try:
            start_time = time.time()
            
            # Step 1: Ingestion
            yield json.dumps({"status": "progress", "step": 1, "log": "Data Ingestion: Ingesting dataset into memory..."}) + "\n"
            df = pd.read_csv(StringIO(file_content))
            df = map_csv_columns(df)

            # Step 2: Harmonization
            yield json.dumps({"status": "progress", "step": 2, "log": "Data Harmonization: Normalizing text tokens..."}) + "\n"
            raw_texts, cleaned_texts = ml_engine.harmonize_data(df)
            csv_offices = df['Office'].astype(str).tolist() if 'Office' in df.columns else []
            aligned_offices = [ml_engine.normalize_office(csv_offices[i]) if i < len(csv_offices) else 'Unassigned' for i in range(len(cleaned_texts))]

            # Step 3: BERT Feature Extraction
            yield json.dumps({"status": "progress", "step": 3, "log": "Feature Fusion: Generating the trained 39-D stacked feature matrix..."}) + "\n"
            features_data = ml_engine.extract_features(cleaned_texts, aligned_offices)
            numpy_features, sentiment_preds, emotion_preds, sentiment_probs, emotion_probs = features_data
            
            # Step 4: XGBoost Classification
            yield json.dumps({"status": "progress", "step": 4, "log": "XGBoost Classification: Evaluating priority levels..."}) + "\n"
            ml_predictions, priority_probs = ml_engine.classify_triage(numpy_features)
            
            # Step 5: Benchmark Evaluation
            yield json.dumps({"status": "progress", "step": 5, "log": "Evaluation: Benchmarking against ground truth..."}) + "\n"
            runtime = time.time() - start_time

            results_data = []
            gt_pri_col = next((c for c in df.columns if 'priority' in str(c).lower()), None)
            date_col = next((c for c in df.columns if any(d in str(c).lower() for d in ['date', 'time', 'submitted', 'timestamp'])), None)

            tp = tn = fp = fn = 0
            for i in range(len(raw_texts)):
                if not str(raw_texts[i]).strip() or str(raw_texts[i]) == 'nan':
                    continue
                
                pred_class = ml_predictions[i]
                pred_priority = ml_engine.priority_map.get(pred_class, 'Moderate')
                actual_sentiment = ml_engine.sent_map.get(sentiment_preds[i], 'Neutral')
                actual_emotion = ml_engine.emot_map.get(emotion_preds[i], 'Neutral')
                
                true_priority = "N/A"
                if gt_pri_col:
                    val = str(df.iloc[i][gt_pri_col]).strip().capitalize()
                    true_priority = "Critical" if val in ["Critical", "Cirtical", "Critcal"] else val
                    
                    t_low = true_priority.lower()
                    p_low = pred_priority.lower()
                    if p_low == 'critical' and t_low == 'critical': tp += 1
                    elif p_low == 'critical' and t_low != 'critical': fp += 1
                    elif p_low != 'critical' and t_low == 'critical': fn += 1
                    elif p_low != 'critical' and t_low != 'critical': tn += 1

                date_val = str(df.iloc[i][date_col]).strip() if date_col else None

                results_data.append({
                    'original_index': i,
                    'date_submitted': date_val if date_val and date_val != 'nan' else None,
                    'text': str(raw_texts[i]),
                    'office': aligned_offices[i],
                    'sentiment': actual_sentiment,
                    'emotion': actual_emotion,
                    'predicted_priority': pred_priority,
                    'true_priority': true_priority
                })

            metrics = {}
            if gt_pri_col and (tp + tn + fp + fn) > 0:
                total_eval = tp + tn + fp + fn
                accuracy = (tp + tn) / total_eval
                prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0

                prec_nc = tn / (tn + fn) if (tn + fn) > 0 else 0
                rec_nc = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_nc = (2 * prec_nc * rec_nc) / (prec_nc + rec_nc) if (prec_nc + rec_nc) > 0 else 0
                macro_f1 = (f1_c + f1_nc) / 2

                p_o = accuracy
                p_yes = ((tp + fn) / total_eval) * ((tp + fp) / total_eval)
                p_no = ((fp + tn) / total_eval) * ((fn + tn) / total_eval)
                kappa = (p_o - (p_yes + p_no)) / (1 - (p_yes + p_no)) if (1 - (p_yes + p_no)) > 0 else 0

                metrics = {
                    'evaluated': total_eval,
                    'accuracy': f"{accuracy*100:.2f}",
                    'precision': f"{prec_c*100:.2f}",
                    'recall': f"{rec_c*100:.2f}",
                    'macro_f1': f"{macro_f1*100:.2f}",
                    'kappa': f"{kappa:.4f}"
                }

            yield json.dumps({
                'status': 'complete',
                'success': True,
                'runtime': f"{runtime:.4f}",
                'rows_processed': len(results_data),
                'results_data': results_data,
                'metrics': metrics
            }) + "\n"

        except Exception as e:
            yield json.dumps({'status': 'error', 'error': str(e)}) + "\n"

    return Response(stream_with_context(generate_pipeline_stream()), mimetype='application/x-ndjson')

@main.route('/api/terminate_developer', methods=['POST'])
def terminate_developer():
    if session.get('role') != 'Developer':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({'success': True})