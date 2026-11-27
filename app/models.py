from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False, default='')
    middle_name = db.Column(db.String(50), nullable=True, default='')
    last_name = db.Column(db.String(50), nullable=False, default='')
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # NEW: Role and Status for Access Control
    role = db.Column(db.String(20), default='Executioner') # Admin or Executioner
    status = db.Column(db.String(20), default='Pending')   # Pending or Approved
    
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def name(self):
        return ' '.join(part for part in [self.first_name, self.middle_name, self.last_name] if part)

    @name.setter
    def name(self, value):
        raw_value = (value or '').strip()
        if not raw_value:
            self.first_name = ''
            self.middle_name = ''
            self.last_name = ''
            return

        parts = raw_value.split()
        self.first_name = parts[0] if parts else ''
        if len(parts) >= 3:
            self.middle_name = ' '.join(parts[1:-1])
            self.last_name = parts[-1]
        elif len(parts) == 2:
            self.middle_name = ''
            self.last_name = parts[1]
        else:
            self.middle_name = ''
            self.last_name = ''

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'first_name': self.first_name,
            'middle_name': self.middle_name,
            'last_name': self.last_name,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'status': self.status,
            'date': self.date_created.strftime("%Y-%m-%d")
        }

class StudentFeedback(db.Model):
    __tablename__ = 'student_feedback'

    id = db.Column(db.Integer, primary_key=True)
    
    raw_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text, nullable=True) 
    source = db.Column(db.String(50), default='CSV Ingestion')
    date_submitted = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)
    
    target_office = db.Column(db.String(50), nullable=False) 
    sentiment = db.Column(db.String(20), nullable=False)     
    emotion = db.Column(db.String(50), nullable=False)       
    
    priority_level = db.Column(db.String(20), nullable=False) 
    priority_score = db.Column(db.Float, nullable=False)      
    is_resolved = db.Column(db.Boolean, default=False)

    sentiment_conf = db.Column(db.Float, nullable=True, default=0.0)
    emotion_conf = db.Column(db.Float, nullable=True, default=0.0)
    priority_conf = db.Column(db.Float, nullable=True, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.raw_text,
            'cleaned_text': self.cleaned_text,
            'source': self.source,
            'office': self.target_office,
            'sentiment': self.sentiment,
            'emotion': self.emotion,
            'priority': self.priority_level,
            'priority_score': round(self.priority_score, 4),
            'sentiment_conf': round(self.sentiment_conf * 100) if self.sentiment_conf else 0,
            'emotion_conf': round(self.emotion_conf * 100) if self.emotion_conf else 0,
            'priority_conf': round(self.priority_conf * 100) if self.priority_conf else 0,
            'is_resolved': self.is_resolved,
            'date': self.date_submitted.strftime("%Y-%m-%d %H:%M") if self.date_submitted else None,
            'resolved_at': self.resolved_at.strftime("%Y-%m-%d %H:%M") if self.resolved_at else None
        }