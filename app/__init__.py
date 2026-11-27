from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

# Initialize the database globally
db = SQLAlchemy()


def split_full_name(value):
    raw_value = (value or '').strip()
    if not raw_value:
        return '', '', ''

    parts = raw_value.split()
    if len(parts) == 1:
        return parts[0], '', ''
    if len(parts) == 2:
        return parts[0], '', parts[1]
    return parts[0], ' '.join(parts[1:-1]), parts[-1]


def migrate_user_name_columns():
    inspector = inspect(db.engine)
    if 'users' not in inspector.get_table_names():
        return

    columns = {col['name'] for col in inspector.get_columns('users')}

    if 'first_name' in columns and 'last_name' in columns and 'name' not in columns:
        return

    with db.engine.begin() as conn:
        # Preserve existing data when the legacy combined `name` column still exists.
        if 'name' in columns:
            rows = conn.execute(text('SELECT id, name, email, password_hash, role, status, date_created FROM users')).fetchall()
            conn.execute(text('ALTER TABLE users RENAME TO users_legacy'))
            conn.execute(text('''
                CREATE TABLE users (
                    id INTEGER NOT NULL PRIMARY KEY,
                    first_name VARCHAR(50) NOT NULL,
                    middle_name VARCHAR(50),
                    last_name VARCHAR(50) NOT NULL,
                    email VARCHAR(120) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20),
                    status VARCHAR(20),
                    date_created DATETIME
                )
            '''))

            for row in rows:
                user_id, full_name, email, password_hash, role, status, date_created = row
                first_name, middle_name, last_name = split_full_name(full_name)
                conn.execute(
                    text('''
                        INSERT INTO users (id, first_name, middle_name, last_name, email, password_hash, role, status, date_created)
                        VALUES (:id, :first_name, :middle_name, :last_name, :email, :password_hash, :role, :status, :date_created)
                    '''),
                    {
                        'id': user_id,
                        'first_name': first_name or '',
                        'middle_name': middle_name or '',
                        'last_name': last_name or '',
                        'email': email,
                        'password_hash': password_hash,
                        'role': role,
                        'status': status,
                        'date_created': date_created,
                    }
                )

            conn.execute(text('DROP TABLE users_legacy'))
            return

        # Handles older tables that already have some split columns but still need final cleanup.
        for column in ['first_name', 'middle_name', 'last_name']:
            if column not in columns:
                conn.execute(text(f'ALTER TABLE users ADD COLUMN {column} VARCHAR(50)'))

        rows = conn.execute(text('SELECT id, name FROM users')).fetchall()
        for user_id, full_name in rows:
            first_name, middle_name, last_name = split_full_name(full_name)
            conn.execute(
                text('UPDATE users SET first_name = :first_name, middle_name = :middle_name, last_name = :last_name WHERE id = :id'),
                {
                    'first_name': first_name or '',
                    'middle_name': middle_name or '',
                    'last_name': last_name or '',
                    'id': user_id,
                }
            )


def migrate_feedback_schema():
    inspector = inspect(db.engine)
    if 'student_feedback' not in inspector.get_table_names():
        return

    columns = {col['name'] for col in inspector.get_columns('student_feedback')}
    if 'resolved_at' not in columns:
        with db.engine.begin() as conn:
            conn.execute(text('ALTER TABLE student_feedback ADD COLUMN resolved_at DATETIME'))


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Bind the database to the app
    db.init_app(app)

    # Register the routing blueprints
    from app.routes import main
    app.register_blueprint(main)

    # Automatically generate tables and seed a default admin account
    with app.app_context():
        migrate_user_name_columns()
        migrate_feedback_schema()
        db.create_all()
        
        from app.models import User
        if not User.query.filter_by(email='admin@isu.edu.ph').first():
            print("Creating default admin account...")
            default_admin = User(first_name='ISU', last_name='Administrator', email='admin@isu.edu.ph', role='Admin', status='Approved')
            default_admin.set_password('admin123')
            db.session.add(default_admin)
            db.session.commit()

    return app