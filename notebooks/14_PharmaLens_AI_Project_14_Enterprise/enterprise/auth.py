from sqlalchemy.orm import Session
from enterprise.models import User
from enterprise.security import hash_password, verify_password

def create_user(db: Session, email: str, password: str, full_name: str,
                company_id: int, role: str = "analyst"):
    email = email.lower().strip()
    if db.query(User).filter(User.email == email).first():
        raise ValueError("User already exists.")
    user = User(email=email, password_hash=hash_password(password),
                full_name=full_name, company_id=company_id, role=role)
    db.add(user); db.commit(); db.refresh(user)
    return user

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not user.is_active:
        return None
    return user if verify_password(password, user.password_hash) else None
