from enterprise.models import AuditLog

def log_action(db, user_id: int, action: str, details: str = ""):
    log = AuditLog(user_id=user_id, action=action, details=details)
    db.add(log); db.commit(); db.refresh(log)
    return log
