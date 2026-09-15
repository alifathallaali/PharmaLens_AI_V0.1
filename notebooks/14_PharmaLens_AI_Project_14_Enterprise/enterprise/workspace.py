from sqlalchemy.orm import Session
from enterprise.models import Company, Workspace

def create_company(db: Session, name: str, country: str = "Egypt"):
    company = Company(name=name, country=country, industry="Pharmaceutical")
    db.add(company); db.commit(); db.refresh(company)
    return company

def create_workspace(db: Session, company_id: int, name: str):
    workspace = Workspace(name=name, company_id=company_id)
    db.add(workspace); db.commit(); db.refresh(workspace)
    return workspace

def get_company_workspaces(db: Session, company_id: int):
    return db.query(Workspace).filter(Workspace.company_id == company_id).all()
