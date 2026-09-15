from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from enterprise.database import get_db, init_db
from enterprise.models import Company, Workspace

app = FastAPI(title="PharmaLens AI Enterprise API",
              description="Enterprise pharmaceutical market intelligence platform.",
              version="1.0.0")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "PharmaLens AI", "version": "1.0.0"}

@app.get("/companies")
def get_companies(db: Session = Depends(get_db)):
    return [{"id": c.id, "name": c.name, "country": c.country, "industry": c.industry}
            for c in db.query(Company).all()]

@app.get("/companies/{company_id}/workspaces")
def get_workspaces(company_id: int, db: Session = Depends(get_db)):
    if not db.query(Company).filter(Company.id == company_id).first():
        raise HTTPException(status_code=404, detail="Company not found.")
    return [{"id": w.id, "name": w.name, "company_id": w.company_id}
            for w in db.query(Workspace).filter(Workspace.company_id == company_id).all()]
