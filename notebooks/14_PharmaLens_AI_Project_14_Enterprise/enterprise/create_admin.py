from enterprise.database import SessionLocal, init_db
from enterprise.workspace import create_company, create_workspace
from enterprise.auth import create_user

def main():
    init_db()
    db = SessionLocal()
    try:
        company = create_company(db, "PharmaLens Demo Company", "Egypt")
        workspace = create_workspace(db, company.id, "Main Workspace")
        user = create_user(db, "admin@pharmalens.ai",
                           "ChangeThisPassword123!",
                           "PharmaLens Administrator",
                           company.id, "admin")
        print("Enterprise account created.")
        print(f"Company ID: {company.id}")
        print(f"Workspace ID: {workspace.id}")
        print(f"Admin: {user.email}")
        print("Change the example password before production use.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
