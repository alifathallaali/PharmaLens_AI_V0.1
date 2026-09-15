# PharmaLens AI — Project 14: Enterprise

Enterprise foundation for the completed PharmaLens AI platform.

## Included
- Company/workspace architecture
- User accounts and roles
- Password hashing
- Private company data upload
- Dataset validation
- Subscription limits foundation
- Audit logging
- SQLite database
- FastAPI starter
- Streamlit Enterprise shell
- Bridge for existing processed Parquet data

## Run
From the PharmaLens AI project root:

```bash
pip install -r requirements_project14.txt
python -m enterprise.create_admin
streamlit run app/enterprise_app.py
```

API:

```bash
uvicorn api.main:app --reload
```

Health endpoint: `/health`

Demo login:
- Email: admin@pharmalens.ai
- Password: ChangeThisPassword123!

Change the password before production use.

## Data
Existing shared processed data:
`data/processed/*.parquet`

Private company data:
`data/private/workspace_<id>/`

Do not commit private data, `.env`, or the SQLite database.

## Production roadmap
PostgreSQL → JWT/OAuth → object storage → tenant isolation → billing →
HTTPS → secrets management → monitoring → backups.
