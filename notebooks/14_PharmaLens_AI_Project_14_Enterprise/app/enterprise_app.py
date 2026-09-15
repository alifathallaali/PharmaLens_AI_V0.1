import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enterprise.database import SessionLocal, init_db
from enterprise.auth import authenticate_user
from enterprise.permissions import has_permission
from enterprise.models import User
from enterprise.data_upload import save_dataset

init_db()
st.set_page_config(page_title="PharmaLens AI Enterprise", page_icon="💊", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

def login_page():
    st.title("💊 PharmaLens AI")
    st.subheader("Enterprise Pharmaceutical Intelligence")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            db = SessionLocal()
            try:
                user = authenticate_user(db, email, password)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.company_id = user.company_id
                    st.session_state.role = user.role
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
            finally:
                db.close()

def main_app():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == st.session_state.user_id).first()
        if not user:
            st.session_state.clear()
            st.rerun()

        st.sidebar.success(f"Logged in as {user.full_name}")
        st.sidebar.caption(f"Role: {user.role}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.title("PharmaLens AI Enterprise")
        st.write(f"Company: {user.company.name}")
        st.divider()

        modules = [
            ("dashboard","Dashboard"), ("market","Market Intelligence"),
            ("brand","Brand Explorer"), ("molecule","Molecule Explorer"),
            ("company","Company Intelligence"), ("forecast","Forecast"),
            ("launch","Launch Intelligence"), ("gtm","GTM Intelligence"),
            ("recommendation","Recommendations"), ("similarity","Drug Similarity"),
            ("agent","AI Agent"), ("reports","Reports"), ("upload","Private Data Upload")
        ]
        pages = [label for key, label in modules if has_permission(user, key)]
        page = st.sidebar.selectbox("Module", pages)

        if page == "Private Data Upload":
            st.header("Private Company Data")
            workspace_id = st.number_input("Workspace ID", min_value=1, step=1)
            uploaded = st.file_uploader("Upload CSV / Excel / Parquet",
                                        type=["csv","xlsx","parquet"])
            if uploaded and st.button("Save Dataset"):
                try:
                    dataset = save_dataset(db, uploaded, int(workspace_id), user.id)
                    st.success(f"Saved dataset: {dataset.file_name}")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.header(page)
            st.info("Connect this page to the corresponding existing Project 01–13 module.")
    finally:
        db.close()

if st.session_state.user_id is None:
    login_page()
else:
    main_app()
