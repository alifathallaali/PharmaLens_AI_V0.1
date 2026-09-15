# ============================================================
# PHARMALENS AI
# PROJECT 13 — STREAMLIT INTEGRATION SNIPPET
# ============================================================
#
# Add this to your existing Streamlit app.
# Do NOT replace your whole Streamlit application with this file.
# ============================================================

from reports.pdf_generator import generate_executive_report


def render_report_generator():
    st.subheader("Executive Report")

    st.write(
        "Generate a management-ready PharmaLens AI PDF report "
        "from the processed pharmaceutical market data."
    )

    if st.button(
        "Generate Executive Report",
        type="primary",
    ):
        with st.spinner(
            "Generating PharmaLens Executive Report..."
        ):
            report_path = generate_executive_report()

        st.success("Executive report generated successfully.")

        with open(report_path, "rb") as file:
            st.download_button(
                label="Download Executive Report",
                data=file.read(),
                file_name=report_path.name,
                mime="application/pdf",
            )
