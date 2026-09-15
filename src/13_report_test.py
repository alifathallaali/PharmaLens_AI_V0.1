# ============================================================
# PHARMALENS AI
# PROJECT 13 — PDF REPORT TEST
# ============================================================

from reports.pdf_generator import generate_executive_report


if __name__ == "__main__":
    print("=" * 65)
    print("PHARMALENS AI")
    print("PROJECT 13 — REPORT / PDF GENERATOR")
    print("=" * 65)

    try:
        report_path = generate_executive_report()

        print("\nSUCCESS")
        print(f"PDF generated at:\n{report_path}")

    except Exception as exc:
        print("\nREPORT GENERATION FAILED")
        print(f"{type(exc).__name__}: {exc}")
        raise
