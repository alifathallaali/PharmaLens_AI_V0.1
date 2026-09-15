REQUIRED_COLUMNS = {
    "Distribution Channel", "Therapeutic Class", "Manufacturer", "Brand Name",
    "Pack Size", "Product Launch", "Drug Strength", "Selling Price",
    "Market Category", "Month", "Year", "Sales Units", "Sales Value"
}

def validate_dataframe(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    return {
        "valid": not missing,
        "missing_columns": sorted(missing),
        "message": "Dataset passed validation." if not missing
                   else "Required columns are missing."
    }
