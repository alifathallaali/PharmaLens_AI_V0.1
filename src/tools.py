# ============================================================
# PharmaLens AI
# Project 09 - AI Agent Tools
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

from typing import Optional, Dict, Any, List


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# ============================================================
# DATA LOADING
# ============================================================

# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cleaned_pharma_data.parquet"
)


def load_data() -> pd.DataFrame:
    """
    Load the cleaned PharmaLens processed dataset.

    Source:
        data/processed/*.parquet

    The processed dataset contains the canonical
    PharmaLens column names.
    """

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Processed dataset not found at: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    # Standardize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df

# ============================================================
# DATA VALIDATION
# ============================================================

EXPECTED_COLUMNS = [
    "Distribution Channel",
    "Therapeutic Class",
    "Manufacturer",
    "Brand Name",
    "Pack Size",
    "Product Launch",
    "Drug Strength",
    "Selling Price",
    "Market Category",
    "Month",
    "Year",
    "Sales Units",
    "Sales Value"
]


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_columns": missing_columns,
        "valid": len(missing_columns) == 0
    }


# ============================================================
# GENERAL DATA SUMMARY
# ============================================================

def market_summary() -> Dict[str, Any]:

    df = load_data()

    total_sales = df["Sales Value"].sum()
    total_units = df["Sales Units"].sum()

    manufacturers = df["Manufacturer"].nunique()
    brands = df["Brand Name"].nunique()
    therapeutic_classes = df["Therapeutic Class"].nunique()

    return {
        "total_sales_value": float(total_sales),
        "total_sales_units": float(total_units),
        "manufacturers": int(manufacturers),
        "brands": int(brands),
        "therapeutic_classes": int(therapeutic_classes),
        "rows": int(len(df))
    }


# ============================================================
# MARKET TOOL
# ============================================================

def market_tool(
    therapeutic_class: Optional[str] = None,
    year: Optional[int] = None
) -> Dict[str, Any]:

    df = load_data()

    if therapeutic_class:
        df = df[
            df["Therapeutic Class"]
            .astype(str)
            .str.contains(
                therapeutic_class,
                case=False,
                na=False
            )
        ]

    if year:
        df = df[df["Year"] == year]

    if df.empty:
        return {
            "status": "no_data",
            "message": "No market data found."
        }

    sales = df["Sales Value"].sum()
    units = df["Sales Units"].sum()

    top_brands = (
        df.groupby("Brand Name")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    top_manufacturers = (
        df.groupby("Manufacturer")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "status": "success",
        "sales_value": float(sales),
        "sales_units": float(units),
        "top_brands": top_brands.to_dict(),
        "top_manufacturers": top_manufacturers.to_dict()
    }


# ============================================================
# BRAND TOOL
# ============================================================

def brand_tool(
    brand_name: str
) -> Dict[str, Any]:

    df = load_data()

    data = df[
        df["Brand Name"]
        .astype(str)
        .str.contains(
            brand_name,
            case=False,
            na=False
        )
    ]

    if data.empty:

        return {
            "status": "not_found",
            "brand": brand_name
        }

    sales = data["Sales Value"].sum()
    units = data["Sales Units"].sum()

    manufacturer = (
        data["Manufacturer"]
        .mode()
        .iloc[0]
        if not data["Manufacturer"].mode().empty
        else None
    )

    therapeutic_class = (
        data["Therapeutic Class"]
        .mode()
        .iloc[0]
        if not data["Therapeutic Class"].mode().empty
        else None
    )

    average_price = data["Selling Price"].mean()

    return {
        "status": "success",
        "brand": brand_name,
        "manufacturer": manufacturer,
        "therapeutic_class": therapeutic_class,
        "sales_value": float(sales),
        "sales_units": float(units),
        "average_price": float(average_price)
    }


# ============================================================
# MOLECULE / ACTIVE INGREDIENT TOOL
# ============================================================

def molecule_tool(
    molecule: str
) -> Dict[str, Any]:

    df = load_data()

    # NOTE:
    # The current dataset contains "Drug Strength".
    # If the active ingredient is embedded in another field
    # in your actual IQVIA file, adapt this search accordingly.

    data = df[
        df["Drug Strength"]
        .astype(str)
        .str.contains(
            molecule,
            case=False,
            na=False
        )
    ]

    if data.empty:

        return {
            "status": "not_found",
            "molecule": molecule
        }

    total_sales = data["Sales Value"].sum()
    total_units = data["Sales Units"].sum()

    brands = (
        data.groupby("Brand Name")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    manufacturers = (
        data.groupby("Manufacturer")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "status": "success",
        "molecule": molecule,
        "sales_value": float(total_sales),
        "sales_units": float(total_units),
        "brand_count": int(data["Brand Name"].nunique()),
        "manufacturer_count": int(
            data["Manufacturer"].nunique()
        ),
        "top_brands": brands.to_dict(),
        "top_manufacturers": manufacturers.to_dict()
    }


# ============================================================
# COMPANY TOOL
# ============================================================

def company_tool(
    company_name: str
) -> Dict[str, Any]:

    df = load_data()

    data = df[
        df["Manufacturer"]
        .astype(str)
        .str.contains(
            company_name,
            case=False,
            na=False
        )
    ]

    if data.empty:

        return {
            "status": "not_found",
            "company": company_name
        }

    sales = data["Sales Value"].sum()
    units = data["Sales Units"].sum()

    brands = (
        data.groupby("Brand Name")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    therapeutic_classes = (
        data.groupby("Therapeutic Class")["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "status": "success",
        "company": company_name,
        "sales_value": float(sales),
        "sales_units": float(units),
        "brand_count": int(
            data["Brand Name"].nunique()
        ),
        "top_brands": brands.to_dict(),
        "top_therapeutic_classes":
            therapeutic_classes.to_dict()
    }


# ============================================================
# FORECAST TOOL
# ============================================================

def forecast_tool(
    brand_name: Optional[str] = None,
    manufacturer: Optional[str] = None
) -> Dict[str, Any]:

    df = load_data()

    if brand_name:

        data = df[
            df["Brand Name"]
            .astype(str)
            .str.contains(
                brand_name,
                case=False,
                na=False
            )
        ]

    elif manufacturer:

        data = df[
            df["Manufacturer"]
            .astype(str)
            .str.contains(
                manufacturer,
                case=False,
                na=False
            )
        ]

    else:

        data = df

    if data.empty:

        return {
            "status": "no_data"
        }

    yearly = (
        data.groupby("Year")["Sales Value"]
        .sum()
        .sort_index()
    )

    if len(yearly) >= 2:

        latest = yearly.iloc[-1]
        previous = yearly.iloc[-2]

        growth = (
            (latest - previous)
            / previous
            * 100
            if previous != 0
            else np.nan
        )

    else:

        latest = yearly.iloc[-1]
        growth = np.nan

    return {
        "status": "success",
        "historical_sales": yearly.to_dict(),
        "latest_sales": float(latest),
        "latest_growth_percent": (
            float(growth)
            if not np.isnan(growth)
            else None
        )
    }


# ============================================================
# LAUNCH TOOL
# ============================================================

def launch_tool(
    brand_name: Optional[str] = None
) -> Dict[str, Any]:

    df = load_data()

    data = df.copy()

    data["Product Launch"] = pd.to_datetime(
        data["Product Launch"],
        errors="coerce"
    )

    if brand_name:

        data = data[
            data["Brand Name"]
            .astype(str)
            .str.contains(
                brand_name,
                case=False,
                na=False
            )
        ]

    launch_count = data["Product Launch"].notna().sum()

    return {
        "status": "success",
        "products_with_launch_date":
            int(launch_count),
        "unique_brands":
            int(data["Brand Name"].nunique()),
        "launch_year_distribution":
            data["Product Launch"]
            .dt.year
            .value_counts()
            .sort_index()
            .to_dict()
    }


# ============================================================
# GTM TOOL
# ============================================================

def gtm_tool(
    brand_name: Optional[str] = None,
    manufacturer: Optional[str] = None
) -> Dict[str, Any]:

    df = load_data()

    data = df.copy()

    if brand_name:

        data = data[
            data["Brand Name"]
            .astype(str)
            .str.contains(
                brand_name,
                case=False,
                na=False
            )
        ]

    if manufacturer:

        data = data[
            data["Manufacturer"]
            .astype(str)
            .str.contains(
                manufacturer,
                case=False,
                na=False
            )
        ]

    if data.empty:

        return {
            "status": "no_data"
        }

    channel_sales = (
        data.groupby("Distribution Channel")
        ["Sales Value"]
        .sum()
        .sort_values(ascending=False)
    )

    category_sales = (
        data.groupby("Market Category")
        ["Sales Value"]
        .sum()
        .sort_values(ascending=False)
    )

    therapeutic_sales = (
        data.groupby("Therapeutic Class")
        ["Sales Value"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "status": "success",
        "channel_sales":
            channel_sales.to_dict(),
        "market_category_sales":
            category_sales.to_dict(),
        "therapeutic_class_sales":
            therapeutic_sales.to_dict()
    }


# ============================================================
# RECOMMENDATION TOOL
# ============================================================

def recommendation_tool(
    therapeutic_class: Optional[str] = None
) -> Dict[str, Any]:

    df = load_data()

    if therapeutic_class:

        data = df[
            df["Therapeutic Class"]
            .astype(str)
            .str.contains(
                therapeutic_class,
                case=False,
                na=False
            )
        ]

    else:

        data = df

    if data.empty:

        return {
            "status": "no_data"
        }

    brand_performance = (
        data.groupby("Brand Name")
        .agg(
            Sales_Value=("Sales Value", "sum"),
            Sales_Units=("Sales Units", "sum")
        )
        .sort_values(
            "Sales_Value",
            ascending=False
        )
    )

    top_opportunities = (
        brand_performance
        .head(10)
        .reset_index()
        .to_dict("records")
    )

    return {
        "status": "success",
        "top_opportunities": top_opportunities
    }


# ============================================================
# SIMILARITY TOOL
# ============================================================

def similarity_tool(
    brand_name: str
) -> Dict[str, Any]:

    df = load_data()

    data = df[
        df["Brand Name"]
        .astype(str)
        .str.contains(
            brand_name,
            case=False,
            na=False
        )
    ]

    if data.empty:

        return {
            "status": "not_found",
            "brand": brand_name
        }

    therapeutic_class = (
        data["Therapeutic Class"]
        .mode()
        .iloc[0]
        if not data["Therapeutic Class"].mode().empty
        else None
    )

    drug_strength = (
        data["Drug Strength"]
        .mode()
        .iloc[0]
        if not data["Drug Strength"].mode().empty
        else None
    )

    candidates = df.copy()

    if therapeutic_class:

        candidates = candidates[
            candidates["Therapeutic Class"]
            .astype(str)
            .str.contains(
                str(therapeutic_class),
                case=False,
                na=False
            )
        ]

    candidates = candidates[
        candidates["Brand Name"].astype(str)
        .str.lower()
        != brand_name.lower()
    ]

    similar = (
        candidates.groupby("Brand Name")
        ["Sales Value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "status": "success",
        "brand": brand_name,
        "therapeutic_class": therapeutic_class,
        "drug_strength": drug_strength,
        "similar_brands": similar.to_dict()
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {

    "market_tool": market_tool,

    "brand_tool": brand_tool,

    "molecule_tool": molecule_tool,

    "company_tool": company_tool,

    "forecast_tool": forecast_tool,

    "launch_tool": launch_tool,

    "gtm_tool": gtm_tool,

    "recommendation_tool":
        recommendation_tool,

    "similarity_tool":
        similarity_tool
}


def get_available_tools():

    return list(TOOL_REGISTRY.keys())


def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any]
):

    if tool_name not in TOOL_REGISTRY:

        return {
            "status": "error",
            "message":
                f"Unknown tool: {tool_name}"
        }

    try:

        return TOOL_REGISTRY[
            tool_name
        ](**parameters)

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }