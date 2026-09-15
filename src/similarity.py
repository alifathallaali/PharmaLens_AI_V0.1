# ============================================================
# PHARMALENS AI
# PROJECT 08 — DRUG SIMILARITY & DRUG INTELLIGENCE
# src/similarity.py
#
# FINAL MEMORY-SAFE VERSION
#
# No full N x N similarity matrices.
# Uses Top-K nearest-neighbor / candidate architecture.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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
    "Sales Value",
]


# ============================================================
# 2. VALIDATION
# ============================================================

def validate_data(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return True


# ============================================================
# 3. CREATE PRODUCT-LEVEL DATASET
# ============================================================

def create_product_dataset(df):

    validate_data(df)

    data = df.copy()

    text_columns = [
        "Distribution Channel",
        "Therapeutic Class",
        "Manufacturer",
        "Brand Name",
        "Pack Size",
        "Drug Strength",
        "Market Category",
    ]

    for col in text_columns:

        data[col] = (
            data[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    numeric_columns = [
        "Selling Price",
        "Sales Units",
        "Sales Value",
        "Year",
    ]

    for col in numeric_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data["Product Launch"] = pd.to_datetime(
        data["Product Launch"],
        errors="coerce"
    )

    data["Month"] = pd.to_datetime(
        data["Month"],
        errors="coerce"
    )

    product_keys = [
        "Brand Name",
        "Therapeutic Class",
        "Manufacturer",
        "Pack Size",
        "Drug Strength",
        "Market Category",
    ]

    product_data = (
        data
        .groupby(
            product_keys,
            dropna=False
        )
        .agg(

            Average_Price=(
                "Selling Price",
                "mean"
            ),

            Median_Price=(
                "Selling Price",
                "median"
            ),

            Min_Price=(
                "Selling Price",
                "min"
            ),

            Max_Price=(
                "Selling Price",
                "max"
            ),

            Price_Std=(
                "Selling Price",
                "std"
            ),

            Total_Sales_Units=(
                "Sales Units",
                "sum"
            ),

            Total_Sales_Value=(
                "Sales Value",
                "sum"
            ),

            Average_Monthly_Units=(
                "Sales Units",
                "mean"
            ),

            Average_Monthly_Value=(
                "Sales Value",
                "mean"
            ),

            First_Launch_Date=(
                "Product Launch",
                "min"
            ),

            Last_Launch_Date=(
                "Product Launch",
                "max"
            ),

            Last_Year=(
                "Year",
                "max"
            ),

            Number_of_Months=(
                "Month",
                "nunique"
            ),

            Distribution_Channels=(
                "Distribution Channel",
                "nunique"
            ),
        )
        .reset_index()
    )

    product_data.insert(
        0,
        "Product_ID",
        range(
            1,
            len(product_data) + 1
        )
    )

    # --------------------------------------------------------
    # Product age
    # --------------------------------------------------------

    analysis_date = pd.Timestamp(
        "2025-12-31"
    )

    product_data[
        "Product_Age_Years"
    ] = (
        (
            analysis_date
            -
            product_data[
                "First_Launch_Date"
            ]
        ).dt.days
        / 365.25
    )

    median_age = (
        product_data[
            "Product_Age_Years"
        ].median()
    )

    product_data[
        "Product_Age_Years"
    ] = (
        product_data[
            "Product_Age_Years"
        ]
        .fillna(
            median_age
        )
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Price CV
    # --------------------------------------------------------

    product_data[
        "Price_CV"
    ] = (
        product_data[
            "Price_Std"
        ]
        /
        product_data[
            "Average_Price"
        ].replace(
            0,
            np.nan
        )
    )

    product_data[
        "Price_CV"
    ] = (
        product_data[
            "Price_CV"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Growth indicators
    # --------------------------------------------------------

    product_data[
        "Sales_Value_Per_Unit"
    ] = (
        product_data[
            "Total_Sales_Value"
        ]
        /
        product_data[
            "Total_Sales_Units"
        ].replace(
            0,
            np.nan
        )
    )

    product_data[
        "Sales_Value_Per_Unit"
    ] = (
        product_data[
            "Sales_Value_Per_Unit"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(
            product_data[
                "Average_Price"
            ]
        )
    )

    return product_data


# ============================================================
# 4. TEXT SIMILARITY
# ============================================================

def build_text_model(
    product_data,
    top_k=20
):

    profiles = (

        product_data[
            "Brand Name"
        ].fillna("Unknown").astype(str)

        + " "

        + product_data[
            "Therapeutic Class"
        ].fillna("Unknown").astype(str)

        + " "

        + product_data[
            "Manufacturer"
        ].fillna("Unknown").astype(str)

        + " "

        + product_data[
            "Pack Size"
        ].fillna("Unknown").astype(str)

        + " "

        + product_data[
            "Drug Strength"
        ].fillna("Unknown").astype(str)

        + " "

        + product_data[
            "Market Category"
        ].fillna("Unknown").astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(
        profiles
    )

    nn = NearestNeighbors(
        n_neighbors=min(
            top_k + 1,
            len(product_data)
        ),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(matrix)

    distances, indices = (
        nn.kneighbors(matrix)
    )

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "indices": indices,
        "distances": distances,
    }


# ============================================================
# 5. CATEGORICAL SIMILARITY
# ============================================================

def build_categorical_similarity(
    product_data,
    top_k=20
):

    columns = [
        "Therapeutic Class",
        "Manufacturer",
        "Market Category",
        "Drug Strength",
        "Pack Size",
    ]

    categorical_data = (
        product_data[
            columns
        ]
        .fillna("Unknown")
        .astype(str)
        .apply(
            lambda x:
            x.str.strip().str.lower()
        )
    )

    # --------------------------------------------------------
    # Build inverted indexes
    # --------------------------------------------------------

    inverted_indexes = {}

    for col in columns:

        groups = (
            categorical_data
            .groupby(col)
            .groups
        )

        inverted_indexes[col] = {
            value:
            np.asarray(
                indices,
                dtype=np.int32
            )

            for value, indices
            in groups.items()
        }

    results = {}

    for i in range(
        len(product_data)
    ):

        candidate_counts = {}

        for col in columns:

            value = (
                categorical_data.iloc[i][col]
            )

            candidates = (
                inverted_indexes[col]
                .get(
                    value,
                    []
                )
            )

            for j in candidates:

                if j == i:
                    continue

                candidate_counts[j] = (
                    candidate_counts.get(
                        j,
                        0
                    )
                    + 1
                )

        scores = [

            (
                int(j),
                shared
                /
                len(columns)
            )

            for j, shared
            in candidate_counts.items()
        ]

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results[i] = (
            scores[:top_k]
        )

    return results


# ============================================================
# 6. NUMERICAL SIMILARITY
# ============================================================

def build_numerical_model(
    product_data,
    top_k=20
):

    columns = [
        "Average_Price",
        "Total_Sales_Units",
        "Total_Sales_Value",
        "Product_Age_Years",
        "Distribution_Channels",
    ]

    features = (
        product_data[
            columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .copy()
    )

    features = features.fillna(
        features.median()
    )

    for col in [
        "Average_Price",
        "Total_Sales_Units",
        "Total_Sales_Value",
    ]:

        features[col] = np.log1p(
            features[col].clip(
                lower=0
            )
        )

    scaler = StandardScaler()

    scaled = (
        scaler
        .fit_transform(features)
        .astype(np.float32)
    )

    nn = NearestNeighbors(
        n_neighbors=min(
            top_k + 1,
            len(product_data)
        ),
        metric="euclidean",
        n_jobs=-1,
    )

    nn.fit(scaled)

    distances, indices = (
        nn.kneighbors(scaled)
    )

    results = {}

    for i in range(
        len(product_data)
    ):

        matches = []

        for position in range(
            1,
            len(indices[i])
        ):

            j = int(
                indices[i][position]
            )

            distance = float(
                distances[i][position]
            )

            similarity = (
                1.0
                /
                (
                    1.0
                    +
                    distance
                )
            )

            matches.append(
                (
                    j,
                    similarity
                )
            )

        results[i] = matches

    return {
        "similarity": results,
        "scaler": scaler,
    }


# ============================================================
# 7. PRICE SIMILARITY
# ============================================================

def calculate_price_similarity(
    price_a,
    price_b
):

    if (
        pd.isna(price_a)
        or pd.isna(price_b)
    ):
        return 0.0

    price_a = float(price_a)
    price_b = float(price_b)

    if price_a < 0 or price_b < 0:
        return 0.0

    denominator = max(
        price_a,
        price_b,
        1e-9
    )

    score = (
        1
        -
        abs(
            price_a
            -
            price_b
        )
        /
        denominator
    )

    return float(
        np.clip(
            score,
            0,
            1
        )
    )


# ============================================================
# 8. PRICE TOP-K MODEL
# ============================================================

def build_price_similarity(
    product_data,
    top_k=20
):

    prices = (
        product_data[
            ["Average_Price"]
        ]
        .fillna(0)
        .values
        .astype(np.float32)
    )

    price_features = np.log1p(
        np.clip(
            prices,
            0,
            None
        )
    )

    nn = NearestNeighbors(
        n_neighbors=min(
            top_k + 1,
            len(product_data)
        ),
        metric="euclidean",
        n_jobs=-1,
    )

    nn.fit(
        price_features
    )

    distances, indices = (
        nn.kneighbors(
            price_features
        )
    )

    results = {}

    for i in range(
        len(product_data)
    ):

        matches = []

        price_a = (
            product_data.iloc[i][
                "Average_Price"
            ]
        )

        for position in range(
            1,
            len(indices[i])
        ):

            j = int(
                indices[i][position]
            )

            price_b = (
                product_data.iloc[j][
                    "Average_Price"
                ]
            )

            score = (
                calculate_price_similarity(
                    price_a,
                    price_b
                )
            )

            matches.append(
                (
                    j,
                    score
                )
            )

        matches.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results[i] = (
            matches[:top_k]
        )

    return results


# ============================================================
# 9. COMBINED SIMILARITY
# ============================================================

def combine_similarity(
    text_model,
    categorical_similarity,
    numerical_similarity,
    price_similarity,
    top_k=20,
    text_weight=0.30,
    categorical_weight=0.25,
    numerical_weight=0.15,
    price_weight=0.30,
):

    indices = text_model[
        "indices"
    ]

    distances = text_model[
        "distances"
    ]

    results = {}

    for i in range(
        len(indices)
    ):

        candidates = set()

        # Text
        for position in range(
            1,
            len(indices[i])
        ):

            candidates.add(
                int(
                    indices[i][position]
                )
            )

        # Categorical
        candidates.update(
            j
            for j, score
            in categorical_similarity.get(
                i,
                []
            )
        )

        # Numerical
        candidates.update(
            j
            for j, score
            in numerical_similarity.get(
                i,
                []
            )
        )

        # Price
        candidates.update(
            j
            for j, score
            in price_similarity.get(
                i,
                []
            )
        )

        text_scores = {}

        for position in range(
            1,
            len(indices[i])
        ):

            j = int(
                indices[i][position]
            )

            text_scores[j] = max(
                0.0,
                1.0
                -
                float(
                    distances[i][position]
                )
            )

        categorical_scores = dict(
            categorical_similarity.get(
                i,
                []
            )
        )

        numerical_scores = dict(
            numerical_similarity.get(
                i,
                []
            )
        )

        price_scores = dict(
            price_similarity.get(
                i,
                []
            )
        )

        scored = []

        for j in candidates:

            text_score = (
                text_scores.get(
                    j,
                    0.0
                )
            )

            categorical_score = (
                categorical_scores.get(
                    j,
                    0.0
                )
            )

            numerical_score = (
                numerical_scores.get(
                    j,
                    0.0
                )
            )

            price_score = (
                price_scores.get(
                    j,
                    0.0
                )
            )

            final_score = (

                text_weight
                * text_score

                +

                categorical_weight
                * categorical_score

                +

                numerical_weight
                * numerical_score

                +

                price_weight
                * price_score
            )

            scored.append(
                (
                    int(j),
                    final_score,
                    text_score,
                    categorical_score,
                    numerical_score,
                    price_score,
                )
            )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results[i] = (
            scored[:top_k]
        )

    return results


# ============================================================
# 10. BUILD COMPLETE ENGINE
# ============================================================

def build_similarity_engine(
    df,
    top_k=20,
    text_weight=0.30,
    categorical_weight=0.25,
    numerical_weight=0.15,
    price_weight=0.30,
):

    product_data = (
        create_product_dataset(df)
    )

    text_model = (
        build_text_model(
            product_data,
            top_k
        )
    )

    categorical_similarity = (
        build_categorical_similarity(
            product_data,
            top_k
        )
    )

    numerical_model = (
        build_numerical_model(
            product_data,
            top_k
        )
    )

    price_similarity = (
        build_price_similarity(
            product_data,
            top_k
        )
    )

    combined_similarity = (
        combine_similarity(
            text_model,
            categorical_similarity,
            numerical_model[
                "similarity"
            ],
            price_similarity,
            top_k,
            text_weight,
            categorical_weight,
            numerical_weight,
            price_weight,
        )
    )

    return {

        "product_data":
            product_data,

        "text_model":
            text_model,

        "categorical_similarity":
            categorical_similarity,

        "numerical_model":
            numerical_model,

        "price_similarity":
            price_similarity,

        "combined_similarity":
            combined_similarity,
    }


# ============================================================
# 11. FIND PRODUCT
# ============================================================

def find_product_index(
    product_data,
    brand_name
):

    matches = product_data[
        product_data[
            "Brand Name"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        str(brand_name)
        .lower()
        .strip()
    ]

    if matches.empty:
        return None

    return matches.index[0]


# ============================================================
# 12. SIMILAR DRUGS
# ============================================================

def get_similar_drugs(
    engine,
    brand_name,
    top_n=10
):

    product_data = engine[
        "product_data"
    ]

    similarity = engine[
        "combined_similarity"
    ]

    idx = find_product_index(
        product_data,
        brand_name
    )

    if idx is None:
        return pd.DataFrame()

    records = []

    for rank, match in enumerate(
        similarity.get(
            idx,
            []
        ),
        start=1
    ):

        (
            j,
            final_score,
            text_score,
            categorical_score,
            numerical_score,
            price_score,
        ) = match

        source = product_data.iloc[idx]
        target = product_data.iloc[j]

        original_price = (
            source[
                "Average_Price"
            ]
        )

        similar_price = (
            target[
                "Average_Price"
            ]
        )

        if (
            pd.isna(original_price)
            or pd.isna(similar_price)
            or original_price == 0
        ):

            price_difference = np.nan

        else:

            price_difference = (
                (
                    similar_price
                    -
                    original_price
                )
                /
                original_price
                * 100
            )

        records.append({

            "Product_ID":
                int(
                    source[
                        "Product_ID"
                    ]
                ),

            "Brand_Name":
                source[
                    "Brand Name"
                ],

            "Manufacturer":
                source[
                    "Manufacturer"
                ],

            "Therapeutic_Class":
                source[
                    "Therapeutic Class"
                ],

            "Similar_Product_ID":
                int(
                    target[
                        "Product_ID"
                    ]
                ),

            "Similar_Brand_Name":
                target[
                    "Brand Name"
                ],

            "Similar_Manufacturer":
                target[
                    "Manufacturer"
                ],

            "Similar_Therapeutic_Class":
                target[
                    "Therapeutic Class"
                ],

            "Original_Price":
                original_price,

            "Similar_Price":
                similar_price,

            "Price_Difference_%":
                price_difference,

            "Similarity_Rank":
                rank,

            "Text_Similarity":
                round(
                    text_score,
                    4
                ),

            "Categorical_Similarity":
                round(
                    categorical_score,
                    4
                ),

            "Numerical_Similarity":
                round(
                    numerical_score,
                    4
                ),

            "Price_Similarity":
                round(
                    price_score,
                    4
                ),

            "Final_Similarity":
                round(
                    final_score,
                    4
                ),

            "Similarity_%":
                round(
                    final_score * 100,
                    2
                ),
        })

        if rank >= top_n:
            break

    return pd.DataFrame(
        records
    )


# ============================================================
# 13. COMPETITORS
# ============================================================

def get_competitors(
    engine,
    brand_name,
    top_n=10
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=top_n * 3
    )

    if results.empty:
        return results

    manufacturer = (
        results.iloc[0][
            "Manufacturer"
        ]
    )

    return (
        results[
            results[
                "Similar_Manufacturer"
            ]
            != manufacturer
        ]
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 14. THERAPEUTIC ALTERNATIVES
# ============================================================

def get_therapeutic_alternatives(
    engine,
    brand_name,
    top_n=10
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=top_n * 3
    )

    if results.empty:
        return results

    therapeutic_class = (
        results.iloc[0][
            "Therapeutic_Class"
        ]
    )

    manufacturer = (
        results.iloc[0][
            "Manufacturer"
        ]
    )

    return (
        results[
            (
                results[
                    "Therapeutic_Class"
                ]
                ==
                therapeutic_class
            )
            &
            (
                results[
                    "Similar_Manufacturer"
                ]
                != manufacturer
            )
        ]
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 15. PRICE ALTERNATIVES
# ============================================================

def get_price_alternatives(
    engine,
    brand_name,
    top_n=10,
    max_price_difference_pct=30
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=top_n * 5
    )

    if results.empty:
        return results

    results = results[
        results[
            "Price_Difference_%"
        ].abs()
        <= max_price_difference_pct
    ]

    return (
        results
        .sort_values(
            [
                "Price_Similarity",
                "Final_Similarity"
            ],
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 16. LOWER-PRICE ALTERNATIVES
# ============================================================

def get_lower_price_alternatives(
    engine,
    brand_name,
    top_n=10
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=top_n * 5
    )

    if results.empty:
        return results

    results = results[
        results[
            "Similar_Price"
        ]
        <
        results[
            "Original_Price"
        ]
    ]

    return (
        results
        .sort_values(
            [
                "Final_Similarity",
                "Price_Similarity"
            ],
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 17. PREMIUM ALTERNATIVES
# ============================================================

def get_premium_alternatives(
    engine,
    brand_name,
    top_n=10
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=top_n * 5
    )

    if results.empty:
        return results

    results = results[
        results[
            "Similar_Price"
        ]
        >
        results[
            "Original_Price"
        ]
    ]

    return (
        results
        .sort_values(
            "Final_Similarity",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 18. COMPETITIVE PRESSURE
# ============================================================

def calculate_competitive_pressure(
    engine
):

    product_data = engine[
        "product_data"
    ]

    similarity = engine[
        "combined_similarity"
    ]

    records = []

    for i, matches in similarity.items():

        source = product_data.iloc[i]

        competitor_scores = []

        for match in matches:

            j = match[0]
            score = match[1]

            target = product_data.iloc[j]

            if (
                target[
                    "Manufacturer"
                ]
                !=
                source[
                    "Manufacturer"
                ]
            ):

                competitor_scores.append(
                    score
                )

        if competitor_scores:

            competitor_scores.sort(
                reverse=True
            )

            top_scores = (
                competitor_scores[:10]
            )

            avg_score = np.mean(
                top_scores
            )

            max_score = max(
                top_scores
            )

            pressure = (
                0.60 * avg_score
                +
                0.40 * max_score
            )

            records.append({

                "Product_ID":
                    source[
                        "Product_ID"
                    ],

                "Brand_Name":
                    source[
                        "Brand Name"
                    ],

                "Manufacturer":
                    source[
                        "Manufacturer"
                    ],

                "Number_of_Competitors":
                    len(
                        competitor_scores
                    ),

                "Average_Competitor_Similarity":
                    avg_score,

                "Maximum_Competitor_Similarity":
                    max_score,

                "Competitive_Pressure_Score":
                    pressure,
            })

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    result[
        "Competitive_Pressure_%"
    ] = (
        result[
            "Competitive_Pressure_Score"
        ]
        * 100
    )

    result[
        "Competitive_Pressure_Level"
    ] = pd.cut(
        result[
            "Competitive_Pressure_Score"
        ],
        bins=[
            -np.inf,
            0.30,
            0.55,
            0.75,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ]
    )

    return (
        result
        .sort_values(
            "Competitive_Pressure_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 19. MARKET OPPORTUNITY
# ============================================================

def calculate_market_opportunity(
    engine
):

    product_data = engine[
        "product_data"
    ].copy()

    pressure = (
        calculate_competitive_pressure(
            engine
        )
    )

    if not pressure.empty:

        product_data = product_data.merge(
            pressure[
                [
                    "Product_ID",
                    "Competitive_Pressure_Score",
                ]
            ],
            on="Product_ID",
            how="left"
        )

    else:

        product_data[
            "Competitive_Pressure_Score"
        ] = 0.0

    product_data[
        "Competitive_Pressure_Score"
    ] = (
        product_data[
            "Competitive_Pressure_Score"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Market opportunity components
    # --------------------------------------------------------

    sales_value = np.log1p(
        product_data[
            "Total_Sales_Value"
        ].clip(
            lower=0
        )
    )

    sales_units = np.log1p(
        product_data[
            "Total_Sales_Units"
        ].clip(
            lower=0
        )
    )

    opportunity_components = pd.DataFrame({

        "Sales_Value":
            sales_value,

        "Sales_Units":
            sales_units,

        "Competitive_Pressure":
            product_data[
                "Competitive_Pressure_Score"
            ],

        "Distribution":
            product_data[
                "Distribution_Channels"
            ],
    })

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    for col in opportunity_components.columns:

        min_val = (
            opportunity_components[col]
            .min()
        )

        max_val = (
            opportunity_components[col]
            .max()
        )

        if max_val > min_val:

            opportunity_components[
                col
            ] = (
                opportunity_components[
                    col
                ]
                - min_val
            ) / (
                max_val
                - min_val
            )

        else:

            opportunity_components[
                col
            ] = 0.0

    # --------------------------------------------------------
    # Opportunity score
    #
    # High sales + broad distribution
    # + manageable competition
    # --------------------------------------------------------

    product_data[
        "Market_Opportunity_Score"
    ] = (

        0.40
        * opportunity_components[
            "Sales_Value"
        ]

        +

        0.20
        * opportunity_components[
            "Sales_Units"
        ]

        +

        0.20
        * opportunity_components[
            "Distribution"
        ]

        +

        0.20
        * (
            1
            -
            opportunity_components[
                "Competitive_Pressure"
            ]
        )
    )

    product_data[
        "Market_Opportunity_%"
    ] = (
        product_data[
            "Market_Opportunity_Score"
        ]
        * 100
    )

    product_data[
        "Market_Opportunity_Tier"
    ] = pd.cut(

        product_data[
            "Market_Opportunity_Score"
        ],

        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            np.inf,
        ],

        labels=[
            "Low Priority",
            "Monitor",
            "Strategic Opportunity",
            "High Opportunity",
        ]
    )

    product_data[
        "Market_Opportunity_Action"
    ] = np.select(

        [
            product_data[
                "Market_Opportunity_Tier"
            ].astype(str)
            ==
            "High Opportunity",

            product_data[
                "Market_Opportunity_Tier"
            ].astype(str)
            ==
            "Strategic Opportunity",

            product_data[
                "Market_Opportunity_Tier"
            ].astype(str)
            ==
            "Monitor",
        ],

        [
            "Invest & Expand",
            "Selective Investment",
            "Monitor & Validate",
        ],

        default="Deprioritize"
    )

    return (
        product_data
        .sort_values(
            "Market_Opportunity_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 20. STRATEGIC NEXT ACTION
# ============================================================

def get_next_strategic_action(
    engine,
    brand_name
):

    product_data = engine[
        "product_data"
    ]

    idx = find_product_index(
        product_data,
        brand_name
    )

    if idx is None:

        return {
            "Brand": brand_name,
            "Action": "Brand not found",
            "Reason": "",
        }

    opportunity = (
        calculate_market_opportunity(
            engine
        )
    )

    row = opportunity[
        opportunity[
            "Product_ID"
        ]
        ==
        product_data.iloc[idx][
            "Product_ID"
        ]
    ]

    if row.empty:

        return {
            "Brand": brand_name,
            "Action": "Insufficient data",
            "Reason": "",
        }

    row = row.iloc[0]

    score = float(
        row[
            "Market_Opportunity_Score"
        ]
    )

    pressure = float(
        row[
            "Competitive_Pressure_Score"
        ]
    )

    # --------------------------------------------------------
    # Strategic decision rules
    # --------------------------------------------------------

    if (
        score >= 0.75
        and pressure < 0.55
    ):

        action = "Invest & Expand"

        reason = (
            "High market opportunity "
            "with manageable competitive pressure."
        )

    elif (
        score >= 0.60
        and pressure >= 0.55
    ):

        action = "Defend & Differentiate"

        reason = (
            "Strong opportunity exists, "
            "but competitive pressure is significant."
        )

    elif (
        score >= 0.50
        and pressure < 0.55
    ):

        action = "Selective Investment"

        reason = (
            "Promising market position "
            "with manageable competition."
        )

    elif pressure >= 0.75:

        action = "Defend / Reposition"

        reason = (
            "Very high competitive pressure "
            "requires differentiation or repositioning."
        )

    elif score < 0.25:

        action = "Deprioritize"

        reason = (
            "Low market opportunity "
            "relative to the portfolio."
        )

    else:

        action = "Monitor & Validate"

        reason = (
            "Market opportunity is moderate; "
            "additional evidence should be monitored."
        )

    return {

        "Brand":
            brand_name,

        "Market_Opportunity_Score":
            score,

        "Competitive_Pressure_Score":
            pressure,

        "Market_Opportunity_Tier":
            str(
                row[
                    "Market_Opportunity_Tier"
                ]
            ),

        "Action":
            action,

        "Reason":
            reason,
    }


# ============================================================
# 21. SIMILARITY PROFILE
# ============================================================

def get_similarity_profile(
    engine,
    brand_name
):

    results = get_similar_drugs(
        engine,
        brand_name,
        top_n=1
    )

    if results.empty:
        return {}

    row = results.iloc[0]

    return {

        "Brand":
            brand_name,

        "Average_Price":
            row[
                "Original_Price"
            ],

        "Closest_Product":
            row[
                "Similar_Brand_Name"
            ],

        "Overall_Similarity":
            row[
                "Final_Similarity"
            ],

        "Text_Similarity":
            row[
                "Text_Similarity"
            ],

        "Categorical_Similarity":
            row[
                "Categorical_Similarity"
            ],

        "Numerical_Similarity":
            row[
                "Numerical_Similarity"
            ],

        "Price_Similarity":
            row[
                "Price_Similarity"
            ],

        "Price_Difference_%":
            row[
                "Price_Difference_%"
            ],
    }


# ============================================================
# 22. BUSINESS INSIGHTS
# ============================================================

def generate_similarity_insights(
    engine,
    brand_name,
    top_n=5
):

    similar = get_similar_drugs(
        engine,
        brand_name,
        top_n
    )

    if similar.empty:

        return {
            "Brand": brand_name,
            "Status": "Brand not found",
            "Insights": [],
        }

    insights = []

    top_match = similar.iloc[0]

    insights.append(
        f"Closest similar product: "
        f"{top_match['Similar_Brand_Name']} "
        f"with "
        f"{top_match['Similarity_%']:.1f}% "
        f"overall similarity."
    )

    price_difference = (
        top_match[
            "Price_Difference_%"
        ]
    )

    if pd.notna(price_difference):

        if price_difference < 0:

            insights.append(
                f"The closest product is "
                f"{abs(price_difference):.1f}% "
                f"cheaper."
            )

        elif price_difference > 0:

            insights.append(
                f"The closest product is "
                f"{price_difference:.1f}% "
                f"more expensive."
            )

        else:

            insights.append(
                "The closest product has "
                "a similar price."
            )

    competitors = get_competitors(
        engine,
        brand_name,
        top_n
    )

    if not competitors.empty:

        strongest = (
            competitors.iloc[0]
        )

        insights.append(
            f"Strongest identified competitor: "
            f"{strongest['Similar_Brand_Name']} "
            f"from "
            f"{strongest['Similar_Manufacturer']}."
        )

    lower_price = (
        get_lower_price_alternatives(
            engine,
            brand_name,
            top_n
        )
    )

    if not lower_price.empty:

        insights.append(
            f"{len(lower_price)} "
            f"lower-priced similar products "
            f"were identified."
        )

    action = get_next_strategic_action(
        engine,
        brand_name
    )

    if action:

        insights.append(
            f"Recommended action: "
            f"{action['Action']}."
        )

    return {

        "Brand":
            brand_name,

        "Status":
            "Success",

        "Insights":
            insights,
    }


# ============================================================
# 23. FINAL DRUG INTELLIGENCE
# ============================================================

def final_drug_intelligence(
    engine,
    brand_name,
    top_n=10
):

    product_data = engine[
        "product_data"
    ]

    idx = find_product_index(
        product_data,
        brand_name
    )

    if idx is None:

        return {
            "status": "Brand not found",
            "brand": brand_name,
        }

    source = product_data.iloc[idx]

    similar = get_similar_drugs(
        engine,
        brand_name,
        top_n
    )

    competitors = get_competitors(
        engine,
        brand_name,
        top_n
    )

    therapeutic = (
        get_therapeutic_alternatives(
            engine,
            brand_name,
            top_n
        )
    )

    price_alternatives = (
        get_price_alternatives(
            engine,
            brand_name,
            top_n
        )
    )

    lower_price = (
        get_lower_price_alternatives(
            engine,
            brand_name,
            top_n
        )
    )

    premium = (
        get_premium_alternatives(
            engine,
            brand_name,
            top_n
        )
    )

    profile = get_similarity_profile(
        engine,
        brand_name
    )

    action = get_next_strategic_action(
        engine,
        brand_name
    )

    insights = generate_similarity_insights(
        engine,
        brand_name,
        top_n
    )

    return {

        "status":
            "Success",

        "brand":
            brand_name,

        "product_profile": {
            "Product_ID":
                int(
                    source[
                        "Product_ID"
                    ]
                ),

            "Manufacturer":
                source[
                    "Manufacturer"
                ],

            "Therapeutic_Class":
                source[
                    "Therapeutic Class"
                ],

            "Market_Category":
                source[
                    "Market Category"
                ],

            "Drug_Strength":
                source[
                    "Drug Strength"
                ],

            "Pack_Size":
                source[
                    "Pack Size"
                ],

            "Average_Price":
                source[
                    "Average_Price"
                ],

            "Total_Sales_Units":
                source[
                    "Total_Sales_Units"
                ],

            "Total_Sales_Value":
                source[
                    "Total_Sales_Value"
                ],
        },

        "similar_products":
            similar,

        "competitors":
            competitors,

        "therapeutic_alternatives":
            therapeutic,

        "price_alternatives":
            price_alternatives,

        "lower_price_alternatives":
            lower_price,

        "premium_alternatives":
            premium,

        "similarity_profile":
            profile,

        "strategic_action":
            action,

        "business_insights":
            insights,
    }


# ============================================================
# 24. AGENT-READY FUNCTIONS
# ============================================================

def drug_similarity_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n
        )
    )

    return get_similar_drugs(
        engine,
        brand_name,
        top_n
    )


def drug_competitor_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n * 3
        )
    )

    return get_competitors(
        engine,
        brand_name,
        top_n
    )


def therapeutic_alternative_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n * 3
        )
    )

    return get_therapeutic_alternatives(
        engine,
        brand_name,
        top_n
    )


def price_alternative_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n * 5
        )
    )

    return get_price_alternatives(
        engine,
        brand_name,
        top_n
    )


def lower_price_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n * 5
        )
    )

    return get_lower_price_alternatives(
        engine,
        brand_name,
        top_n
    )


def market_opportunity_tool(
    df,
    brand_name
):

    engine = build_similarity_engine(
        df,
        top_k=20
    )

    opportunity = (
        calculate_market_opportunity(
            engine
        )
    )

    product_data = engine[
        "product_data"
    ]

    idx = find_product_index(
        product_data,
        brand_name
    )

    if idx is None:
        return {}

    product_id = (
        product_data.iloc[idx][
            "Product_ID"
        ]
    )

    result = opportunity[
        opportunity[
            "Product_ID"
        ]
        ==
        product_id
    ]

    if result.empty:
        return {}

    return result.iloc[0].to_dict()


def drug_intelligence_tool(
    df,
    brand_name,
    top_n=10
):

    engine = build_similarity_engine(
        df,
        top_k=max(
            20,
            top_n * 5
        )
    )

    return final_drug_intelligence(
        engine,
        brand_name,
        top_n
    )