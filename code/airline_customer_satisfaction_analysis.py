# =========================================================
# ITAO7105 Data Mining
# Assignment: Descriptive, Predictive, and Text Analytics for Airline Customer Reviews
# Student: Aqeel Shahzad
# =========================================================

# =========================================================
# 0. COLAB SETUP
# =========================================================
!pip -q install vaderSentiment transformers torch accelerate sentencepiece

# =========================================================
# 1. IMPORTS AND GLOBAL SETTINGS
# =========================================================
import warnings
import re
import random
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from google.colab import files
from IPython.display import display

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", None)

RANDOM_STATE = 40495557
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 140

print("Random state:", RANDOM_STATE)

# =========================================================
# 1A. DOWNLOAD NLTK RESOURCES
# =========================================================
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================================================
# 3. DATA UPLOAD
# =========================================================
uploaded = files.upload()

if len(uploaded) == 0:
    raise FileNotFoundError("No file uploaded. Please upload the airline review CSV.")

uploaded_filenames = list(uploaded.keys())
csv_candidates = [f for f in uploaded_filenames if f.lower().endswith(".csv")]

if len(csv_candidates) == 0:
    raise FileNotFoundError("No CSV file detected among uploaded files.")

DATA_PATH = csv_candidates[0]
print("Using CSV file:", DATA_PATH)

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def print_section(title: str) -> None:
    """Print a consistent section header for readable notebook/script output."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def show_table(title: str, table: pd.DataFrame, index: bool = True) -> None:
    """Display a titled table using a consistent print format."""
    print_section(title)
    if index:
        display(table)
    else:
        display(table.reset_index(drop=True))


def finalise_plot(title: str) -> None:
    """
    Apply consistent final formatting to plots so visual style stays aligned
    across the full assignment.
    """
    ax = plt.gca()
    ax.set_title(title, fontsize=13.5, pad=12, weight="bold")
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.show()


def add_bar_labels(
    bars,
    labels,
    y_offset: float,
    fontsize: int = 10,
) -> None:
    """Add value labels above bars to improve readability."""
    for bar, label in zip(bars, labels):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def clean_review_date(value: str) -> pd.Timestamp:
    """
    Standardise review dates by removing ordinal suffixes before parsing.
    This is needed for later temporal analysis and robustness testing.
    """
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()
    value = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", value)

    return pd.to_datetime(value, format="%d %B %Y", errors="coerce")


def clean_text_field(series: pd.Series) -> pd.Series:
    """
    Standardise object/string columns so text comparisons and grouping behave
    consistently across the script.
    """
    return series.astype("string").str.strip()

# =========================================================
# 5. DATA LOADING AND VARIABLE SETUP
# =========================================================
df = pd.read_csv(DATA_PATH)
df_raw = df.copy()

print_section("DATASET LOADED")
print(f"Shape: {df.shape}")
display(df.head())

numeric_cols = [
    "Overall_Rating",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]

categorical_cols = [
    "Airline Name",
    "Review_Title",
    "Review",
    "Aircraft",
    "Type Of Traveller",
    "Seat Type",
    "Route",
]

optional_services = [
    "Wifi & Connectivity",
    "Inflight Entertainment",
    "Food & Beverages",
]

core_services = [
    "Seat Comfort",
    "Cabin Staff Service",
    "Ground Service",
]

date_cols = [
    "Review Date",
    "Date Flown",
]

# %% =========================================================
# SECTION 3. DATA PRE-PROCESSING (DATA CLEANING AND PREPARATION)
# =========================================================

# ---------------------------------------------------------
# 3.1 For Numerical Data:
# Initial inspection and type standardisation
# ---------------------------------------------------------
print_section("3.1 INITIAL DATA INSPECTION")
print(df.dtypes)
print("\nRecommended values:", df["Recommended"].unique())
print("Verified values:", df["Verified"].unique())
print("Seat Type values:", df["Seat Type"].unique()[:10])

# Standardise the binary target early
df["Recommended"] = (
    df["Recommended"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
    .astype("Int64")
)

# Convert numerical ratings
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Standardise key text/categorical variables
for col in categorical_cols:
    df[col] = clean_text_field(df[col])

# Parse dates
df["Review Date"] = df["Review Date"].apply(clean_review_date)
df["Date Flown"] = pd.to_datetime(
    df["Date Flown"].astype("string"),
    format="%b-%y",
    errors="coerce",
)

df["Review_Year"] = df["Review Date"].dt.year
df["Review_Month"] = df["Review Date"].dt.month
df["Flown_Year"] = df["Date Flown"].dt.year
df["Flown_Month"] = df["Date Flown"].dt.month

print_section("3.1 POST-STANDARDISATION DATA TYPES")
print(df.dtypes)

# ---------------------------------------------------------
# 3.1A For Numerical Data:
# Flag reviews recorded before the reported flight month
# ---------------------------------------------------------
df["Review_Before_Flight_Flag"] = (
    (df["Review Date"].notna()) &
    (df["Date Flown"].notna()) &
    (df["Review Date"] < df["Date Flown"])
).astype("Int64")

# ---------------------------------------------------------
# 3.1B For Numerical Data:
# Print full reviews for cases where Review Date < Date Flown
# ---------------------------------------------------------
past_review_full = df.loc[
    df["Review_Before_Flight_Flag"] == 1,
    [
        "Airline Name",
        "Review Date",
        "Date Flown",
        "Review_Title",
        "Review",
        "Type Of Traveller",
        "Seat Type",
        "Route",
        "Recommended",
    ]
].copy()

past_review_full["Days_Difference"] = (
    past_review_full["Date Flown"] - past_review_full["Review Date"]
).dt.days

past_review_full = past_review_full.sort_values(
    by=["Days_Difference", "Review Date"],
    ascending=[False, True]
).reset_index(drop=True)

print_section("3.1B FULL REVIEWS: REVIEW DATE BEFORE FLIGHT DATE")

if past_review_full.empty:
    print("No cases found where the review date is earlier than the recorded flight date.")
else:
    for i, row in past_review_full.iterrows():
        print("\n" + "-" * 100)
        print(f"Case {i + 1}")
        print("-" * 100)
        print(f"Airline Name      : {row['Airline Name']}")
        print(f"Review Date       : {row['Review Date']}")
        print(f"Date Flown        : {row['Date Flown']}")
        print(f"Days Difference   : {row['Days_Difference']}")
        print(f"Review Title      : {row['Review_Title']}")
        print(f"Traveller Type    : {row['Type Of Traveller']}")
        print(f"Seat Type         : {row['Seat Type']}")
        print(f"Route             : {row['Route']}")
        print(f"Recommended       : {row['Recommended']}")
        print("\nFull Review:")
        print(row["Review"])

# ---------------------------------------------------------
# 3.1C Manual validation note
# ---------------------------------------------------------
print_section("3.1C MANUAL VALIDATION NOTE")
print(
    "All cases where the review date appeared earlier than the recorded flight date "
    "were manually read and assessed. These entries were retained because the review "
    "text indicated valid complaints or pre-flight service issues, rather than clear "
    "data-entry errors."
)

# ---------------------------------------------------------
# 3.2 For Numerical Data:
# Identify missingness before cleaning
# ---------------------------------------------------------
missing_before = pd.DataFrame(
    {
        "Missing_Before": df[numeric_cols].isna().sum(),
        "Percent_Before": (df[numeric_cols].isna().mean() * 100).round(2),
    }
).sort_values("Percent_Before", ascending=False)

show_table("3.2 MISSING VALUES BEFORE CLEANING", missing_before)

# ---------------------------------------------------------
# 3.3 For Numerical Data:
# Examine systematic missingness in optional services
# ---------------------------------------------------------
optional_summary_rows = []

for col in optional_services:
    available_mask = df[col].notna()
    optional_summary_rows.append(
        {
            "Feature": col,
            "Mean_Rating_Available_Only": round(df.loc[available_mask, col].mean(), 2),
            "Availability_Percent": round(available_mask.mean() * 100, 2),
            "Not_Available_Percent": round((1 - available_mask.mean()) * 100, 2),
        }
    )

optional_summary = pd.DataFrame(optional_summary_rows)
show_table(
    "3.3 OPTIONAL SERVICES: QUALITY VERSUS AVAILABILITY",
    optional_summary,
    index=False,
)

# ---------------------------------------------------------
# 3.4 For Numerical Data:
# Logical conflicts between Overall Rating and Recommendation
# ---------------------------------------------------------
df["Inconsistent_Rating_Flag"] = (
    ((pd.to_numeric(df["Overall_Rating"], errors="coerce") <= 5) & (df["Recommended"] == 1)) |
    ((pd.to_numeric(df["Overall_Rating"], errors="coerce") > 5) & (df["Recommended"] == 0))
).astype("Int64")

inconsistency_summary = pd.DataFrame({
    "Count": df["Inconsistent_Rating_Flag"].value_counts(dropna=False).sort_index()
})
inconsistency_summary.index = ["Consistent / Not Flagged", "Potentially Inconsistent"]

show_table("3.4 LOGICAL CONFLICT FLAG: OVERALL RATING VS RECOMMENDATION", inconsistency_summary)

# ---------------------------------------------------------
# 3.5 For Numerical Data:
# Alternative cleaning / imputation strategies
# ---------------------------------------------------------

# Strategy comparison 1:
# Dropping all rows with missing optional-service ratings
df_drop_optional = df.dropna(subset=optional_services).copy()

rec_original = df["Recommended"].value_counts(normalize=True)
rec_dropped = df_drop_optional["Recommended"].value_counts(normalize=True)

optional_strategy_summary = pd.DataFrame(
    {
        "Metric": [
            "Total Rows",
            "Rows Removed",
            "Percent Rows Removed",
            "Not Recommended Percent",
            "Recommended Percent",
        ],
        "Original Dataset": [
            f"{len(df):,}",
            "0",
            "0.00",
            f"{rec_original.get(0, 0) * 100:.2f}",
            f"{rec_original.get(1, 0) * 100:.2f}",
        ],
        "After Dropping Optional-Service Missingness": [
            f"{len(df_drop_optional):,}",
            f"{len(df) - len(df_drop_optional):,}",
            f"{((len(df) - len(df_drop_optional)) / len(df)) * 100:.2f}",
            f"{rec_dropped.get(0, 0) * 100:.2f}",
            f"{rec_dropped.get(1, 0) * 100:.2f}",
        ],
    }
)

show_table(
    "3.5 ALTERNATIVE STRATEGY 1: DROPPING OPTIONAL-SERVICE MISSINGNESS",
    optional_strategy_summary,
    index=False,
)

# Strategy comparison 2:
# Zero-imputation versus median-imputation for core services
df_zero_core = df.copy()
df_median_core = df.copy()

for col in core_services:
    df_zero_core[col] = df_zero_core[col].fillna(0)
    df_median_core[col] = df_median_core[col].fillna(df_median_core[col].median())

core_strategy_comparison = pd.DataFrame(
    {
        "Original_Mean": df[core_services].mean().round(2),
        "Zero_Imputed_Mean": df_zero_core[core_services].mean().round(2),
        "Median_Imputed_Mean": df_median_core[core_services].mean().round(2),
        "Original_Median": df[core_services].median().round(2),
        "Zero_Imputed_Median": df_zero_core[core_services].median().round(2),
        "Median_Imputed_Median": df_median_core[core_services].median().round(2),
        "Missing_Count": df[core_services].isna().sum(),
    }
)

show_table(
    "3.5 ALTERNATIVE STRATEGY 2: ZERO VERSUS MEDIAN IMPUTATION FOR CORE SERVICES",
    core_strategy_comparison,
)

# ---------------------------------------------------------
# 3.6 For Numerical Data:
# ---------------------------------------------------------
# Final strategy retained:
# - Drop rows missing Overall_Rating or Value For Money
# - Treat optional-service missingness as service non-availability
# - Median-impute core service ratings

df_clean = df.copy()

df_clean = df_clean.dropna(subset=["Overall_Rating", "Value For Money"]).copy()

for col in optional_services:
    df_clean[f"{col}_available"] = df_clean[col].notna().astype(int)
    df_clean[col] = df_clean[col].fillna(0)

for col in core_services:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# ---------------------------------------------------------
# 3.7 For Numerical Data:
# Pre- and post-cleaning comparison
# ---------------------------------------------------------
missing_after = pd.DataFrame(
    {
        "Missing_After": df_clean[numeric_cols].isna().sum(),
        "Percent_After": (df_clean[numeric_cols].isna().mean() * 100).round(2),
    }
).sort_values("Percent_After", ascending=False)

show_table("3.7 MISSING VALUES AFTER CLEANING", missing_after)

missing_compare = missing_before.join(missing_after)
missing_compare["Reduction"] = (
    missing_compare["Missing_Before"] - missing_compare["Missing_After"]
)
missing_compare["Percent_Reduction"] = (
    missing_compare["Percent_Before"] - missing_compare["Percent_After"]
).round(2)

show_table("3.7 BEFORE AND AFTER MISSING-VALUE COMPARISON", missing_compare)

comparison_plot_df = pd.DataFrame(
    {
        "Original": df[core_services].mean(),
        "Zero Imputation": df_zero_core[core_services].mean(),
        "Median Imputation": df_median_core[core_services].mean(),
    }
)

comparison_plot_df.plot(kind="bar", figsize=(10, 5), edgecolor="black")
plt.ylabel("Average Rating")
plt.xlabel("Core Service Variable")
plt.xticks(rotation=0)
plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
plt.gca().set_axisbelow(True)
finalise_plot("3.7 Impact of Alternative Imputation Choices on Core Service Ratings")

print_section("3.7 FINAL CLEANED DATASET")
print(f"Final cleaned shape: {df_clean.shape}")
print(df_clean[numeric_cols].isna().sum())

# ---------------------------------------------------------
# 3.8 Cleaning audit table
# ---------------------------------------------------------
audit_rows = []

def add_audit_row(
    variable: str,
    rows_before: int,
    method: str,
    rows_removed: int,
    rows_after: int,
) -> None:
    """Store a transparent record of how each cleaning decision affected the data."""
    audit_rows.append(
        {
            "Variable_Name": variable,
            "Rows_Before": rows_before,
            "Cleaning_Method_Used": method,
            "Rows_Removed": rows_removed,
            "Rows_After": rows_after,
        }
    )

df_audit = df.copy()

rows_before = len(df_audit)
rows_removed = df_audit["Overall_Rating"].isna().sum()
df_audit = df_audit.dropna(subset=["Overall_Rating"]).copy()
add_audit_row("Overall_Rating", rows_before, "Drop missing rows", rows_removed, len(df_audit))

rows_before = len(df_audit)
rows_removed = df_audit["Value For Money"].isna().sum()
df_audit = df_audit.dropna(subset=["Value For Money"]).copy()
add_audit_row("Value For Money", rows_before, "Drop missing rows", rows_removed, len(df_audit))

for col in optional_services:
    rows_before = len(df_audit)
    df_audit[f"{col}_available"] = df_audit[col].notna().astype(int)
    df_audit[col] = df_audit[col].fillna(0)
    add_audit_row(
        col,
        rows_before,
        "Add availability indicator and fill missing with 0",
        0,
        len(df_audit),
    )

for col in core_services:
    rows_before = len(df_audit)
    median_value = df_audit[col].median()
    df_audit[col] = df_audit[col].fillna(median_value)
    add_audit_row(
        col,
        rows_before,
        f"Fill missing with median ({median_value:.1f})",
        0,
        len(df_audit),
    )

cleaning_audit = pd.DataFrame(audit_rows)
show_table("3.8 VARIABLE-LEVEL CLEANING AUDIT", cleaning_audit, index=False)

# ---------------------------------------------------------
# 3.9 For Textual Data:
# Create text analysis dataset
# ---------------------------------------------------------
text_df = df_clean[
    [
        "Review",
        "Recommended",
        "Overall_Rating",
        "Airline Name",
        "Seat Type",
        "Type Of Traveller",
        "Route",
    ]
].dropna(subset=["Review"]).copy()

print_section("3.9 TEXT DATASET CREATED")
print(f"Text dataset shape: {text_df.shape}")

# ---------------------------------------------------------
# 3.10 For Textual Data:
# Context-specific preprocessing resources
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()

base_stopwords = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never", "cannot"}
custom_stopwords = base_stopwords - negation_words

domain_keywords = {
    "delay", "delayed", "late",
    "cancel", "cancelled", "canceled",
    "refund", "voucher",
    "staff", "crew", "service",
    "seat", "comfort",
    "food", "meal",
    "wifi", "internet",
    "baggage", "luggage", "bag",
    "boarding", "checkin", "check-in",
    "customer", "support",
}

# ---------------------------------------------------------
# 3.11 For Textual Data:
# Text cleaning functions
# ---------------------------------------------------------
def basic_clean_text(text: str) -> str:
    """
    Clean review text while preserving sentiment-carrying structure such as
    negation.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")
    text = text.replace("n't", " not")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    text = text.replace("'m", " am")

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> list:
    """
    Tokenise and lemmatise while preserving negation and airline-specific vocabulary.
    """
    tokens = text.split()
    cleaned_tokens = []

    for token in tokens:
        if token in negation_words:
            cleaned_tokens.append(token)
        elif token in domain_keywords:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
        elif token not in custom_stopwords and len(token) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(token))

    return cleaned_tokens

# ---------------------------------------------------------
# 3.12 For Textual Data:
# Apply preprocessing pipeline
# ---------------------------------------------------------
text_df["review_clean_basic"] = text_df["Review"].apply(basic_clean_text)
text_df["review_tokens"] = text_df["review_clean_basic"].apply(tokenize_and_lemmatize)
text_df["review_clean_final"] = text_df["review_tokens"].apply(lambda tokens: " ".join(tokens))

print_section("3.12 SAMPLE CLEANED REVIEWS")
print(text_df[["Review", "review_clean_final"]].head(3))

# ---------------------------------------------------------
# 3.13 For Textual Data:
# Dataset-specific issue themes
# ---------------------------------------------------------
theme_dictionary = {
    "delay_theme": ["delay", "delayed", "late"],
    "cancellation_theme": ["cancel", "cancelled", "canceled"],
    "refund_theme": ["refund", "voucher"],
    "staff_theme": ["staff", "crew", "service"],
    "seat_theme": ["seat", "comfort"],
    "food_theme": ["food", "meal"],
    "wifi_theme": ["wifi", "internet"],
    "baggage_theme": ["baggage", "luggage", "bag"],
    "boarding_theme": ["boarding", "checkin", "check", "gate"],
    "customer_support_theme": ["customer", "support", "call", "email"],
}

for theme_name, keywords in theme_dictionary.items():
    text_df[theme_name] = text_df["review_clean_final"].apply(
        lambda text: int(any(word in text.split() for word in keywords))
    )

theme_summary = pd.DataFrame(
    {"Theme_Count": text_df[list(theme_dictionary.keys())].sum()}
).sort_values(by="Theme_Count", ascending=False)

show_table("3.13 SERVICE-THEME SUMMARY", theme_summary)

theme_by_recommendation_rows = []

for theme_name in theme_dictionary.keys():
    grouped = text_df.groupby("Recommended")[theme_name].mean() * 100
    theme_by_recommendation_rows.append(
        {
            "Theme": theme_name,
            "Not_Recommended_Percent": round(grouped.get(0, np.nan), 2),
            "Recommended_Percent": round(grouped.get(1, np.nan), 2),
            "Difference_pp": round(grouped.get(0, 0) - grouped.get(1, 0), 2),
        }
    )

theme_by_recommendation_df = pd.DataFrame(theme_by_recommendation_rows).sort_values(
    "Difference_pp",
    ascending=False,
)

show_table(
    "3.13 THEME PREVALENCE BY RECOMMENDATION",
    theme_by_recommendation_df,
    index=False,
)

# ---------------------------------------------------------
# 3.14 For Textual Data:
# Distinctive linguistic patterns via bigrams
# ---------------------------------------------------------
vectorizer_bigrams = CountVectorizer(
    ngram_range=(2, 2),
    min_df=20,
    max_df=0.80,
)

X_bigrams = vectorizer_bigrams.fit_transform(text_df["review_clean_final"])
bigram_counts = np.asarray(X_bigrams.sum(axis=0)).ravel()

bigrams_df = pd.DataFrame(
    {
        "Bigram": vectorizer_bigrams.get_feature_names_out(),
        "Count": bigram_counts,
    }
).sort_values(by="Count", ascending=False)

show_table("3.14 TOP BIGRAMS OVERALL", bigrams_df.head(20), index=False)

not_rec_text = text_df.loc[text_df["Recommended"] == 0, "review_clean_final"]
rec_text = text_df.loc[text_df["Recommended"] == 1, "review_clean_final"]

vectorizer_not = CountVectorizer(ngram_range=(2, 2), min_df=10, max_df=0.85)
vectorizer_yes = CountVectorizer(ngram_range=(2, 2), min_df=10, max_df=0.85)

X_not = vectorizer_not.fit_transform(not_rec_text)
X_yes = vectorizer_yes.fit_transform(rec_text)

not_bigrams_df = pd.DataFrame(
    {
        "Bigram": vectorizer_not.get_feature_names_out(),
        "Count": np.asarray(X_not.sum(axis=0)).ravel(),
    }
).sort_values(by="Count", ascending=False)

yes_bigrams_df = pd.DataFrame(
    {
        "Bigram": vectorizer_yes.get_feature_names_out(),
        "Count": np.asarray(X_yes.sum(axis=0)).ravel(),
    }
).sort_values(by="Count", ascending=False)

show_table(
    "3.14 TOP BIGRAMS IN NOT-RECOMMENDED REVIEWS",
    not_bigrams_df.head(15),
    index=False,
)
show_table(
    "3.14 TOP BIGRAMS IN RECOMMENDED REVIEWS",
    yes_bigrams_df.head(15),
    index=False,
)

# ---------------------------------------------------------
# 3.15 For Textual Data:
# Mixed and contradictory wording flags
# ---------------------------------------------------------
positive_cues = {
    "good", "great", "excellent", "comfortable", "friendly",
    "helpful", "pleasant", "smooth", "nice", "amazing",
}

negative_cues = {
    "bad", "poor", "terrible", "worst", "delay",
    "cancel", "dirty", "rude", "unprofessional",
}

def count_positive_cues(tokens: list) -> int:
    return sum(1 for word in tokens if word in positive_cues)

def count_negative_cues(tokens: list) -> int:
    return sum(1 for word in tokens if word in negative_cues)

def has_negated_positive(tokens: list) -> int:
    for i in range(len(tokens) - 1):
        if tokens[i] == "not" and tokens[i + 1] in positive_cues:
            return 1
    return 0

text_df["positive_cue_count"] = text_df["review_tokens"].apply(count_positive_cues)
text_df["negative_cue_count"] = text_df["review_tokens"].apply(count_negative_cues)
text_df["negated_positive"] = text_df["review_tokens"].apply(has_negated_positive)

mixed_sentiment_reviews = text_df[
    (text_df["positive_cue_count"] >= 2) &
    (text_df["negative_cue_count"] >= 2)
][["Overall_Rating", "Recommended", "Review"]].head(10)

contradiction_reviews = text_df[
    (text_df["Recommended"] == 0) &
    (text_df["positive_cue_count"] >= 2) &
    (text_df["negated_positive"] == 0)
][["Overall_Rating", "Recommended", "Review"]].head(10)

show_table("3.15 MIXED-SENTIMENT REVIEW EXAMPLES", mixed_sentiment_reviews, index=False)
show_table(
    "3.15 POSITIVE-WORDING BUT NOT-RECOMMENDED EXAMPLES",
    contradiction_reviews,
    index=False
)

all_tokens = [token for tokens in text_df["review_tokens"] for token in tokens]
token_summary = pd.DataFrame(
    Counter(all_tokens).most_common(30),
    columns=["Token", "Count"],
)
show_table("3.15 TOP MEANINGFUL TOKENS AFTER PREPROCESSING", token_summary, index=False)

# %% =========================================================
# SECTION 4.a. DESCRIPTIVE AND PREDICTIVE ANALYTICS (NUMERICAL FEATURES)
# =========================================================

# ---------------------------------------------------------
# 4.a.1 Descriptive statistics for all numerical features
# ---------------------------------------------------------
desc_stats = (
    df_clean[numeric_cols]
    .describe()
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(2)
)

desc_stats["missing_count"] = df_clean[numeric_cols].isna().sum()
desc_stats["missing_percent"] = (df_clean[numeric_cols].isna().mean() * 100).round(2)
desc_stats = desc_stats.reset_index().rename(columns={"index": "Feature"})

show_table("4.a.1 DESCRIPTIVE STATISTICS FOR NUMERICAL VARIABLES", desc_stats, index=False)

# ---------------------------------------------------------
# 4.a.2 Visualisation 1:
# Distribution of customer recommendation
# ---------------------------------------------------------
recommend_counts = df_clean["Recommended"].value_counts().sort_index()
recommend_labels = ["No", "Yes"]
recommend_perc = (recommend_counts / recommend_counts.sum() * 100).round(1)

plt.figure(figsize=(7.6, 5.4))

bars = plt.bar(
    recommend_labels,
    recommend_counts.values,
    color=["#F08D39", "#2F5D8A"],
    edgecolor="black",
    linewidth=1.0,
    width=0.45,
)

plt.xlabel("Recommended", fontsize=11)
plt.ylabel("Number of Reviews", fontsize=11)
plt.ylim(0, recommend_counts.max() * 1.12)

plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
plt.gca().grid(False, axis="x")
plt.gca().set_axisbelow(True)

for bar, count, pct in zip(bars, recommend_counts.values, recommend_perc.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count + recommend_counts.max() * 0.015,
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#bdbdbd")
ax.spines["bottom"].set_color("#bdbdbd")

finalise_plot("Distribution of Customer Recommendation")

# ---------------------------------------------------------
# 4.a.3 Visualisation 2:
# Consistent versus inconsistent rating-recommendation cases
# ---------------------------------------------------------
df_clean["Inconsistent_Rating"] = (
    ((df_clean["Overall_Rating"] <= 5) & (df_clean["Recommended"] == 1)) |
    ((df_clean["Overall_Rating"] > 5) & (df_clean["Recommended"] == 0))
)

consistency_counts = (
    df_clean["Inconsistent_Rating"]
    .value_counts()
    .reindex([False, True], fill_value=0)
)

consistency_labels = ["Consistent", "Inconsistent"]
consistency_perc = (consistency_counts / consistency_counts.sum() * 100).round(1)

plt.figure(figsize=(7.8, 5.4))

bars = plt.bar(
    consistency_labels,
    consistency_counts.values,
    color=["#2F5D8A", "#F08D39"],
    edgecolor="black",
    linewidth=1.0,
    width=0.45,
)

plt.xlabel("")
plt.ylabel("Number of Reviews", fontsize=11)
plt.ylim(0, consistency_counts.max() * 1.12)

plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
plt.gca().grid(False, axis="x")
plt.gca().set_axisbelow(True)

for bar, count, pct in zip(bars, consistency_counts.values, consistency_perc.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count + consistency_counts.max() * 0.015,
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#bdbdbd")
ax.spines["bottom"].set_color("#bdbdbd")

finalise_plot("Consistent vs Inconsistent Rating-Recommendation Cases")

# ---------------------------------------------------------
# 4.a.4 Visualisation 3:
# Recommendation rate by value for money
# ---------------------------------------------------------
value_rec = df_clean.groupby("Value For Money")["Recommended"].mean() * 100
value_counts = df_clean["Value For Money"].value_counts().sort_index()

plt.figure(figsize=(8.8, 5.6))

plt.plot(
    value_rec.index,
    value_rec.values,
    marker="o",
    linewidth=2.2,
    markersize=7.5,
    color="#2F5D8A",
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=3,
)

for x, y in zip(value_rec.index, value_rec.values):
    if y < 5:
        offset = 3.0
    elif y > 90:
        offset = 1.5
    else:
        offset = 2.0

    plt.text(
        x,
        y + offset,
        f"n={value_counts.get(x, 0)}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylim(0, 102)
plt.xlim(-0.25, 5.25)
plt.xlabel("Value for Money Rating", fontsize=11)
plt.ylabel("Recommendation Rate (%)", fontsize=11)
plt.xticks(sorted(df_clean["Value For Money"].dropna().unique()))

plt.grid(axis="y", color="#e9e9e9", linewidth=0.7)
plt.gca().grid(False, axis="x")
plt.gca().set_axisbelow(True)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#bdbdbd")
ax.spines["bottom"].set_color("#bdbdbd")

finalise_plot("Recommendation Rate by Value for Money")

# ---------------------------------------------------------
# 4.a.5 Visualisation 4:
# Correlation between service rating features
# ---------------------------------------------------------
corr_matrix = df_clean[numeric_cols].corr()

display_labels = [
    "Overall Rating",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]

corr_display = corr_matrix.copy()
corr_display.index = display_labels
corr_display.columns = display_labels

mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)

plt.figure(figsize=(8.8, 7.2))
ax = sns.heatmap(
    corr_display,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.05,
    vmax=1.00,
    center=0.50,
    square=True,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={
        "shrink": 0.82,
        "pad": 0.03,
    },
    annot_kws={"size": 8.5},
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_facecolor("white")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)

finalise_plot("Correlation Heatmap of Service Features")

# ---------------------------------------------------------
# 4.a.6 Visualisation 5:
# Recommendation rate by seat type
# ---------------------------------------------------------
seat_df = df_clean.dropna(subset=["Seat Type", "Recommended"]).copy()
seat_df["Seat Type"] = seat_df["Seat Type"].astype(str).str.strip()

seat_order = ["First Class", "Business Class", "Premium Economy", "Economy Class"]

seat_summary = (
    seat_df.groupby("Seat Type", as_index=False)
    .agg(
        Recommendation_Rate=("Recommended", "mean"),
        Review_Count=("Recommended", "size"),
    )
)

seat_summary = (
    seat_summary[seat_summary["Seat Type"].isin(seat_order)]
    .assign(
        Seat_Type_Order=lambda x: pd.Categorical(
            x["Seat Type"], categories=seat_order, ordered=True
        )
    )
    .sort_values("Seat_Type_Order")
)

seat_summary["Recommendation_Rate"] = (seat_summary["Recommendation_Rate"] * 100).round(1)

seat_colors = {
    "First Class": "#1F3A5F",
    "Business Class": "#2F5D8A",
    "Premium Economy": "#6F8FAF",
    "Economy Class": "#B7C7D9",
}

bar_colors = [seat_colors[s] for s in seat_summary["Seat Type"]]

plt.figure(figsize=(8.6, 5.6))

bars = plt.barh(
    seat_summary["Seat Type"],
    seat_summary["Recommendation_Rate"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.8,
    height=0.75,
)

plt.gca().invert_yaxis()

for bar, rate, count in zip(
    bars, seat_summary["Recommendation_Rate"], seat_summary["Review_Count"]
):
    plt.text(
        rate + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{rate:.1f}%  (n={count:,})",
        va="center",
        ha="left",
        fontsize=9,
    )

plt.xlim(0, 70)
plt.xlabel("Recommendation Rate (%)", fontsize=11)
plt.ylabel("Seat Type", fontsize=11)

plt.grid(axis="x", color="#eaeaea", linewidth=0.6)
plt.gca().grid(False, axis="y")
plt.gca().set_axisbelow(True)

ax = plt.gca()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_color("#c0c0c0")
ax.spines["bottom"].set_color("#c0c0c0")

finalise_plot("Recommendation Rate by Seat Type")

# ---------------------------------------------------------
# 4.a.7 Visualisation 6:
# Correlation of numerical features with recommendation
# ---------------------------------------------------------
corr_target = (
    df_clean[numeric_cols]
    .corrwith(df_clean["Recommended"])
    .sort_values(ascending=True)
)

display_name_map = {
    "Overall_Rating": "Overall Rating",
    "Seat Comfort": "Seat Comfort",
    "Cabin Staff Service": "Cabin Staff Service",
    "Food & Beverages": "Food & Beverages",
    "Ground Service": "Ground Service",
    "Inflight Entertainment": "Inflight Entertainment",
    "Wifi & Connectivity": "Wifi & Connectivity",
    "Value For Money": "Value For Money",
}

corr_plot = corr_target.rename(index=display_name_map)

cmap = plt.cm.Blues
norm = plt.Normalize(corr_plot.min(), corr_plot.max())
colors = cmap(0.20 + 0.78 * norm(corr_plot.values))

plt.figure(figsize=(8.8, 5.8))

bars = plt.barh(
    corr_plot.index,
    corr_plot.values,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
    height=0.76,
)

for bar, value in zip(bars, corr_plot.values):
    plt.text(
        value + 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        ha="left",
        fontsize=9,
    )

plt.xlim(0, 0.89)
plt.xlabel("Correlation with Recommendation", fontsize=11)
plt.ylabel("")

plt.grid(axis="x", color="#eaeaea", linewidth=0.6)
plt.gca().grid(False, axis="y")
plt.gca().set_axisbelow(True)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c0c0c0")
ax.spines["bottom"].set_color("#c0c0c0")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

finalise_plot("Correlation of Numerical Features with Recommendation")

# ---------------------------------------------------------
# 4.a.8 Visualisation 7:
# Recommendation rate by overall rating
# ---------------------------------------------------------
rating_rec = df_clean.groupby("Overall_Rating")["Recommended"].mean() * 100
rating_counts = df_clean["Overall_Rating"].value_counts().sort_index()

plt.figure(figsize=(8.8, 5.6))

plt.plot(
    rating_rec.index,
    rating_rec.values,
    marker="o",
    linewidth=2.2,
    markersize=7,
    color="#2F5D8A",
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=3,
)

for x, y in zip(rating_rec.index, rating_rec.values):
    if y < 10:
        offset = 3.2
    elif y > 90:
        offset = 1.5
    else:
        offset = 2.2

    plt.text(
        x,
        y + offset,
        f"n={rating_counts.get(x, 0):,}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

plt.ylim(0, 102)
plt.xlim(0.6, 9.4)
plt.xlabel("Overall Rating", fontsize=11)
plt.ylabel("Recommendation Rate (%)", fontsize=11)
plt.xticks(range(1, 10))

plt.grid(axis="y", color="#e9e9e9", linewidth=0.7)
plt.gca().grid(False, axis="x")
plt.gca().set_axisbelow(True)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#bdbdbd")
ax.spines["bottom"].set_color("#bdbdbd")

finalise_plot("Recommendation Rate by Overall Rating")

# ---------------------------------------------------------
# 4.a.9 Imports for predictive analytics
# ---------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 4.a.10 Modelling helper functions
# ---------------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob, model_name: str) -> dict:
    """
    Evaluate a binary classifier using standard metrics.
    """
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
    }

    print_section(f"MODEL PERFORMANCE: {model_name}")
    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return results


def evaluate_at_threshold(y_true, y_prob, threshold: float, model_name: str) -> dict:
    """
    Evaluate performance at a specified threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    print_section(f"THRESHOLD EVALUATION: {model_name} | threshold={threshold}")
    for metric, value in results.items():
        if metric not in ["Model", "Threshold"]:
            print(f"{metric}: {value:.4f}")

    return results


def print_conf_matrix(y_true, y_pred, model_name: str) -> pd.DataFrame:
    """Print a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    show_table(f"CONFUSION MATRIX: {model_name}", cm_df)
    return cm_df


def format_results_table(results_list: list, labels: list, title: str) -> pd.DataFrame:
    """
    Convert a list of results dictionaries into one clean comparison table.
    """
    results_df = pd.DataFrame(results_list).copy()
    results_df["Model"] = labels
    results_df = (
        results_df
        .drop(columns=["Threshold"], errors="ignore")
        .set_index("Model")
        .round(4)
    )
    if "ROC_AUC" in results_df.columns:
        results_df["ROC_AUC"] = results_df["ROC_AUC"].fillna("-")
    show_table(title, results_df)
    return results_df

# ---------------------------------------------------------
# 4.a.11 Modelling dataset and random split
# ---------------------------------------------------------
model_features = [
    "Overall_Rating",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
    "Wifi & Connectivity_available",
    "Inflight Entertainment_available",
    "Food & Beverages_available",
]

target = "Recommended"

model_df = df_clean[model_features + [target, "Review Date", "Airline Name"]].copy()

print_section("4.a.11 MODELLING DATASET")
print("Shape:", model_df.shape)
print(model_df[target].value_counts(normalize=True).round(4))

X = model_df[model_features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print_section("4.a.11 RANDOM TRAIN-TEST SPLIT")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print(y_train.value_counts(normalize=True).round(4))
print(y_test.value_counts(normalize=True).round(4))

# ---------------------------------------------------------
# 4.a.12 Logistic Regression model
# ---------------------------------------------------------
log_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_pipeline.fit(X_train, y_train)

log_pred = log_pipeline.predict(X_test)
log_prob = log_pipeline.predict_proba(X_test)[:, 1]

log_results = evaluate_model(
    y_test,
    log_pred,
    log_prob,
    "Logistic Regression",
)

# ---------------------------------------------------------
# 4.a.13 Random Forest model
# ---------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_results = evaluate_model(
    y_test,
    rf_pred,
    rf_prob,
    "Random Forest",
)

# ---------------------------------------------------------
# 4.a.14 Threshold comparison
# ---------------------------------------------------------
log_threshold_03 = evaluate_at_threshold(
    y_test,
    log_prob,
    0.30,
    "Logistic Regression",
)

rf_threshold_03 = evaluate_at_threshold(
    y_test,
    rf_prob,
    0.30,
    "Random Forest",
)

log_pred_03 = (log_prob >= 0.30).astype(int)
rf_pred_03 = (rf_prob >= 0.30).astype(int)

baseline_comparison_full = format_results_table(
    results_list=[log_results, rf_results, log_threshold_03, rf_threshold_03],
    labels=[
        "Logistic Regression (0.5)",
        "Random Forest (0.5)",
        "Logistic Regression (0.3)",
        "Random Forest (0.3)",
    ],
    title="4.a.14 BASELINE MODEL COMPARISON",
)

# ---------------------------------------------------------
# 4.a.15 LASSO Logistic Regression for feature selection
# ---------------------------------------------------------
lasso_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

lasso_param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
}

lasso_grid = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid=lasso_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

lasso_grid.fit(X_train, y_train)

best_lasso = lasso_grid.best_estimator_

print_section("4.a.15 LASSO TUNING RESULTS")
print("Best parameters:", lasso_grid.best_params_)
print("Best CV ROC-AUC:", round(lasso_grid.best_score_, 4))

lasso_coef = best_lasso.named_steps["model"].coef_[0]

lasso_features = pd.DataFrame(
    {
        "Feature": model_features,
        "Coefficient": lasso_coef,
    }
)

selected_features = (
    lasso_features[lasso_features["Coefficient"] != 0]
    .copy()
    .sort_values(by="Coefficient", key=np.abs, ascending=False)
)

show_table("4.a.15 FEATURES SELECTED BY LASSO", selected_features, index=False)

lasso_pred = best_lasso.predict(X_test)
lasso_prob = best_lasso.predict_proba(X_test)[:, 1]

lasso_results = evaluate_model(
    y_test,
    lasso_pred,
    lasso_prob,
    "LASSO Logistic Regression",
)

lasso_threshold_03 = evaluate_at_threshold(
    y_test,
    lasso_prob,
    0.30,
    "LASSO Logistic Regression",
)

lasso_pred_03 = (lasso_prob >= 0.30).astype(int)

all_model_results = format_results_table(
    results_list=[
        log_results,
        rf_results,
        lasso_results,
        log_threshold_03,
        rf_threshold_03,
        lasso_threshold_03,
    ],
    labels=[
        "Logistic Regression (0.5)",
        "Random Forest (0.5)",
        "LASSO Logistic Regression (0.5)",
        "Logistic Regression (0.3)",
        "Random Forest (0.3)",
        "LASSO Logistic Regression (0.3)",
    ],
    title="4.a.15 MODEL COMPARISON INCLUDING LASSO",
)

# ---------------------------------------------------------
# 4.a.16 Confusion matrices
# ---------------------------------------------------------
print_conf_matrix(y_test, log_pred, "Logistic Regression (0.5)")
print_conf_matrix(y_test, rf_pred, "Random Forest (0.5)")
print_conf_matrix(y_test, lasso_pred, "LASSO Logistic Regression (0.5)")
print_conf_matrix(y_test, log_pred_03, "Logistic Regression (0.3)")
print_conf_matrix(y_test, rf_pred_03, "Random Forest (0.3)")
print_conf_matrix(y_test, lasso_pred_03, "LASSO Logistic Regression (0.3)")

# ---------------------------------------------------------
# 4.a.17 Hyperparameter tuning
# ---------------------------------------------------------
log_tuned_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"],
}

log_grid = GridSearchCV(
    estimator=log_tuned_pipeline,
    param_grid=log_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

log_grid.fit(X_train, y_train)

best_log_model = log_grid.best_estimator_

print_section("4.a.17 TUNED LOGISTIC REGRESSION RESULTS")
print("Best parameters:", log_grid.best_params_)
print("Best CV ROC-AUC:", round(log_grid.best_score_, 4))

log_tuned_pred = best_log_model.predict(X_test)
log_tuned_prob = best_log_model.predict_proba(X_test)[:, 1]

log_tuned_results = evaluate_model(
    y_test,
    log_tuned_pred,
    log_tuned_prob,
    "Tuned Logistic Regression",
)

log_tuned_threshold_03 = evaluate_at_threshold(
    y_test,
    log_tuned_prob,
    0.30,
    "Tuned Logistic Regression",
)

log_tuned_pred_03 = (log_tuned_prob >= 0.30).astype(int)

rf_param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
    "class_weight": [None, "balanced"],
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=rf_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

rf_grid.fit(X_train, y_train)

best_rf_model = rf_grid.best_estimator_

print_section("4.a.17 TUNED RANDOM FOREST RESULTS")
print("Best parameters:", rf_grid.best_params_)
print("Best CV ROC-AUC:", round(rf_grid.best_score_, 4))

rf_tuned_pred = best_rf_model.predict(X_test)
rf_tuned_prob = best_rf_model.predict_proba(X_test)[:, 1]

rf_tuned_results = evaluate_model(
    y_test,
    rf_tuned_pred,
    rf_tuned_prob,
    "Tuned Random Forest",
)

rf_tuned_threshold_03 = evaluate_at_threshold(
    y_test,
    rf_tuned_prob,
    0.30,
    "Tuned Random Forest",
)

rf_tuned_pred_03 = (rf_tuned_prob >= 0.30).astype(int)

tuned_model_results = format_results_table(
    results_list=[
        log_tuned_results,
        rf_tuned_results,
        log_tuned_threshold_03,
        rf_tuned_threshold_03,
    ],
    labels=[
        "Tuned Logistic Regression (0.5)",
        "Tuned Random Forest (0.5)",
        "Tuned Logistic Regression (0.3)",
        "Tuned Random Forest (0.3)",
    ],
    title="4.a.17 TUNED MODEL COMPARISON",
)

baseline_vs_tuned = format_results_table(
    results_list=[
        log_results,
        rf_results,
        log_tuned_results,
        rf_tuned_results,
    ],
    labels=[
        "Baseline Logistic Regression",
        "Baseline Random Forest",
        "Tuned Logistic Regression",
        "Tuned Random Forest",
    ],
    title="4.a.17 BASELINE VERSUS TUNED MODELS",
)

print_conf_matrix(y_test, log_tuned_pred, "Tuned Logistic Regression (0.5)")
print_conf_matrix(y_test, rf_tuned_pred, "Tuned Random Forest (0.5)")
print_conf_matrix(y_test, log_tuned_pred_03, "Tuned Logistic Regression (0.3)")
print_conf_matrix(y_test, rf_tuned_pred_03, "Tuned Random Forest (0.3)")

# ---------------------------------------------------------
# 4.a.18 Model robustness / stress test:
# Time-based split
# ---------------------------------------------------------
time_df = model_df.dropna(subset=["Review Date"]).copy()
time_df = time_df.sort_values("Review Date").reset_index(drop=True)

split_index = int(len(time_df) * 0.80)

train_time = time_df.iloc[:split_index].copy()
test_time = time_df.iloc[split_index:].copy()

X_train_time = train_time[model_features]
y_train_time = train_time[target]

X_test_time = test_time[model_features]
y_test_time = test_time[target]

print_section("4.a.18 TIME-BASED TRAIN-TEST SPLIT")
print("Training rows:", len(train_time))
print("Testing rows:", len(test_time))
print("Training period:", train_time["Review Date"].min(), "to", train_time["Review Date"].max())
print("Testing period :", test_time["Review Date"].min(), "to", test_time["Review Date"].max())

log_time_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_time_model.fit(X_train_time, y_train_time)

log_time_pred = log_time_model.predict(X_test_time)
log_time_prob = log_time_model.predict_proba(X_test_time)[:, 1]

log_time_results = evaluate_model(
    y_test_time,
    log_time_pred,
    log_time_prob,
    "Logistic Regression (Time-Based Split)",
)

log_time_threshold_03 = evaluate_at_threshold(
    y_test_time,
    log_time_prob,
    0.30,
    "Logistic Regression (Time-Based Split)",
)

log_time_pred_03 = (log_time_prob >= 0.30).astype(int)

rf_time_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
)

rf_time_model.fit(X_train_time, y_train_time)

rf_time_pred = rf_time_model.predict(X_test_time)
rf_time_prob = rf_time_model.predict_proba(X_test_time)[:, 1]

rf_time_results = evaluate_model(
    y_test_time,
    rf_time_pred,
    rf_time_prob,
    "Random Forest (Time-Based Split)",
)

rf_time_threshold_03 = evaluate_at_threshold(
    y_test_time,
    rf_time_prob,
    0.30,
    "Random Forest (Time-Based Split)",
)

rf_time_pred_03 = (rf_time_prob >= 0.30).astype(int)

lasso_time_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_STATE,
                C=lasso_grid.best_params_["model__C"],
            ),
        ),
    ]
)

lasso_time_model.fit(X_train_time, y_train_time)

lasso_time_pred = lasso_time_model.predict(X_test_time)
lasso_time_prob = lasso_time_model.predict_proba(X_test_time)[:, 1]

lasso_time_results = evaluate_model(
    y_test_time,
    lasso_time_pred,
    lasso_time_prob,
    "LASSO Logistic Regression (Time-Based Split)",
)

lasso_time_threshold_03 = evaluate_at_threshold(
    y_test_time,
    lasso_time_prob,
    0.30,
    "LASSO Logistic Regression (Time-Based Split)",
)

lasso_time_pred_03 = (lasso_time_prob >= 0.30).astype(int)

time_based_results = format_results_table(
    results_list=[
        log_time_results,
        rf_time_results,
        lasso_time_results,
        log_time_threshold_03,
        rf_time_threshold_03,
        lasso_time_threshold_03,
    ],
    labels=[
        "Logistic Regression Time (0.5)",
        "Random Forest Time (0.5)",
        "LASSO Logistic Regression Time (0.5)",
        "Logistic Regression Time (0.3)",
        "Random Forest Time (0.3)",
        "LASSO Logistic Regression Time (0.3)",
    ],
    title="4.a.18 TIME-BASED MODEL COMPARISON",
)

robustness_comparison = format_results_table(
    results_list=[
        log_results,
        rf_results,
        lasso_results,
        log_time_results,
        rf_time_results,
        lasso_time_results,
    ],
    labels=[
        "Logistic Regression (Random Split)",
        "Random Forest (Random Split)",
        "LASSO Logistic Regression (Random Split)",
        "Logistic Regression (Time-Based Split)",
        "Random Forest (Time-Based Split)",
        "LASSO Logistic Regression (Time-Based Split)",
    ],
    title="4.a.18 RANDOM SPLIT VERSUS TIME-BASED SPLIT",
)

print_conf_matrix(y_test_time, log_time_pred, "Logistic Regression Time (0.5)")
print_conf_matrix(y_test_time, rf_time_pred, "Random Forest Time (0.5)")
print_conf_matrix(y_test_time, lasso_time_pred, "LASSO Logistic Regression Time (0.5)")
print_conf_matrix(y_test_time, log_time_pred_03, "Logistic Regression Time (0.3)")
print_conf_matrix(y_test_time, rf_time_pred_03, "Random Forest Time (0.3)")
print_conf_matrix(y_test_time, lasso_time_pred_03, "LASSO Logistic Regression Time (0.3)")

# ROC coordinates
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

# AUC values
auc_log = roc_auc_score(y_test, log_prob)
auc_rf = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(8.8, 5.6))

plt.plot(
    fpr_log,
    tpr_log,
    linewidth=2.2,
    color="#2F5D8A",
    label=f"Logistic Regression (ROC-AUC = {auc_log:.3f})",
    zorder=3,
)

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2.2,
    color="#F08D39",
    label=f"Random Forest (ROC-AUC = {auc_rf:.3f})",
    zorder=3,
)

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)

plt.grid(axis="y", color="#e9e9e9", linewidth=0.7)
plt.gca().grid(False, axis="x")
plt.gca().set_axisbelow(True)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#bdbdbd")
ax.spines["bottom"].set_color("#bdbdbd")

plt.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="#cfcfcf",
    framealpha=0.96,
    fontsize=10,
    title="Models",
    title_fontsize=10,
    borderpad=0.8,
    labelspacing=0.6,
)

finalise_plot("ROC Curve Comparison for Predicting Customer Recommendation")

# ---------------------------------------------------------
# 4.b.1 Imports and text-modelling helpers
# ---------------------------------------------------------
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
from transformers import pipeline

VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05
ROBERTA_POS_THRESHOLD = 0.60
ROBERTA_NEG_THRESHOLD = 0.60

# ---------------------------------------------------------
# 4.b.2 VADER sentiment extraction
# ---------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment_score(text):
    """
    Compute the VADER compound score.
    """
    if pd.isna(text):
        return np.nan
    return analyzer.polarity_scores(str(text))["compound"]


def vader_sentiment_label(score):
    """
    Convert continuous VADER score into categorical label.
    """
    if pd.isna(score):
        return np.nan
    if score >= VADER_POS_THRESHOLD:
        return "Positive"
    if score <= VADER_NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


text_df = text_df.copy()
text_df["vader_score"] = text_df["Review"].apply(vader_sentiment_score)
text_df["vader_label"] = text_df["vader_score"].apply(vader_sentiment_label)
text_df["Review_ID"] = text_df.index

print_section("4.b.2 VADER SENTIMENT DISTRIBUTION")
print(text_df["vader_label"].value_counts(dropna=False))

# ---------------------------------------------------------
# 4.b.3 RoBERTa sentiment extraction
# ---------------------------------------------------------
import os
import math

device = -1
print("RoBERTa device: CPU")

roberta_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

def map_roberta_output(label: str, score: float) -> str:
    if pd.isna(label):
        return np.nan

    label_clean = str(label).strip().lower()

    if label_clean in ["negative", "label_0"]:
        return "Negative"
    if label_clean in ["neutral", "label_1"]:
        return "Neutral"
    if label_clean in ["positive", "label_2"]:
        return "Positive"

    if "neg" in label_clean:
        return "Negative"
    if "neu" in label_clean or "mixed" in label_clean:
        return "Neutral"
    if "pos" in label_clean:
        return "Positive"

    return "Neutral"


def run_roberta_in_batches_with_checkpoint(
    df_input,
    text_col="Review",
    id_col="Review_ID",
    batch_size=8,
    max_length=96,
    progress_every=20,
    checkpoint_every=50,
    checkpoint_path="roberta_full_checkpoint.csv",
    final_output_path="text_df_roberta_full_scored.csv"
):
    """
    Run RoBERTa on the full dataset with checkpointing and resume support.
    """

    df_work = df_input[[id_col, text_col]].copy()
    df_work[text_col] = df_work[text_col].fillna("").astype(str)

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        scored_df = pd.read_csv(checkpoint_path)
        done_ids = set(scored_df[id_col].tolist())
        df_remaining = df_work[~df_work[id_col].isin(done_ids)].copy()

        print(f"Checkpoint found: {checkpoint_path}")
        print(f"Already scored: {len(scored_df):,}")
        print(f"Remaining: {len(df_remaining):,}")
    else:
        scored_df = pd.DataFrame(columns=[
            id_col,
            "roberta_raw_label",
            "roberta_confidence",
            "roberta_label"
        ])
        df_remaining = df_work.copy()
        print("No checkpoint found. Starting full run from scratch.")

    texts = df_remaining[text_col].tolist()
    ids = df_remaining[id_col].tolist()

    total_batches = math.ceil(len(texts) / batch_size)
    batch_counter = 0

    new_rows = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        outputs = roberta_pipe(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )

        for rid, out in zip(batch_ids, outputs):
            raw_label = out["label"]
            conf = out["score"]
            mapped_label = map_roberta_output(raw_label, conf)

            new_rows.append({
                id_col: rid,
                "roberta_raw_label": raw_label,
                "roberta_confidence": conf,
                "roberta_label": mapped_label
            })

        batch_counter += 1

        if batch_counter % progress_every == 0 or batch_counter == total_batches:
            print(f"Processed batch {batch_counter}/{total_batches}")

        if batch_counter % checkpoint_every == 0:
            temp_df = pd.concat([scored_df, pd.DataFrame(new_rows)], ignore_index=True)
            temp_df.to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved at batch {batch_counter}: {checkpoint_path}")

    # Final save
    final_scored = pd.concat([scored_df, pd.DataFrame(new_rows)], ignore_index=True)
    final_scored.to_csv(checkpoint_path, index=False)
    final_scored.to_csv(final_output_path, index=False)

    print(f"Final checkpoint saved: {checkpoint_path}")
    print(f"Final output saved: {final_output_path}")

    return final_scored


# Ensure Review_ID exists
if "Review_ID" not in text_df.columns:
    text_df = text_df.copy()
    text_df["Review_ID"] = text_df.index

# Run on FULL DATASET
roberta_full_results = run_roberta_in_batches_with_checkpoint(
    df_input=text_df,
    text_col="Review",
    id_col="Review_ID",
    batch_size=8,
    max_length=96,
    progress_every=20,
    checkpoint_every=50,
    checkpoint_path="roberta_full_checkpoint.csv",
    final_output_path="text_df_roberta_full_scored.csv"
)

# Merge results back into text_df
text_df = text_df.drop(
    columns=["roberta_raw_label", "roberta_confidence", "roberta_label", "roberta_score_signed"],
    errors="ignore"
).merge(
    roberta_full_results,
    on="Review_ID",
    how="left"
)

# Signed score for later visualisation alignment with ratings
roberta_score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
text_df["roberta_score_signed"] = (
    text_df["roberta_label"].map(roberta_score_map) * text_df["roberta_confidence"]
)

print_section("4.b.3 ROBERTA SENTIMENT DISTRIBUTION (FULL DATASET)")
print(text_df["roberta_label"].value_counts(dropna=False))

# To Upload text_df_roberta_full_scored.csv (takes 2 hours to run - Can Skip Re-run with file Upload)
from google.colab import files
uploaded = files.upload()

roberta_full = pd.read_csv("text_df_roberta_full_scored.csv")
print(roberta_full.shape)
roberta_full.head()

roberta_full = pd.read_csv("text_df_roberta_full_scored.csv")

roberta_cols = [
    "Review_ID",
    "roberta_raw_label",
    "roberta_confidence",
    "roberta_label"
]

text_df = text_df.drop(
    columns=[
        "roberta_raw_label",
        "roberta_confidence",
        "roberta_label",
        "roberta_score_signed"
    ],
    errors="ignore"
).merge(
    roberta_full[roberta_cols],
    on="Review_ID",
    how="left"
)

roberta_score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}

text_df["roberta_score_signed"] = (
    text_df["roberta_label"].map(roberta_score_map)
    * text_df["roberta_confidence"]
)

text_df_compare = text_df.dropna(subset=["roberta_label"]).copy()

print(text_df_compare.shape)
print(text_df_compare["roberta_label"].value_counts(dropna=False))

# ---------------------------------------------------------
# 4.b.4 Sentiment label alignment with recommendation
# ---------------------------------------------------------
def sentiment_recommendation_summary(df_input, label_col, title):
    summary = (
        df_input.groupby(label_col, as_index=False)
        .agg(
            Total_Reviews=("Recommended", "size"),
            Recommended_Count=("Recommended", "sum")
        )
    )

    summary["Not_Recommended_Count"] = (
        summary["Total_Reviews"] - summary["Recommended_Count"]
    )
    summary["Recommendation_Rate"] = (
        summary["Recommended_Count"] / summary["Total_Reviews"] * 100
    ).round(1)

    summary = summary.rename(columns={label_col: "Sentiment_Label"})
    show_table(title, summary, index=False)
    return summary


vader_rec_summary = sentiment_recommendation_summary(
    text_df_compare,
    "vader_label",
    "4.b.4 VADER SENTIMENT LABELS VERSUS RECOMMENDATION"
)

roberta_rec_summary = sentiment_recommendation_summary(
    text_df_compare,
    "roberta_label",
    "4.b.4 ROBERTA SENTIMENT LABELS VERSUS RECOMMENDATION"
)

# ---------------------------------------------------------
# 4.b.5 Candidate ambiguous and contradictory cases
# ---------------------------------------------------------
candidate_cases = text_df_compare[
    (
        ((text_df_compare["vader_label"] == "Positive") & (text_df_compare["Recommended"] == 0)) |
        ((text_df_compare["vader_label"] == "Negative") & (text_df_compare["Recommended"] == 1)) |
        ((text_df_compare["roberta_label"] == "Positive") & (text_df_compare["Recommended"] == 0)) |
        ((text_df_compare["roberta_label"] == "Negative") & (text_df_compare["Recommended"] == 1)) |
        (text_df_compare["vader_label"] != text_df_compare["roberta_label"]) |
        ((text_df_compare["positive_cue_count"] >= 2) & (text_df_compare["negative_cue_count"] >= 2))
    )
].copy()

show_table(
    "4.b.5 CANDIDATE AMBIGUOUS OR MISCLASSIFIED REVIEWS",
    candidate_cases.head(25),
    index=False
)

# ---------------------------------------------------------
# 4.b.6 Automated review groups for manual inspection
# ---------------------------------------------------------
base_cols = [
    "Review_ID",
    "Overall_Rating",
    "Recommended",
    "vader_score",
    "vader_label",
    "roberta_confidence",
    "roberta_label",
    "positive_cue_count",
    "negative_cue_count",
    "Review"
]

df_groups = text_df_compare.copy()

df_groups["vader_wrong"] = (
    ((df_groups["vader_label"] == "Positive") & (df_groups["Recommended"] == 0)) |
    ((df_groups["vader_label"] == "Negative") & (df_groups["Recommended"] == 1))
)

df_groups["roberta_wrong"] = (
    ((df_groups["roberta_label"] == "Positive") & (df_groups["Recommended"] == 0)) |
    ((df_groups["roberta_label"] == "Negative") & (df_groups["Recommended"] == 1))
)

df_groups["method_disagree"] = df_groups["vader_label"] != df_groups["roberta_label"]

df_groups["mixed_cues"] = (
    (df_groups["positive_cue_count"] >= 2) &
    (df_groups["negative_cue_count"] >= 2)
)

group1_reviews = df_groups[
    (df_groups["vader_wrong"] == True) &
    (df_groups["roberta_wrong"] == False)
][base_cols].copy()
group1_reviews["Review_Group"] = "VADER misclassification"

group2_reviews = df_groups[
    (df_groups["roberta_wrong"] == True) &
    (df_groups["vader_wrong"] == False)
][base_cols].copy()
group2_reviews["Review_Group"] = "RoBERTa misclassification"

group3_reviews = df_groups[
    (
        ((df_groups["vader_wrong"] == True) & (df_groups["roberta_wrong"] == True)) |
        (df_groups["method_disagree"] == True) |
        (df_groups["mixed_cues"] == True)
    )
][base_cols].copy()
group3_reviews["Review_Group"] = "Ambiguity / mixed-language cases"

group3_reviews = group3_reviews[
    ~group3_reviews["Review_ID"].isin(group1_reviews["Review_ID"]) &
    ~group3_reviews["Review_ID"].isin(group2_reviews["Review_ID"])
]

cols_order = [
    "Review_Group",
    "Review_ID",
    "Overall_Rating",
    "Recommended",
    "vader_score",
    "vader_label",
    "roberta_confidence",
    "roberta_label",
    "positive_cue_count",
    "negative_cue_count",
    "Review"
]

group1_reviews = group1_reviews[cols_order].copy()
group2_reviews = group2_reviews[cols_order].copy()
group3_reviews = group3_reviews[cols_order].copy()

print_section("4.b.6 MUTUALLY EXCLUSIVE GROUP COUNTS")
print("VADER misclassification:", len(group1_reviews))
print("RoBERTa misclassification:", len(group2_reviews))
print("Ambiguity / mixed-language cases:", len(group3_reviews))

# ---------------------------------------------------------
# 4.b.7 Final manually reviewed cases
# ---------------------------------------------------------
group1_reviews["selection_strength"] = (
    group1_reviews["vader_score"].abs() +
    group1_reviews["negative_cue_count"] +
    group1_reviews["positive_cue_count"]
)

group2_reviews["selection_strength"] = (
    group2_reviews["roberta_confidence"] +
    group2_reviews["negative_cue_count"] +
    group2_reviews["positive_cue_count"]
)

group3_reviews["selection_strength"] = (
    group3_reviews["vader_score"].abs() +
    group3_reviews["roberta_confidence"] +
    group3_reviews["negative_cue_count"] +
    group3_reviews["positive_cue_count"]
)

group1_final = (
    group1_reviews
    .sort_values(["selection_strength", "Review_ID"], ascending=[False, True])
    .head(5)
    .drop(columns="selection_strength")
)

group2_final = (
    group2_reviews
    .sort_values(["selection_strength", "Review_ID"], ascending=[False, True])
    .head(5)
    .drop(columns="selection_strength")
)

group3_final = (
    group3_reviews
    .sort_values(["selection_strength", "Review_ID"], ascending=[False, True])
    .head(5)
    .drop(columns="selection_strength")
)

manual_review_df = pd.concat(
    [group1_final, group2_final, group3_final],
    axis=0
).reset_index(drop=True)

show_table(
    "4.b.7 FINAL REVIEWS SELECTED FOR MANUAL ANALYSIS",
    manual_review_df,
    index=False
)

comparison_table = manual_review_df.copy()

# ---------------------------------------------------------
# 4.b.8 Manual review notes
# ---------------------------------------------------------
def manual_review_reason(row):
    reasons = []

    if row["Review_Group"] == "VADER misclassification":
        reasons.append("VADER misclassifies sentiment relative to recommendation")

    if row["Review_Group"] == "RoBERTa misclassification":
        reasons.append("RoBERTa misclassifies sentiment despite stronger contextual modelling")

    if row["Review_Group"] == "Ambiguity / mixed-language cases":
        reasons.append("Review contains ambiguity, mixed sentiment, or conflicting evaluative cues")

    if row["positive_cue_count"] >= 2 and row["negative_cue_count"] >= 2:
        reasons.append("Contains both positive and negative cues")

    return "; ".join(reasons)

manual_review_df["manual_review_reason"] = manual_review_df.apply(
    manual_review_reason,
    axis=1
)

show_table(
    "4.b.8 MANUAL REVIEW NOTES",
    manual_review_df[[
        "Review_Group",
        "Review_ID",
        "Recommended",
        "vader_label",
        "roberta_label",
        "manual_review_reason"
    ]],
    index=False
)

# ---------------------------------------------------------
# 4.b.9 Prepare text visualisation data
# ---------------------------------------------------------
text_vis = text_df_compare.copy()

text_vis["Recommended_label"] = text_vis["Recommended"].map({
    1: "Recommended",
    0: "Not Recommended"
})

text_vis["vader_label"] = pd.Categorical(
    text_vis["vader_label"],
    categories=["Negative", "Neutral", "Positive"],
    ordered=True
)

text_vis["roberta_label"] = pd.Categorical(
    text_vis["roberta_label"],
    categories=["Negative", "Neutral", "Positive"],
    ordered=True
)

text_vis["contradiction_flag"] = np.where(
    (
        ((text_vis["vader_label"] == "Positive") & (text_vis["Recommended"] == 0)) |
        ((text_vis["vader_label"] == "Negative") & (text_vis["Recommended"] == 1)) |
        ((text_vis["roberta_label"] == "Positive") & (text_vis["Recommended"] == 0)) |
        ((text_vis["roberta_label"] == "Negative") & (text_vis["Recommended"] == 1))
    ),
    "Contradictory",
    "Aligned"
)

text_vis["is_mixed_sentiment"] = np.where(
    (text_vis["positive_cue_count"] > 0) & (text_vis["negative_cue_count"] > 0),
    "Mixed",
    "Not Mixed"
)

comparison_plot = comparison_table.copy()
label_to_num = {"Negative": -1, "Neutral": 0, "Positive": 1}

comparison_plot["vader_num"] = comparison_plot["vader_label"].map(label_to_num)
comparison_plot["roberta_num"] = comparison_plot["roberta_label"].map(label_to_num)
comparison_plot["disagreement_magnitude"] = (
    comparison_plot["vader_num"] - comparison_plot["roberta_num"]
).abs()

# ---------------------------------------------------------
# 4.b.10 Plot 1: Overall rating versus VADER score
# ---------------------------------------------------------
plot_df = text_vis.dropna(subset=["Overall_Rating", "vader_score", "Recommended"]).copy()
contradiction_df = plot_df[plot_df["contradiction_flag"] == "Contradictory"].copy()

fig, ax = plt.subplots(figsize=(11, 8))

ax.scatter(
    plot_df["Overall_Rating"],
    plot_df["vader_score"],
    s=10,
    alpha=0.08,
    color="grey",
    edgecolor="none"
)

ax.scatter(
    contradiction_df["Overall_Rating"],
    contradiction_df["vader_score"],
    s=35,
    alpha=0.70,
    c=np.where(contradiction_df["Recommended"] == 0, "crimson", "darkorange"),
    edgecolor="black",
    linewidth=0.3
)

ax.axvline(5, linestyle="--", color="black", linewidth=1)
ax.axhline(0, linestyle="--", color="black", linewidth=1)

ax.set_xlabel("Overall Rating")
ax.set_ylabel("VADER Compound Sentiment Score")
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-1.05, 1.05)

legend_elements = [
    Line2D([0], [0], marker="o", color="w",
           label="Positive sentiment / Not recommended",
           markerfacecolor="crimson", markeredgecolor="black", markersize=8),
    Line2D([0], [0], marker="o", color="w",
           label="Negative sentiment / Recommended",
           markerfacecolor="darkorange", markeredgecolor="black", markersize=8)
]

ax.legend(handles=legend_elements, frameon=True, loc="lower right")
finalise_plot("Alignment Between Overall Rating and VADER Sentiment")

# ---------------------------------------------------------
# 4.b.11 Plot 2: VADER distribution by recommendation
# ---------------------------------------------------------
plot_df = text_vis.dropna(subset=["vader_score", "Recommended_label"]).copy()

plt.figure(figsize=(10, 7))

sns.violinplot(
    data=plot_df,
    x="Recommended_label",
    y="vader_score",
    palette=["#5E7AC4", "#F08D39"],
    inner=None,
    cut=0,
    linewidth=1.2
)

sns.boxplot(
    data=plot_df,
    x="Recommended_label",
    y="vader_score",
    width=0.22,
    showcaps=True,
    showfliers=False,
    boxprops={"facecolor": "white", "zorder": 3},
    whiskerprops={"linewidth": 1.2},
    medianprops={"color": "black", "linewidth": 1.4}
)

sns.stripplot(
    data=plot_df.sample(min(2000, len(plot_df)), random_state=RANDOM_STATE),
    x="Recommended_label",
    y="vader_score",
    color="black",
    alpha=0.10,
    size=2.3,
    jitter=0.22
)

plt.axhline(0, linestyle="--", color="black", linewidth=1)
plt.xlabel("")
plt.ylabel("VADER Compound Sentiment Score")
plt.ylim(-1.05, 1.05)
finalise_plot("Distribution of VADER Sentiment Scores by Recommendation Outcome")

# ---------------------------------------------------------
# 4.b.14 Plot 5: RoBERTa distribution by recommendation
# ---------------------------------------------------------
plot_df_r = text_vis.dropna(subset=["roberta_label", "Recommended"]).copy()

plot_df_r["Recommended_label"] = plot_df_r["Recommended"].map({0: "Not Recommended", 1: "Recommended"})
plot_df_r["roberta_label"] = plot_df_r["roberta_label"].astype(str).str.strip().str.title()

print(plot_df_r[["Recommended", "Recommended_label", "roberta_label"]].head())
print(plot_df_r["Recommended_label"].value_counts(dropna=False))
print(plot_df_r["roberta_label"].value_counts(dropna=False))

# Build percentage table safely
counts = pd.crosstab(plot_df_r["Recommended_label"], plot_df_r["roberta_label"])
dist_df = counts.div(counts.sum(axis=1), axis=0).mul(100)

# force column order if present
for col in ["Negative", "Neutral", "Positive"]:
    if col not in dist_df.columns:
        dist_df[col] = 0

dist_df = dist_df[["Negative", "Neutral", "Positive"]]
dist_df = dist_df.reindex(["Not Recommended", "Recommended"])

print(dist_df)

sns.set_theme(style="ticks")

ax = dist_df.plot(
    kind="bar",
    figsize=(8.5, 5.5),
    edgecolor="black",
    width=0.72
)

ax.grid(False)
sns.despine()

plt.xlabel("")
plt.ylabel("Percentage of Reviews")
plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.legend(
    title="RoBERTa Sentiment",
    fontsize=10,
    title_fontsize=11,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=True
)

finalise_plot("RoBERTa Sentiment Distribution by Recommendation Outcome")

plot_df_v = text_vis.dropna(subset=["vader_label", "Recommended"]).copy()

plot_df_v["Recommended_label"] = plot_df_v["Recommended"].map({0: "Not Recommended", 1: "Recommended"})
plot_df_v["vader_label"] = plot_df_v["vader_label"].astype(str).str.strip().str.title()

print(plot_df_v[["Recommended", "Recommended_label", "vader_label"]].head())
print(plot_df_v["Recommended_label"].value_counts(dropna=False))
print(plot_df_v["vader_label"].value_counts(dropna=False))

# Build percentage table safely
counts = pd.crosstab(plot_df_v["Recommended_label"], plot_df_v["vader_label"])
dist_df_v = counts.div(counts.sum(axis=1), axis=0).mul(100)

# Force column order if present
for col in ["Negative", "Neutral", "Positive"]:
    if col not in dist_df_v.columns:
        dist_df_v[col] = 0

dist_df_v = dist_df_v[["Negative", "Neutral", "Positive"]]
dist_df_v = dist_df_v.reindex(["Not Recommended", "Recommended"])

print(dist_df_v)

sns.set_theme(style="ticks")

ax = dist_df_v.plot(
    kind="bar",
    figsize=(8.5, 5.5),
    edgecolor="black",
    width=0.72,
    color=["#2F5D8A", "#F39C12", "#3A9D5D"]
)

ax.grid(False)
sns.despine()

plt.xlabel("")
plt.ylabel("Percentage of Reviews")
plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.legend(
    title="VADER Sentiment",
    fontsize=10,
    title_fontsize=11,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=True
)

finalise_plot("VADER Sentiment Distribution by Recommendation Outcome")
plt.show()

# ---------------------------------------------------------
# 4.b.12 Plot 3: Recommendation heatmap by rating and VADER sentiment
# ---------------------------------------------------------
heat_df = text_vis.dropna(subset=["Overall_Rating", "vader_score", "Recommended"]).copy()

heat_df["Rating_Band"] = pd.cut(
    heat_df["Overall_Rating"],
    bins=[0, 2, 4, 6, 8, 10],
    labels=["1-2", "3-4", "5-6", "7-8", "9-10"],
    include_lowest=True
)

heat_df["Sentiment_Band"] = pd.cut(
    heat_df["vader_score"],
    bins=[-1.01, -0.3, 0.3, 1.01],
    labels=["Negative", "Neutral/Mixed", "Positive"],
    include_lowest=True
)

heatmap_df = (
    heat_df.groupby(["Sentiment_Band", "Rating_Band"], observed=False)["Recommended"]
    .mean()
    .mul(100)
    .unstack()
)

count_df = (
    heat_df.groupby(["Sentiment_Band", "Rating_Band"], observed=False)["Recommended"]
    .size()
    .unstack()
)

heatmap_df = heatmap_df.astype(float)
count_df = count_df.fillna(0).astype(int)

annot = heatmap_df.round(1).astype(str) + "%\n(n=" + count_df.astype(str) + ")"

plt.figure(figsize=(11, 6))
sns.heatmap(
    heatmap_df,
    annot=annot,
    fmt="",
    cmap="RdYlGn",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Recommendation rate (%)"}
)

plt.xlabel("Overall Rating Band")
plt.ylabel("VADER Sentiment Band")
finalise_plot("Recommendation Rate Across Rating and Sentiment Bands")

# ---------------------------------------------------------
# 4.b.13 Plot 4: Recommendation heatmap by rating and RoBERTa sentiment
# ---------------------------------------------------------
heat_df_r = text_df.dropna(subset=["Overall_Rating", "roberta_label", "Recommended"]).copy()

heat_df_r["Rating_Band"] = pd.cut(
    heat_df_r["Overall_Rating"],
    bins=[0, 2, 4, 6, 8, 10],
    labels=["1-2", "3-4", "5-6", "7-8", "9-10"],
    include_lowest=True
)

heat_df_r["Sentiment_Band"] = pd.Categorical(
    heat_df_r["roberta_label"],
    categories=["Negative", "Neutral", "Positive"],
    ordered=True
)

heatmap_df_r = (
    heat_df_r.groupby(["Sentiment_Band", "Rating_Band"], observed=False)["Recommended"]
    .mean()
    .mul(100)
    .unstack()
)

count_df_r = (
    heat_df_r.groupby(["Sentiment_Band", "Rating_Band"], observed=False)["Recommended"]
    .size()
    .unstack()
)

heatmap_df_r = heatmap_df_r.astype(float)
count_df_r = count_df_r.fillna(0).astype(int)

annot_r = heatmap_df_r.round(1).astype(str) + "%\n(n=" + count_df_r.astype(str) + ")"

plt.figure(figsize=(11, 6))
sns.heatmap(
    heatmap_df_r,
    annot=annot_r,
    fmt="",
    cmap="RdYlGn",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Recommendation rate (%)"}
)

plt.xlabel("Overall Rating Band")
plt.ylabel("RoBERTa Sentiment Band")
finalise_plot("Recommendation Rate Across Rating and RoBERTa Sentiment Bands")

# ---------------------------------------------------------
# 4.b.13 Plot 4: VADER versus RoBERTa agreement heatmap
# ---------------------------------------------------------
agreement_df = text_vis[["vader_label", "roberta_label"]].dropna().copy()
order = ["Negative", "Neutral", "Positive"]

counts = pd.crosstab(
    agreement_df["vader_label"],
    agreement_df["roberta_label"]
).reindex(index=order, columns=order, fill_value=0)

row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
annot = counts.astype(str) + "\n(" + row_pct.round(1).astype(str) + "%)"

plt.figure(figsize=(8.5, 5.8))
sns.heatmap(
    counts,
    annot=annot,
    fmt="",
    cmap="Purples",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Number of Reviews"},
    annot_kws={"size": 10}
)

plt.xlabel("RoBERTa Sentiment Label")
plt.ylabel("VADER Sentiment Label")
finalise_plot("Agreement Between VADER and RoBERTa Sentiment Labels")

# ---------------------------------------------------------
# 4.b.14 Plot 5: Dumbbell plot for manually reviewed cases
# ---------------------------------------------------------
case_df = comparison_plot.dropna(subset=["vader_num", "roberta_num"]).copy()

case_df["gap"] = (case_df["vader_num"] - case_df["roberta_num"]).abs()
case_df = case_df.sort_values(
    by=["gap", "Review_ID"],
    ascending=[False, True]
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13.2, 7.0))
y_positions = np.arange(len(case_df))

# connection lines
for i, row in case_df.iterrows():
    if row["gap"] > 0:
        strong = row["gap"] >= 2
        ax.plot(
            [row["vader_num"], row["roberta_num"]],
            [i, i],
            color="#2f2f2f" if strong else "#c7c7c7",
            linewidth=1.5 if strong else 0.9,
            alpha=0.60 if strong else 0.45,
            zorder=1
        )

# agreement vs disagreement
agree_df = case_df[case_df["gap"] == 0].copy()
disagree_df = case_df[case_df["gap"] > 0].copy()

agree_y = agree_df.index.to_numpy()
disagree_y = disagree_df.index.to_numpy()

# disagreement points
vader_scatter = ax.scatter(
    disagree_df["vader_num"],
    disagree_y,
    s=88,
    color="#4C72B0",
    label="VADER",
    zorder=3
)

roberta_scatter = ax.scatter(
    disagree_df["roberta_num"],
    disagree_y,
    s=88,
    marker="s",
    color="#DD8452",
    label="RoBERTa",
    zorder=3
)

# agreement points
agreement_scatter = ax.scatter(
    agree_df["vader_num"],
    agree_y,
    s=98,
    marker="D",
    color="#3a3a3a",
    label="Agreement",
    zorder=4
)

# y-axis labels
labels = [f"ID {rid}" for rid in case_df["Review_ID"]]
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=10)

# x-axis labels
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(["Negative", "Neutral", "Positive"], fontsize=11)
ax.set_xlabel("Sentiment Classification", fontsize=13)
ax.set_ylabel("")

# grid styling
ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)

# spine cleanup
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# legend outside
handles = [agreement_scatter, vader_scatter, roberta_scatter]
legend_labels = ["Agreement", "VADER", "RoBERTa"]

ax.legend(
    handles,
    legend_labels,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    fontsize=11,
    title_fontsize=12,
    borderaxespad=0.0,
    labelspacing=0.9,
    handletextpad=0.8,
    borderpad=0.8,
    handlelength=1.6,
    markerscale=1.0
)

plt.tight_layout(rect=[0, 0, 0.82, 1])
finalise_plot("Agreement and Disagreement Across Manually Reviewed Cases")

# ---------------------------------------------------------
# 4.b.14 Plot 5: Contradiction rate by sentiment method
# Contradictions defined as:
# - Positive sentiment + Not Recommended
# - Negative sentiment + Recommended
# ---------------------------------------------------------

contradiction_df = text_df.dropna(
    subset=["Recommended", "vader_label", "roberta_label"]
).copy()

contradiction_df["Recommended"] = pd.to_numeric(
    contradiction_df["Recommended"], errors="coerce"
)

contradiction_df["vader_label"] = contradiction_df["vader_label"].astype(str).str.strip().str.title()
contradiction_df["roberta_label"] = contradiction_df["roberta_label"].astype(str).str.strip().str.title()

plot_rows = []

# VADER Positive contradiction:
# among VADER Positive reviews, % that are Not Recommended
temp = contradiction_df[contradiction_df["vader_label"] == "Positive"]
plot_rows.append({
    "Category": "VADER Positive\n→ Not Recommended",
    "Contradiction_Rate": (temp["Recommended"] == 0).mean() * 100,
    "Count": len(temp)
})

# RoBERTa Positive contradiction:
# among RoBERTa Positive reviews, % that are Not Recommended
temp = contradiction_df[contradiction_df["roberta_label"] == "Positive"]
plot_rows.append({
    "Category": "RoBERTa Positive\n→ Not Recommended",
    "Contradiction_Rate": (temp["Recommended"] == 0).mean() * 100,
    "Count": len(temp)
})

# VADER Negative contradiction:
# among VADER Negative reviews, % that are Recommended
temp = contradiction_df[contradiction_df["vader_label"] == "Negative"]
plot_rows.append({
    "Category": "VADER Negative\n→ Recommended",
    "Contradiction_Rate": (temp["Recommended"] == 1).mean() * 100,
    "Count": len(temp)
})

# RoBERTa Negative contradiction:
# among RoBERTa Negative reviews, % that are Recommended
temp = contradiction_df[contradiction_df["roberta_label"] == "Negative"]
plot_rows.append({
    "Category": "RoBERTa Negative\n→ Recommended",
    "Contradiction_Rate": (temp["Recommended"] == 1).mean() * 100,
    "Count": len(temp)
})

contradiction_plot_df = pd.DataFrame(plot_rows)

plt.figure(figsize=(10.5, 6.2))
ax = sns.barplot(
    data=contradiction_plot_df,
    x="Category",
    y="Contradiction_Rate",
    palette=["#3A9D5D", "#2F5D8A", "#3A9D5D", "#2F5D8A"],
    edgecolor="black",
    linewidth=1.0
)

# Annotate bars
for i, row in contradiction_plot_df.iterrows():
    ax.text(
        i,
        row["Contradiction_Rate"] + 0.8,
        f'{row["Contradiction_Rate"]:.1f}%\n(n={row["Count"]:,})',
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xlabel("")
plt.ylabel("Contradiction rate (%)")
plt.ylim(0, contradiction_plot_df["Contradiction_Rate"].max() * 1.18)
plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
finalise_plot("Contradictory Sentiment Classifications by Method")
plt.show()

# ---------------------------------------------------------
# 4.b.15 Plot 6: Recommendation rate by service theme
# Colourful version
# ---------------------------------------------------------

theme_df = text_df.dropna(subset=["Recommended"]).copy()
theme_df["Recommended"] = pd.to_numeric(theme_df["Recommended"], errors="coerce")

theme_cols = [
    "delay_theme",
    "cancellation_theme",
    "refund_theme",
    "staff_theme",
    "seat_theme",
    "food_theme"
]

theme_labels = {
    "delay_theme": "Delay",
    "cancellation_theme": "Cancellation",
    "refund_theme": "Refund",
    "staff_theme": "Staff",
    "seat_theme": "Seat",
    "food_theme": "Food"
}

plot_rows = []

for col in theme_cols:
    temp = theme_df[theme_df[col] == 1].copy()
    plot_rows.append({
        "Theme": theme_labels[col],
        "Recommendation_Rate": temp["Recommended"].mean() * 100 if len(temp) > 0 else 0,
        "Count": len(temp)
    })

theme_plot_df = pd.DataFrame(plot_rows)
theme_plot_df = theme_plot_df.sort_values(
    by="Recommendation_Rate",
    ascending=True
).reset_index(drop=True)

# Colour progression from low-recommendation themes to high-recommendation themes
bar_colors = [
    "#C0392B",  # Refund
    "#E74C3C",  # Cancellation
    "#F39C12",  # Delay
    "#5DA5DA",  # Staff
    "#3E7CB1",  # Seat
    "#2E8B57"   # Food
]

plt.figure(figsize=(11, 6.4))
ax = sns.barplot(
    data=theme_plot_df,
    x="Theme",
    y="Recommendation_Rate",
    palette=bar_colors,
    edgecolor="black",
    linewidth=1.1
)

for i, row in theme_plot_df.iterrows():
    ax.text(
        i,
        row["Recommendation_Rate"] + 1.3,
        f'{row["Recommendation_Rate"]:.1f}%\n(n={row["Count"]:,})',
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xlabel("")
plt.ylabel("Recommendation rate (%)")
plt.ylim(0, 100)
plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
finalise_plot("Recommendation Rate by Service Theme")
plt.show()

# %% =========================================================
# 4.b. PREDICTIVE ANALYTICS (TEXT FEATURES)
# =========================================================

# %% ---------------------------------------------------------
# 4.b.15 Text modelling dataset and train-test split
# ---------------------------------------------------------
text_model_df = text_df[[
    "Review_ID",
    "review_clean_final",
    "Recommended",
    "Overall_Rating",
    "vader_score",
    "positive_cue_count",
    "negative_cue_count"
]].dropna(subset=["review_clean_final", "Recommended"]).copy()

text_model_df["text_length"] = text_model_df["review_clean_final"].str.split().apply(len)

print_section("4.b.15 TEXT MODELLING DATASET")
print("Shape:", text_model_df.shape)
print(text_model_df["Recommended"].value_counts(normalize=True).round(4))

X_text = text_model_df["review_clean_final"]
y_text = text_model_df["Recommended"]

X_train_text, X_test_text, y_train_text, y_test_text, id_train_text, id_test_text = train_test_split(
    X_text,
    y_text,
    text_model_df["Review_ID"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_text
)

print_section("4.b.15 TEXT TRAIN-TEST SPLIT")
print("X_train_text shape:", X_train_text.shape)
print("X_test_text shape:", X_test_text.shape)
print(y_train_text.value_counts(normalize=True).round(4))
print(y_test_text.value_counts(normalize=True).round(4))


# %% ---------------------------------------------------------
# 4.b.16 TF-IDF feature engineering
# ---------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    sublinear_tf=True
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

print_section("4.b.16 TF-IDF FEATURE SPACE")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)
print("Number of text features:", len(feature_names))


# %% ---------------------------------------------------------
# 4.b.17 Logistic regression using TF-IDF
# ---------------------------------------------------------
text_log_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

text_log_model.fit(X_train_tfidf, y_train_text)

text_log_pred = text_log_model.predict(X_test_tfidf)
text_log_prob = text_log_model.predict_proba(X_test_tfidf)[:, 1]

text_log_results = evaluate_model(
    y_test_text,
    text_log_pred,
    text_log_prob,
    "Text Logistic Regression"
)

text_log_cm = print_conf_matrix(
    y_test_text,
    text_log_pred,
    "Text Logistic Regression"
)

# %% ---------------------------------------------------------
# 4.b.18 Random forest using TF-IDF
# ---------------------------------------------------------
text_rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

text_rf_model.fit(X_train_tfidf, y_train_text)

text_rf_pred = text_rf_model.predict(X_test_tfidf)
text_rf_prob = text_rf_model.predict_proba(X_test_tfidf)[:, 1]

text_rf_results = evaluate_model(
    y_test_text,
    text_rf_pred,
    text_rf_prob,
    "Text Random Forest"
)

text_rf_cm = print_conf_matrix(
    y_test_text,
    text_rf_pred,
    "Text Random Forest"
)


# %% ---------------------------------------------------------
# 4.b.19 Additional benchmark: Linear SVM using TF-IDF
# ---------------------------------------------------------
text_svm_base = LinearSVC(
    random_state=RANDOM_STATE
)

text_svm_model = CalibratedClassifierCV(
    estimator=text_svm_base,
    method="sigmoid",
    cv=3
)

text_svm_model.fit(X_train_tfidf, y_train_text)

text_svm_pred = text_svm_model.predict(X_test_tfidf)
text_svm_prob = text_svm_model.predict_proba(X_test_tfidf)[:, 1]

text_svm_results = evaluate_model(
    y_test_text,
    text_svm_pred,
    text_svm_prob,
    "Text Linear SVM"
)

text_svm_cm = print_conf_matrix(
    y_test_text,
    text_svm_pred,
    "Text Linear SVM"
)


# %% ---------------------------------------------------------
# 4.b.20 Text model comparison tables
# ---------------------------------------------------------
text_model_results = pd.DataFrame([
    text_log_results,
    text_rf_results,
    text_svm_results
]).set_index("Model").round(4)

show_table("4.b.20 TEXT MODEL COMPARISON", text_model_results)

part4a_vs_text = pd.DataFrame([
    {
        "Model": "Numerical Logistic Regression",
        "Accuracy": log_results["Accuracy"],
        "Precision": log_results["Precision"],
        "Recall": log_results["Recall"],
        "F1": log_results["F1"],
        "ROC_AUC": log_results["ROC_AUC"]
    },
    {
        "Model": "Numerical Random Forest",
        "Accuracy": rf_results["Accuracy"],
        "Precision": rf_results["Precision"],
        "Recall": rf_results["Recall"],
        "F1": rf_results["F1"],
        "ROC_AUC": rf_results["ROC_AUC"]
    },
    {
        "Model": "Text Logistic Regression",
        "Accuracy": text_log_results["Accuracy"],
        "Precision": text_log_results["Precision"],
        "Recall": text_log_results["Recall"],
        "F1": text_log_results["F1"],
        "ROC_AUC": text_log_results["ROC_AUC"]
    },
    {
        "Model": "Text Random Forest",
        "Accuracy": text_rf_results["Accuracy"],
        "Precision": text_rf_results["Precision"],
        "Recall": text_rf_results["Recall"],
        "F1": text_rf_results["F1"],
        "ROC_AUC": text_rf_results["ROC_AUC"]
    },
    {
        "Model": "Text Linear SVM",
        "Accuracy": text_svm_results["Accuracy"],
        "Precision": text_svm_results["Precision"],
        "Recall": text_svm_results["Recall"],
        "F1": text_svm_results["F1"],
        "ROC_AUC": text_svm_results["ROC_AUC"]
    }
]).set_index("Model").round(4)

show_table("4.b.20 NUMERICAL VERSUS TEXT MODEL COMPARISON", part4a_vs_text)

# %% ---------------------------------------------------------
# 4.b.21 Plot 6: ROC curve comparison for text models
# ---------------------------------------------------------
fpr_log, tpr_log, _ = roc_curve(y_test_text, text_log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test_text, text_rf_prob)
fpr_svm, tpr_svm, _ = roc_curve(y_test_text, text_svm_prob)

plt.figure(figsize=(9.6, 6.6))

plt.plot(
    fpr_log, tpr_log,
    linewidth=2.6,
    label=f"Logistic Regression (AUC = {text_log_results['ROC_AUC']:.3f})"
)

plt.plot(
    fpr_rf, tpr_rf,
    linewidth=2.6,
    label=f"Random Forest (AUC = {text_rf_results['ROC_AUC']:.3f})"
)

plt.plot(
    fpr_svm, tpr_svm,
    linewidth=2.6,
    label=f"Linear SVM (AUC = {text_svm_results['ROC_AUC']:.3f})"
)

plt.plot([0, 1], [0, 1],
         linestyle="--",
         color="black",
         linewidth=1,
         alpha=0.6)

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(
    loc="lower right",
    frameon=True,
    fontsize=11
)

finalise_plot("ROC Curve Comparison for Text Models")


# %% ---------------------------------------------------------
# 4.b.22 Plot 7: Predicted probability distributions
# ---------------------------------------------------------
probability_plot_df = pd.DataFrame({
    "Actual": y_test_text.values,
    "Logistic_Probability": text_log_prob,
    "Random_Forest_Probability": text_rf_prob,
    "Linear_SVM_Probability": text_svm_prob
})

probability_plot_df["Actual_Label"] = probability_plot_df["Actual"].map({
    0: "Not Recommended",
    1: "Recommended"
})

plt.figure(figsize=(9.5, 6.2))

sns.kdeplot(
    data=probability_plot_df,
    x="Linear_SVM_Probability",
    hue="Actual_Label",
    fill=True,
    common_norm=False,
    alpha=0.35,
    linewidth=2,
    bw_adjust=0.9
)

plt.axvline(0.5, linestyle="--", color="black", linewidth=1)
plt.xlim(-0.15, 1.15)

plt.xlabel("Predicted Probability of Recommendation")
plt.ylabel("Density")

ymax = plt.gca().get_ylim()[1]
plt.text(
    0.5, ymax * 0.85,
    "Threshold = 0.5",
    ha="center",
    fontsize=10,
    color="black"
)

legend = plt.gca().get_legend()
if legend is not None:
    legend.set_title("Actual Outcome")
    legend.get_title().set_fontsize(11)
    for text in legend.get_texts():
        text.set_fontsize(10)

finalise_plot("Predicted Probability Distribution for the Linear SVM Text Model")

# %% ---------------------------------------------------------
# 4.b.23 Top predictive words from logistic regression
# ---------------------------------------------------------
logistic_coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": text_log_model.coef_[0]
})

top_positive_terms = (
    logistic_coef_df
    .sort_values("Coefficient", ascending=False)
    .head(20)
    .sort_values("Coefficient", ascending=True)
)

top_negative_terms = (
    logistic_coef_df
    .sort_values("Coefficient", ascending=True)
    .head(20)
    .sort_values("Coefficient", ascending=False)
)

show_table("TOP POSITIVE TEXT FEATURES FROM LOGISTIC REGRESSION", top_positive_terms, index=False)
show_table("TOP NEGATIVE TEXT FEATURES FROM LOGISTIC REGRESSION", top_negative_terms, index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 9))

axes[0].hlines(
    y=top_negative_terms["Feature"],
    xmin=0,
    xmax=top_negative_terms["Coefficient"],
    linewidth=2.5
)
axes[0].plot(
    top_negative_terms["Coefficient"],
    top_negative_terms["Feature"],
    "o",
    markersize=8
)
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_xlabel("Coefficient")
axes[0].set_title("Terms Driving Non-Recommendation", fontsize=13, weight="bold")

axes[1].hlines(
    y=top_positive_terms["Feature"],
    xmin=0,
    xmax=top_positive_terms["Coefficient"],
    linewidth=2.5
)
axes[1].plot(
    top_positive_terms["Coefficient"],
    top_positive_terms["Feature"],
    "o",
    markersize=8
)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_xlabel("Coefficient")
axes[1].set_title("Terms Driving Recommendation", fontsize=13, weight="bold")

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------
# 4.b.24 Distinctive language by class using average TF-IDF
# ---------------------------------------------------------
full_tfidf = tfidf_vectorizer.transform(text_model_df["review_clean_final"])

recommended_mask = (text_model_df["Recommended"].astype(int).to_numpy() == 1)
not_recommended_mask = (text_model_df["Recommended"].astype(int).to_numpy() == 0)

avg_tfidf_rec = np.asarray(full_tfidf[recommended_mask, :].mean(axis=0)).ravel()
avg_tfidf_not = np.asarray(full_tfidf[not_recommended_mask, :].mean(axis=0)).ravel()

class_language_df = pd.DataFrame({
    "Feature": feature_names,
    "Avg_TFIDF_Recommended": avg_tfidf_rec,
    "Avg_TFIDF_Not_Recommended": avg_tfidf_not
})

class_language_df["Lift_Recommended"] = (
    class_language_df["Avg_TFIDF_Recommended"] - class_language_df["Avg_TFIDF_Not_Recommended"]
)

class_language_df["Lift_Not_Recommended"] = (
    class_language_df["Avg_TFIDF_Not_Recommended"] - class_language_df["Avg_TFIDF_Recommended"]
)

recommended_language = class_language_df.sort_values(
    "Lift_Recommended", ascending=False
).head(15).copy()

not_recommended_language = class_language_df.sort_values(
    "Lift_Not_Recommended", ascending=False
).head(15).copy()

show_table(
    "LANGUAGE MOST CHARACTERISTIC OF RECOMMENDED REVIEWS",
    recommended_language,
    index=False
)

show_table(
    "LANGUAGE MOST CHARACTERISTIC OF NOT-RECOMMENDED REVIEWS",
    not_recommended_language,
    index=False
)


# %% ---------------------------------------------------------
# 4.b.25 Plot 8: Language contrast between classes
# ---------------------------------------------------------
recommended_language_plot = recommended_language.head(12).copy()
not_recommended_language_plot = not_recommended_language.head(12).copy()

rec_plot = (
    recommended_language_plot[["Feature", "Lift_Recommended"]]
    .rename(columns={"Lift_Recommended": "Lift"})
    .sort_values("Lift", ascending=True)
    .copy()
)

not_plot = (
    not_recommended_language_plot[["Feature", "Lift_Not_Recommended"]]
    .rename(columns={"Lift_Not_Recommended": "Lift"})
    .sort_values("Lift", ascending=True)
    .copy()
)

max_val = max(rec_plot["Lift"].max(), not_plot["Lift"].max())

fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), sharex=True)

axes[0].hlines(
    y=rec_plot["Feature"],
    xmin=0,
    xmax=rec_plot["Lift"],
    linewidth=2.4,
    color="#DD8452",
    alpha=0.85
)
axes[0].scatter(
    rec_plot["Lift"],
    rec_plot["Feature"],
    s=85,
    color="#DD8452",
    zorder=3
)
axes[0].set_xlim(0, max_val * 1.08)
axes[0].set_xlabel("Average TF-IDF Lift")
axes[0].set_ylabel("")
axes[0].set_title("Key Positive Language Driving Recommendations", fontsize=13, weight="bold")
axes[0].grid(axis="x", color="#e6e6e6", linewidth=0.8)
axes[0].set_axisbelow(True)

axes[1].hlines(
    y=not_plot["Feature"],
    xmin=0,
    xmax=not_plot["Lift"],
    linewidth=2.4,
    color="#4C72B0",
    alpha=0.85
)
axes[1].scatter(
    not_plot["Lift"],
    not_plot["Feature"],
    s=85,
    color="#4C72B0",
    zorder=3
)
axes[1].set_xlim(0, max_val * 1.08)
axes[1].set_xlabel("Average TF-IDF Lift")
axes[1].set_ylabel("")
axes[1].set_title("Key Negative Language Driving Non-Recommendations", fontsize=13, weight="bold")
axes[1].grid(axis="x", color="#e6e6e6", linewidth=0.8)
axes[1].set_axisbelow(True)

plt.suptitle(
    "Distinctive Language Patterns in Recommended and Not-Recommended Reviews",
    fontsize=15,
    weight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% ---------------------------------------------------------
# 4.b.26 Misclassified reviews for qualitative follow-up
# ---------------------------------------------------------
if text_log_results["ROC_AUC"] >= text_rf_results["ROC_AUC"]:
    best_text_model_name = "Text Logistic Regression"
    best_pred = text_log_pred
    best_prob = text_log_prob
else:
    best_text_model_name = "Text Random Forest"
    best_pred = text_rf_pred
    best_prob = text_rf_prob

misclassified_text_df = pd.DataFrame({
    "Review_ID": id_test_text.values,
    "Actual": y_test_text.values,
    "Predicted": best_pred,
    "Predicted_Probability": best_prob
})

misclassified_text_df["Correct"] = np.where(
    misclassified_text_df["Actual"] == misclassified_text_df["Predicted"],
    "Yes",
    "No"
)

misclassified_text_df["Prediction_Confidence"] = np.where(
    misclassified_text_df["Predicted"] == 1,
    misclassified_text_df["Predicted_Probability"],
    1 - misclassified_text_df["Predicted_Probability"]
)

misclassified_text_df = misclassified_text_df.merge(
    text_df[[
        "Review_ID",
        "Review",
        "Overall_Rating",
        "vader_score",
        "positive_cue_count",
        "negative_cue_count"
    ]],
    on="Review_ID",
    how="left"
)

top_misclassified_reviews = misclassified_text_df[
    misclassified_text_df["Correct"] == "No"
].sort_values("Prediction_Confidence", ascending=False).head(10)

show_table(
    f"TOP 10 MISCLASSIFIED REVIEWS FROM {best_text_model_name}",
    top_misclassified_reviews,
    index=False
)

# %% =========================================================
# 4.c. PREDICTIVE ANALYTICS COMBINING NUMERICAL AND TEXT FEATURES
# =========================================================

# %% ---------------------------------------------------------
# 4.c.1 Proposed feature integration strategies
# ---------------------------------------------------------
integration_strategies = pd.DataFrame([
    {
        "Strategy": "Early Fusion",
        "Description": "Concatenate scaled numerical features and TF-IDF text features into one feature matrix, then train a single classifier.",
        "Implemented": "Yes"
    },
    {
        "Strategy": "Late Fusion",
        "Description": "Train numerical and text models separately, then combine predicted probabilities using a simple average or weighted average.",
        "Implemented": "No"
    }
])

show_table("4.c.1 PROPOSED FEATURE INTEGRATION STRATEGIES", integration_strategies, index=False)


# %% ---------------------------------------------------------
# 4.c.2 Combined modelling dataset
# ---------------------------------------------------------
combined_df = df_clean[[
    "Review",
    "Review Date",
    "Recommended"
] + model_features].copy()

combined_df["review_clean_basic"] = combined_df["Review"].apply(basic_clean_text)
combined_df["review_tokens"] = combined_df["review_clean_basic"].apply(tokenize_and_lemmatize)
combined_df["review_clean_final"] = combined_df["review_tokens"].apply(lambda x: " ".join(x))

combined_df = combined_df.dropna(subset=["review_clean_final", "Recommended"]).copy()
combined_df["Review_ID"] = combined_df.index

print_section("4.c.2 COMBINED MODELLING DATASET")
print("Shape:", combined_df.shape)
print(combined_df["Recommended"].value_counts(normalize=True).round(4))


# %% ---------------------------------------------------------
# 4.c.3 Train-test split for combined features
# ---------------------------------------------------------
X_num_combined = combined_df[model_features].copy()
X_text_combined = combined_df["review_clean_final"].copy()
y_combined = combined_df["Recommended"].copy()

(
    X_num_train,
    X_num_test,
    X_text_train,
    X_text_test,
    y_train_combined,
    y_test_combined,
    id_train_combined,
    id_test_combined
) = train_test_split(
    X_num_combined,
    X_text_combined,
    y_combined,
    combined_df["Review_ID"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_combined
)

print_section("4.c.3 COMBINED TRAIN-TEST SPLIT")
print("Numerical train shape:", X_num_train.shape)
print("Numerical test shape :", X_num_test.shape)
print("Text train shape     :", X_text_train.shape)
print("Text test shape      :", X_text_test.shape)
print(y_train_combined.value_counts(normalize=True).round(4))
print(y_test_combined.value_counts(normalize=True).round(4))


# %% ---------------------------------------------------------
# 4.c.4 Early fusion feature matrix
# ---------------------------------------------------------
combined_scaler = StandardScaler()
X_num_train_scaled = combined_scaler.fit_transform(X_num_train)
X_num_test_scaled = combined_scaler.transform(X_num_test)

combined_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    sublinear_tf=True
)

X_text_train_tfidf = combined_tfidf.fit_transform(X_text_train)
X_text_test_tfidf = combined_tfidf.transform(X_text_test)

X_num_train_sparse = csr_matrix(X_num_train_scaled)
X_num_test_sparse = csr_matrix(X_num_test_scaled)

X_train_fused = hstack([X_num_train_sparse, X_text_train_tfidf], format="csr")
X_test_fused = hstack([X_num_test_sparse, X_text_test_tfidf], format="csr")

combined_feature_names = np.concatenate([
    np.array([f"NUM__{col}" for col in model_features]),
    np.array([f"TEXT__{feat}" for feat in combined_tfidf.get_feature_names_out()])
])

print_section("4.c.4 EARLY FUSION FEATURE SPACE")
print("Fused training matrix shape:", X_train_fused.shape)
print("Fused testing matrix shape :", X_test_fused.shape)
print("Numerical feature count    :", len(model_features))
print("Text feature count         :", len(combined_tfidf.get_feature_names_out()))
print("Total fused feature count  :", len(combined_feature_names))

# %% ---------------------------------------------------------
# 4.c.5 Combined logistic regression
# ---------------------------------------------------------
combined_log_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

combined_log_model.fit(X_train_fused, y_train_combined)

combined_log_pred = combined_log_model.predict(X_test_fused)
combined_log_prob = combined_log_model.predict_proba(X_test_fused)[:, 1]

combined_log_results = evaluate_model(
    y_test_combined,
    combined_log_pred,
    combined_log_prob,
    "Combined Logistic Regression"
)

combined_log_cm = print_conf_matrix(
    y_test_combined,
    combined_log_pred,
    "Combined Logistic Regression"
)


# %% ---------------------------------------------------------
# 4.c.6 Optional benchmark: combined linear SVM
# ---------------------------------------------------------
combined_svm_base = LinearSVC(
    random_state=RANDOM_STATE
)

combined_svm_model = CalibratedClassifierCV(
    estimator=combined_svm_base,
    method="sigmoid",
    cv=3
)

combined_svm_model.fit(X_train_fused, y_train_combined)

combined_svm_pred = combined_svm_model.predict(X_test_fused)
combined_svm_prob = combined_svm_model.predict_proba(X_test_fused)[:, 1]

combined_svm_results = evaluate_model(
    y_test_combined,
    combined_svm_pred,
    combined_svm_prob,
    "Combined Linear SVM"
)

combined_svm_cm = print_conf_matrix(
    y_test_combined,
    combined_svm_pred,
    "Combined Linear SVM"
)


# %% ---------------------------------------------------------
# 4.c.7 Performance impact of feature integration
# ---------------------------------------------------------
if text_log_results["ROC_AUC"] >= text_rf_results["ROC_AUC"]:
    best_core_text_name = "Text Logistic Regression"
    best_core_text_results = text_log_results
else:
    best_core_text_name = "Text Random Forest"
    best_core_text_results = text_rf_results

combined_comparison = pd.DataFrame([
    {
        "Model": "Best Numerical Model (Random Forest)",
        "Accuracy": rf_results["Accuracy"],
        "Precision": rf_results["Precision"],
        "Recall": rf_results["Recall"],
        "F1": rf_results["F1"],
        "ROC_AUC": rf_results["ROC_AUC"]
    },
    {
        "Model": f"Best Core Text Model ({best_core_text_name.replace('Text ', '')})",
        "Accuracy": best_core_text_results["Accuracy"],
        "Precision": best_core_text_results["Precision"],
        "Recall": best_core_text_results["Recall"],
        "F1": best_core_text_results["F1"],
        "ROC_AUC": best_core_text_results["ROC_AUC"]
    },
    {
        "Model": "Text Linear SVM (Benchmark)",
        "Accuracy": text_svm_results["Accuracy"],
        "Precision": text_svm_results["Precision"],
        "Recall": text_svm_results["Recall"],
        "F1": text_svm_results["F1"],
        "ROC_AUC": text_svm_results["ROC_AUC"]
    },
    {
        "Model": "Combined Logistic Regression",
        "Accuracy": combined_log_results["Accuracy"],
        "Precision": combined_log_results["Precision"],
        "Recall": combined_log_results["Recall"],
        "F1": combined_log_results["F1"],
        "ROC_AUC": combined_log_results["ROC_AUC"]
    },
    {
        "Model": "Combined Linear SVM",
        "Accuracy": combined_svm_results["Accuracy"],
        "Precision": combined_svm_results["Precision"],
        "Recall": combined_svm_results["Recall"],
        "F1": combined_svm_results["F1"],
        "ROC_AUC": combined_svm_results["ROC_AUC"]
    }
]).set_index("Model").round(4)

show_table("4.c.7 PERFORMANCE IMPACT OF FEATURE INTEGRATION", combined_comparison)

combined_delta = pd.DataFrame([
    {
        "Comparison": "Combined Logistic vs Best Numerical",
        "Accuracy_Delta": combined_log_results["Accuracy"] - rf_results["Accuracy"],
        "Recall_Delta": combined_log_results["Recall"] - rf_results["Recall"],
        "F1_Delta": combined_log_results["F1"] - rf_results["F1"],
        "ROC_AUC_Delta": combined_log_results["ROC_AUC"] - rf_results["ROC_AUC"]
    },
    {
        "Comparison": "Combined Logistic vs Best Core Text Model",
        "Accuracy_Delta": combined_log_results["Accuracy"] - best_core_text_results["Accuracy"],
        "Recall_Delta": combined_log_results["Recall"] - best_core_text_results["Recall"],
        "F1_Delta": combined_log_results["F1"] - best_core_text_results["F1"],
        "ROC_AUC_Delta": combined_log_results["ROC_AUC"] - best_core_text_results["ROC_AUC"]
    },
    {
        "Comparison": "Combined Logistic vs Text SVM Benchmark",
        "Accuracy_Delta": combined_log_results["Accuracy"] - text_svm_results["Accuracy"],
        "Recall_Delta": combined_log_results["Recall"] - text_svm_results["Recall"],
        "F1_Delta": combined_log_results["F1"] - text_svm_results["F1"],
        "ROC_AUC_Delta": combined_log_results["ROC_AUC"] - text_svm_results["ROC_AUC"]
    },
    {
        "Comparison": "Combined SVM vs Best Numerical",
        "Accuracy_Delta": combined_svm_results["Accuracy"] - rf_results["Accuracy"],
        "Recall_Delta": combined_svm_results["Recall"] - rf_results["Recall"],
        "F1_Delta": combined_svm_results["F1"] - rf_results["F1"],
        "ROC_AUC_Delta": combined_svm_results["ROC_AUC"] - rf_results["ROC_AUC"]
    },
    {
        "Comparison": "Combined SVM vs Best Core Text Model",
        "Accuracy_Delta": combined_svm_results["Accuracy"] - best_core_text_results["Accuracy"],
        "Recall_Delta": combined_svm_results["Recall"] - best_core_text_results["Recall"],
        "F1_Delta": combined_svm_results["F1"] - best_core_text_results["F1"],
        "ROC_AUC_Delta": combined_svm_results["ROC_AUC"] - best_core_text_results["ROC_AUC"]
    },
    {
        "Comparison": "Combined SVM vs Text SVM Benchmark",
        "Accuracy_Delta": combined_svm_results["Accuracy"] - text_svm_results["Accuracy"],
        "Recall_Delta": combined_svm_results["Recall"] - text_svm_results["Recall"],
        "F1_Delta": combined_svm_results["F1"] - text_svm_results["F1"],
        "ROC_AUC_Delta": combined_svm_results["ROC_AUC"] - text_svm_results["ROC_AUC"]
    }
]).set_index("Comparison").round(4)

show_table("4.c.7 CHANGE IN PERFORMANCE AFTER INTEGRATION", combined_delta)

# %% ---------------------------------------------------------
# 4.c.6B Optional benchmark: combined Random Forest
# ---------------------------------------------------------
combined_rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

combined_rf_model.fit(X_train_fused, y_train_combined)

combined_rf_pred = combined_rf_model.predict(X_test_fused)
combined_rf_prob = combined_rf_model.predict_proba(X_test_fused)[:, 1]

combined_rf_results = evaluate_model(
    y_test_combined,
    combined_rf_pred,
    combined_rf_prob,
    "Combined Random Forest"
)

combined_rf_cm = print_conf_matrix(
    y_test_combined,
    combined_rf_pred,
    "Combined Random Forest"
)

# %% ---------------------------------------------------------
# 4.c.8 Dominant predictors in the combined logistic model
# ---------------------------------------------------------
combined_coef_df = pd.DataFrame({
    "Feature": combined_feature_names,
    "Coefficient": combined_log_model.coef_[0]
})

combined_coef_df["Feature_Type"] = np.where(
    combined_coef_df["Feature"].str.startswith("NUM__"),
    "Numerical",
    "Text"
)

combined_coef_df["Abs_Coefficient"] = combined_coef_df["Coefficient"].abs()

top_combined_positive = (
    combined_coef_df
    .sort_values("Coefficient", ascending=False)
    .head(20)
    .copy()
)

top_combined_negative = (
    combined_coef_df
    .sort_values("Coefficient", ascending=True)
    .head(20)
    .copy()
)

show_table(
    "4.c.8 TOP POSITIVE PREDICTORS IN THE COMBINED LOGISTIC MODEL",
    top_combined_positive,
    index=False
)
show_table(
    "4.c.8 TOP NEGATIVE PREDICTORS IN THE COMBINED LOGISTIC MODEL",
    top_combined_negative,
    index=False
)

dominance_summary = (
    combined_coef_df
    .sort_values("Abs_Coefficient", ascending=False)
    .head(30)
    .groupby("Feature_Type")
    .size()
    .reset_index(name="Count_in_Top_30")
)

show_table(
    "4.c.8 FEATURE DOMINANCE IN TOP 30 COMBINED LOGISTIC PREDICTORS",
    dominance_summary,
    index=False
)


# %% ---------------------------------------------------------
# 4.c.9 Plot 1: Performance comparison across modelling approaches
# ---------------------------------------------------------
best_core_text_label = best_core_text_name.replace("Text ", "")

combined_candidates = [
    ("Combined Logistic Regression", combined_log_results),
    ("Combined Linear SVM", combined_svm_results),
    ("Combined Random Forest", combined_rf_results),
]

best_combined_name, best_combined_results = max(
    combined_candidates,
    key=lambda x: x[1]["ROC_AUC"]
)

combined_comparison_plot = pd.DataFrame([
    {
        "Model": "Best Numerical Model\n(Random Forest)",
        "Accuracy": rf_results["Accuracy"],
        "Recall": rf_results["Recall"],
        "F1": rf_results["F1"],
        "ROC_AUC": rf_results["ROC_AUC"]
    },
    {
        "Model": f"Best Core Text Model\n({best_core_text_label})",
        "Accuracy": best_core_text_results["Accuracy"],
        "Recall": best_core_text_results["Recall"],
        "F1": best_core_text_results["F1"],
        "ROC_AUC": best_core_text_results["ROC_AUC"]
    },
    {
        "Model": "Text Linear SVM\n(Benchmark)",
        "Accuracy": text_svm_results["Accuracy"],
        "Recall": text_svm_results["Recall"],
        "F1": text_svm_results["F1"],
        "ROC_AUC": text_svm_results["ROC_AUC"]
    },
    {
        "Model": "Combined Logistic Regression",
        "Accuracy": combined_log_results["Accuracy"],
        "Recall": combined_log_results["Recall"],
        "F1": combined_log_results["F1"],
        "ROC_AUC": combined_log_results["ROC_AUC"]
    },
    {
        "Model": "Combined Linear SVM",
        "Accuracy": combined_svm_results["Accuracy"],
        "Recall": combined_svm_results["Recall"],
        "F1": combined_svm_results["F1"],
        "ROC_AUC": combined_svm_results["ROC_AUC"]
    },
    {
        "Model": "Combined Random Forest",
        "Accuracy": combined_rf_results["Accuracy"],
        "Recall": combined_rf_results["Recall"],
        "F1": combined_rf_results["F1"],
        "ROC_AUC": combined_rf_results["ROC_AUC"]
    }
])

model_order = [
    "Best Numerical Model\n(Random Forest)",
    f"Best Core Text Model\n({best_core_text_label})",
    "Text Linear SVM\n(Benchmark)",
    "Combined Logistic Regression",
    "Combined Linear SVM",
    "Combined Random Forest"
]

plot_metrics_df = combined_comparison_plot.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Recall", "F1", "ROC_AUC"],
    var_name="Metric",
    value_name="Score"
)

plot_metrics_df["Model"] = pd.Categorical(
    plot_metrics_df["Model"],
    categories=model_order,
    ordered=True
)

plt.figure(figsize=(14, 7.4))
ax = sns.barplot(
    data=plot_metrics_df,
    x="Metric",
    y="Score",
    hue="Model",
    hue_order=model_order
)

plt.ylim(0.88, 1.00)
plt.ylabel("Performance Score")
plt.xlabel("")
plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
plt.gca().set_axisbelow(True)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8)

plt.legend(
    title="Model",
    frameon=True,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

finalise_plot("Performance Comparison Across Numerical, Text, and Combined Models")


# %% ---------------------------------------------------------
# 4.c.10 Plot 2: Top predictors in the combined logistic model
# ---------------------------------------------------------
top_n_each_side = 16

top_combined_positive = (
    combined_coef_df
    .sort_values("Coefficient", ascending=False)
    .head(top_n_each_side)
    .copy()
)

top_combined_negative = (
    combined_coef_df
    .sort_values("Coefficient", ascending=True)
    .head(top_n_each_side)
    .copy()
)

top_combined_plot = pd.concat(
    [top_combined_negative, top_combined_positive],
    axis=0
).copy()

top_combined_plot["Feature_Type"] = np.where(
    top_combined_plot["Feature"].str.startswith("NUM__"),
    "Numerical",
    "Text"
)

drop_features = {
    "TEXT__another",
    "TEXT__told",
    "TEXT__okay",
    "TEXT__sof",
    "TEXT__overall",
    "TEXT__flew",
    "TEXT__front",
    "TEXT__new",
}

top_combined_plot = top_combined_plot[
    ~top_combined_plot["Feature"].isin(drop_features)
].copy()

neg_plot = (
    top_combined_plot[top_combined_plot["Coefficient"] < 0]
    .sort_values("Coefficient", ascending=True)
    .head(12)
    .copy()
)

pos_plot = (
    top_combined_plot[top_combined_plot["Coefficient"] > 0]
    .sort_values("Coefficient", ascending=False)
    .head(12)
    .copy()
)

top_combined_plot = pd.concat([neg_plot, pos_plot], axis=0).copy()

display_name_map = {
    "NUM__Value For Money": "Value for Money",
    "TEXT__flight delayed": "flight delayed",
    "TEXT__friendly": "friendly",
    "TEXT__good": "good",
    "TEXT__excellent": "excellent",
    "TEXT__great": "great",
    "TEXT__pleasant": "pleasant",
    "TEXT__nice": "nice",
    "TEXT__thank": "thank",
    "TEXT__comfortable": "comfortable",
    "TEXT__efficient": "efficient",
    "TEXT__drink": "drink",
    "TEXT__clean": "clean",
    "TEXT__best": "best",
    "TEXT__helpful": "helpful",
    "TEXT__impressed": "impressed",
    "TEXT__flight time": "flight time",
    "TEXT__not": "not",
    "TEXT__no": "no",
    "TEXT__delayed": "delayed",
    "TEXT__hour": "hour",
    "TEXT__worst": "worst",
    "TEXT__never": "never",
    "TEXT__terrible": "terrible",
    "TEXT__poor": "poor",
    "TEXT__disappointing": "disappointing",
    "TEXT__would not": "would not",
    "TEXT__disappointed": "disappointed",
    "TEXT__cannot": "cannot",
    "TEXT__customer": "customer",
    "TEXT__changed": "changed",
}

top_combined_plot["Display_Feature"] = top_combined_plot["Feature"].map(display_name_map)
top_combined_plot["Display_Feature"] = top_combined_plot["Display_Feature"].fillna(
    top_combined_plot["Feature"]
    .str.replace("NUM__", "", regex=False)
    .str.replace("TEXT__", "", regex=False)
)

top_combined_plot = (
    top_combined_plot
    .sort_values("Coefficient", ascending=True)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(11.8, 8.0))

text_color = "#4C72B0"
num_color = "#DD8452"

ax.axvline(0, color="black", linewidth=1.0, alpha=0.75, zorder=1)

for _, row in top_combined_plot.iterrows():
    line_color = text_color if row["Feature_Type"] == "Text" else num_color
    ax.hlines(
        y=row["Display_Feature"],
        xmin=0,
        xmax=row["Coefficient"],
        color=line_color,
        linewidth=2.0,
        alpha=0.78,
        zorder=2
    )

text_features_plot = top_combined_plot[top_combined_plot["Feature_Type"] == "Text"]
text_scatter = ax.scatter(
    text_features_plot["Coefficient"],
    text_features_plot["Display_Feature"],
    s=95,
    color=text_color,
    edgecolor="white",
    linewidth=0.8,
    label="Text",
    zorder=3
)

num_features_plot = top_combined_plot[top_combined_plot["Feature_Type"] == "Numerical"]
num_scatter = ax.scatter(
    num_features_plot["Coefficient"],
    num_features_plot["Display_Feature"],
    s=180,
    marker="X",
    color=num_color,
    edgecolor="black",
    linewidth=1.8,
    label="Numerical",
    zorder=4
)

limit = np.ceil(np.max(np.abs(top_combined_plot["Coefficient"])) * 10) / 10
ax.set_xlim(-limit - 0.25, limit + 0.25)

ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
ax.set_ylabel("")
ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.margins(y=0.02)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=11)

ax.set_title(
    "Top Predictors in the Combined Logistic Model (Early Fusion)",
    fontsize=16,
    fontweight="bold",
    pad=22
)

ax.text(
    0.18, 1.015,
    "← Drives Non-Recommendation",
    transform=ax.transAxes,
    fontsize=10,
    ha="center",
    va="bottom"
)

ax.text(
    0.82, 1.015,
    "Drives Recommendation →",
    transform=ax.transAxes,
    fontsize=10,
    ha="center",
    va="bottom"
)

legend = ax.legend(
    handles=[text_scatter, num_scatter],
    title="Feature Type",
    frameon=True,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.12)
)
legend.get_title().set_fontsize(11)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %% ---------------------------------------------------------
# 4.c.11 Brief failure analysis for the combined model
# ---------------------------------------------------------
best_combined_name = "Combined Logistic Regression"
best_combined_pred = combined_log_pred
best_combined_prob = combined_log_prob

combined_failure_df = pd.DataFrame({
    "Review_ID": id_test_combined.values,
    "Actual": y_test_combined.astype(int).to_numpy(),
    "Predicted": pd.Series(best_combined_pred).astype(int).to_numpy(),
    "Predicted_Probability": best_combined_prob
})

combined_failure_df["Correct"] = np.where(
    combined_failure_df["Actual"] == combined_failure_df["Predicted"],
    "Yes",
    "No"
)

combined_failure_df["Prediction_Confidence"] = np.where(
    combined_failure_df["Predicted"] == 1,
    combined_failure_df["Predicted_Probability"],
    1 - combined_failure_df["Predicted_Probability"]
)

failure_context = combined_df[[
    "Review_ID",
    "Review",
    "review_clean_final",
    "Overall_Rating",
    "Value For Money",
    "Seat Comfort",
    "Cabin Staff Service",
    "Ground Service"
]].copy()

combined_failure_df = combined_failure_df.merge(
    failure_context,
    on="Review_ID",
    how="left"
)

top_combined_failures = combined_failure_df[
    combined_failure_df["Correct"] == "No"
].sort_values("Prediction_Confidence", ascending=False).head(10)

show_table(
    f"4.c.11 TOP 10 MISCLASSIFIED CASES FROM {best_combined_name}",
    top_combined_failures,
    index=False
)

# %% =========================================================
# APPENDIX A. ADDITIONAL EXPLORATORY ANALYSIS
# =========================================================

# %% ---------------------------------------------------------
# A.1 Helper copies and safe standardisation
# ---------------------------------------------------------
appendix_df = df_clean.copy()

for col in ["Verified", "Aircraft", "Type Of Traveller", "Seat Type", "Route"]:
    if col in appendix_df.columns:
        appendix_df[col] = appendix_df[col].astype("string").str.strip()

appendix_df["Recommended_Label"] = appendix_df["Recommended"].map({1: "Yes", 0: "No"})


# %% ---------------------------------------------------------
# A.2 VERIFIED: counts, recommendation rate, and rating profile
# ---------------------------------------------------------
if "Verified" in appendix_df.columns:
    verified_df = appendix_df.dropna(subset=["Verified", "Recommended"]).copy()

    verified_df["Verified_Clean"] = (
        verified_df["Verified"]
        .str.lower()
        .replace({
            "true": "Verified",
            "false": "Not Verified",
            "yes": "Verified",
            "no": "Not Verified"
        })
    )

    verified_df = verified_df[
        verified_df["Verified_Clean"].isin(["Verified", "Not Verified"])
    ].copy()

    verified_summary = (
        verified_df.groupby("Verified_Clean", as_index=False)
        .agg(
            Review_Count=("Recommended", "size"),
            Recommendation_Rate=("Recommended", "mean"),
            Mean_Overall_Rating=("Overall_Rating", "mean"),
            Mean_Value_For_Money=("Value For Money", "mean")
        )
    )

    verified_summary["Recommendation_Rate"] = (verified_summary["Recommendation_Rate"] * 100).round(1)
    verified_summary["Mean_Overall_Rating"] = verified_summary["Mean_Overall_Rating"].round(2)
    verified_summary["Mean_Value_For_Money"] = verified_summary["Mean_Value_For_Money"].round(2)

    show_table("APPENDIX A.2 VERIFIED REVIEW SUMMARY", verified_summary, index=False)

    verified_crosstab = pd.crosstab(
        verified_df["Verified_Clean"],
        verified_df["Recommended_Label"],
        margins=True
    )
    show_table("APPENDIX A.2 VERIFIED BY RECOMMENDATION CROSSTAB", verified_crosstab)

    plt.figure(figsize=(8.4, 5.4))

    order = ["Verified", "Not Verified"]
    plot_df = verified_summary.set_index("Verified_Clean").reindex(order).reset_index()

    bars = plt.bar(
        plot_df["Verified_Clean"],
        plot_df["Recommendation_Rate"],
        color=["#2F5D8A", "#B7C7D9"],
        edgecolor="black",
        linewidth=0.8,
        width=0.5
    )

    for bar, rate, count in zip(bars, plot_df["Recommendation_Rate"], plot_df["Review_Count"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.2,
            f"{rate:.1f}%\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylim(0, min(100, plot_df["Recommendation_Rate"].max() + 12))
    plt.xlabel("")
    plt.ylabel("Recommendation Rate (%)")
    plt.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    plt.gca().grid(False, axis="x")
    plt.gca().set_axisbelow(True)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c0c0c0")
    ax.spines["bottom"].set_color("#c0c0c0")

    finalise_plot("Appendix A.2 Recommendation Rate by Verification Status")


# %% ---------------------------------------------------------
# A.3 AIRCRAFT BRAND / FAMILY ANALYSIS
# ---------------------------------------------------------
def classify_aircraft_brand(text: str) -> str:
    """Map aircraft descriptions into broad aircraft-brand families."""
    if pd.isna(text):
        return np.nan

    t = str(text).lower().strip()

    if t == "" or t == "nan":
        return np.nan

    if any(x in t for x in [
        "boeing", "b737", "b-737", "737", "b738", "b739", "b747", "747",
        "b757", "757", "b767", "767", "b777", "777", "b787", "787",
        "max 8", "max8", "max 9", "max9", "dreamliner"
    ]):
        return "Boeing"

    if any(x in t for x in [
        "airbus", "a220", "a300", "a310", "a318", "a319", "a320", "a321",
        "a330", "a340", "a350", "a380"
    ]):
        return "Airbus"

    if any(x in t for x in ["embraer", "emb", "e170", "e175", "e190", "e195", "erj"]):
        return "Embraer"

    if any(x in t for x in ["bombardier", "crj", "dash 8", "dash8", "q400", "q300", "q200", "cseries"]):
        return "Bombardier / De Havilland"

    if any(x in t for x in ["atr 42", "atr42", "atr 72", "atr72", "atr"]):
        return "ATR"

    if any(x in t for x in ["md-", "md ", "md80", "md81", "md82", "md83", "md87", "md88", "md90", "dc-"]):
        return "McDonnell Douglas"

    if any(x in t for x in ["fokker", "saab", "tupolev", "ilyushin", "antonov", "comac", "yak", "sukhoi", "superjet"]):
        return "Other Manufacturer"

    return "Other / Unclear"


if "Aircraft" in appendix_df.columns:
    aircraft_df = appendix_df.dropna(subset=["Aircraft", "Recommended"]).copy()
    aircraft_df["Aircraft_Brand"] = aircraft_df["Aircraft"].apply(classify_aircraft_brand)

    aircraft_summary = (
        aircraft_df.dropna(subset=["Aircraft_Brand"])
        .groupby("Aircraft_Brand", as_index=False)
        .agg(
            Review_Count=("Recommended", "size"),
            Recommendation_Rate=("Recommended", "mean"),
            Mean_Overall_Rating=("Overall_Rating", "mean"),
            Mean_Value_For_Money=("Value For Money", "mean")
        )
        .sort_values("Review_Count", ascending=False)
    )

    aircraft_summary["Recommendation_Rate"] = (aircraft_summary["Recommendation_Rate"] * 100).round(1)
    aircraft_summary["Mean_Overall_Rating"] = aircraft_summary["Mean_Overall_Rating"].round(2)
    aircraft_summary["Mean_Value_For_Money"] = aircraft_summary["Mean_Value_For_Money"].round(2)

    show_table("APPENDIX A.3 AIRCRAFT BRAND SUMMARY", aircraft_summary, index=False)

    aircraft_plot_df = aircraft_summary[aircraft_summary["Review_Count"] >= 100].copy()
    aircraft_plot_df = aircraft_plot_df.sort_values("Recommendation_Rate", ascending=True)

    if len(aircraft_plot_df) > 0:
        plt.figure(figsize=(9.6, 6.2))

        bars = plt.barh(
            aircraft_plot_df["Aircraft_Brand"],
            aircraft_plot_df["Recommendation_Rate"],
            color="#6F8FAF",
            edgecolor="black",
            linewidth=0.8,
            height=0.72
        )

        for bar, rate, count in zip(bars, aircraft_plot_df["Recommendation_Rate"], aircraft_plot_df["Review_Count"]):
            plt.text(
                rate + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  (n={count:,})",
                va="center",
                ha="left",
                fontsize=9
            )

        plt.xlim(0, min(100, aircraft_plot_df["Recommendation_Rate"].max() + 12))
        plt.xlabel("Recommendation Rate (%)")
        plt.ylabel("")
        plt.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        plt.gca().grid(False, axis="y")
        plt.gca().set_axisbelow(True)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c0c0c0")
        ax.spines["bottom"].set_color("#c0c0c0")

        finalise_plot("Appendix A.3 Recommendation Rate by Aircraft Brand Family")


# %% ---------------------------------------------------------
# A.4 TYPE OF TRAVELLER ANALYSIS
# ---------------------------------------------------------
if "Type Of Traveller" in appendix_df.columns:
    traveller_df = appendix_df.dropna(subset=["Type Of Traveller", "Recommended"]).copy()

    traveller_summary = (
        traveller_df.groupby("Type Of Traveller", as_index=False)
        .agg(
            Review_Count=("Recommended", "size"),
            Recommendation_Rate=("Recommended", "mean"),
            Mean_Overall_Rating=("Overall_Rating", "mean"),
            Mean_Value_For_Money=("Value For Money", "mean")
        )
        .sort_values("Recommendation_Rate", ascending=False)
    )

    traveller_summary["Recommendation_Rate"] = (traveller_summary["Recommendation_Rate"] * 100).round(1)
    traveller_summary["Mean_Overall_Rating"] = traveller_summary["Mean_Overall_Rating"].round(2)
    traveller_summary["Mean_Value_For_Money"] = traveller_summary["Mean_Value_For_Money"].round(2)

    show_table("APPENDIX A.4 TYPE OF TRAVELLER SUMMARY", traveller_summary, index=False)

    traveller_plot_df = traveller_summary[traveller_summary["Review_Count"] >= 100].copy()
    traveller_plot_df = traveller_plot_df.sort_values("Recommendation_Rate", ascending=True)

    plt.figure(figsize=(10.2, 6.4))

    bars = plt.barh(
        traveller_plot_df["Type Of Traveller"],
        traveller_plot_df["Recommendation_Rate"],
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.8,
        height=0.72
    )

    for bar, rate, count in zip(bars, traveller_plot_df["Recommendation_Rate"], traveller_plot_df["Review_Count"]):
        plt.text(
            rate + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%  (n={count:,})",
            va="center",
            ha="left",
            fontsize=9
        )

    plt.xlim(0, min(100, traveller_plot_df["Recommendation_Rate"].max() + 12))
    plt.xlabel("Recommendation Rate (%)")
    plt.ylabel("")
    plt.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    plt.gca().grid(False, axis="y")
    plt.gca().set_axisbelow(True)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c0c0c0")
    ax.spines["bottom"].set_color("#c0c0c0")

    finalise_plot("Appendix A.4 Recommendation Rate by Type of Traveller")


# %% ---------------------------------------------------------
# A.5 SEAT TYPE ANALYSIS (appendix version)
# ---------------------------------------------------------
if "Seat Type" in appendix_df.columns:
    appendix_seat_df = appendix_df.dropna(subset=["Seat Type", "Recommended"]).copy()
    appendix_seat_df["Seat Type"] = appendix_seat_df["Seat Type"].astype(str).str.strip()

    seat_order = ["First Class", "Business Class", "Premium Economy", "Economy Class"]

    appendix_seat_summary = (
        appendix_seat_df.groupby("Seat Type", as_index=False)
        .agg(
            Review_Count=("Recommended", "size"),
            Recommendation_Rate=("Recommended", "mean"),
            Mean_Overall_Rating=("Overall_Rating", "mean"),
            Mean_Value_For_Money=("Value For Money", "mean")
        )
    )

    appendix_seat_summary = (
        appendix_seat_summary[appendix_seat_summary["Seat Type"].isin(seat_order)]
        .assign(
            Seat_Type_Order=lambda x: pd.Categorical(
                x["Seat Type"],
                categories=seat_order,
                ordered=True
            )
        )
        .sort_values("Seat_Type_Order")
        .drop(columns="Seat_Type_Order")
    )

    appendix_seat_summary["Recommendation_Rate"] = (appendix_seat_summary["Recommendation_Rate"] * 100).round(1)
    appendix_seat_summary["Mean_Overall_Rating"] = appendix_seat_summary["Mean_Overall_Rating"].round(2)
    appendix_seat_summary["Mean_Value_For_Money"] = appendix_seat_summary["Mean_Value_For_Money"].round(2)

    show_table("APPENDIX A.5 SEAT TYPE SUMMARY", appendix_seat_summary, index=False)


# %% ---------------------------------------------------------
# A.6 ROUTE ANALYSIS: top routes only
# ---------------------------------------------------------
if "Route" in appendix_df.columns:
    route_df = appendix_df.dropna(subset=["Route", "Recommended"]).copy()
    route_df["Route"] = route_df["Route"].astype(str).str.strip()

    route_summary = (
        route_df.groupby("Route", as_index=False)
        .agg(
            Review_Count=("Recommended", "size"),
            Recommendation_Rate=("Recommended", "mean"),
            Mean_Overall_Rating=("Overall_Rating", "mean"),
            Mean_Value_For_Money=("Value For Money", "mean")
        )
    )

    route_summary["Recommendation_Rate"] = (route_summary["Recommendation_Rate"] * 100).round(1)
    route_summary["Mean_Overall_Rating"] = route_summary["Mean_Overall_Rating"].round(2)
    route_summary["Mean_Value_For_Money"] = route_summary["Mean_Value_For_Money"].round(2)

    top_routes = (
        route_summary.sort_values("Review_Count", ascending=False)
        .head(15)
        .copy()
    )

    show_table("APPENDIX A.6 TOP 15 ROUTES BY REVIEW COUNT", top_routes, index=False)

    route_plot_df = top_routes.sort_values("Recommendation_Rate", ascending=True).copy()

    plt.figure(figsize=(11.5, 7.6))

    bars = plt.barh(
        route_plot_df["Route"],
        route_plot_df["Recommendation_Rate"],
        color="#8DAA91",
        edgecolor="black",
        linewidth=0.8,
        height=0.72
    )

    for bar, rate, count in zip(bars, route_plot_df["Recommendation_Rate"], route_plot_df["Review_Count"]):
        plt.text(
            rate + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%  (n={count:,})",
            va="center",
            ha="left",
            fontsize=8.5
        )

    plt.xlim(0, min(100, route_plot_df["Recommendation_Rate"].max() + 12))
    plt.xlabel("Recommendation Rate (%)")
    plt.ylabel("")
    plt.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    plt.gca().grid(False, axis="y")
    plt.gca().set_axisbelow(True)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c0c0c0")
    ax.spines["bottom"].set_color("#c0c0c0")

    finalise_plot("Appendix A.6 Recommendation Rate Across the 15 Most Frequent Routes")
