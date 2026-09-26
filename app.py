import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aspect_sentiment import analyze_review
from src.sentiment import SentimentAnalyzer


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AutoSense-NLP",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .sentiment-positive {
            color: #16803c;
            font-weight: bold;
        }

        .sentiment-negative {
            color: #c62828;
            font-weight: bold;
        }

        .sentiment-neutral {
            color: #555;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    return SentimentAnalyzer()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🚗 AutoSense-NLP</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-Based Automotive Review and Customer Sentiment Analytics
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Single Review Analysis",
        "Batch CSV Analysis",
        "Analytics",
        "Vehicle Comparison",
    ],
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

with st.spinner("Loading sentiment model..."):
    analyzer = load_sentiment_model()


# =========================================================
# SINGLE REVIEW ANALYSIS
# =========================================================

if page == "Single Review Analysis":

    st.header("Single Review Analysis")

    st.write(
        "Enter an automotive review to detect automotive "
        "aspects and determine sentiment for each aspect."
    )

    review = st.text_area(
        "Enter automotive review",
        height=150,
        placeholder=(
            "Example: The battery range is excellent "
            "but charging takes too long."
        ),
    )

    analyze_button = st.button(
        "🔍 Analyze Review",
        type="primary",
    )

    if analyze_button:

        if not review.strip():

            st.warning("Please enter a review.")

        else:

            with st.spinner("Analyzing review..."):

                result = analyze_review(
                    review,
                    analyzer,
                )

            # Overall sentiment
            st.subheader("Overall Sentiment")

            overall = result["overall_sentiment"]

            if overall == "POSITIVE":
                st.success("😊 POSITIVE")

            elif overall == "NEGATIVE":
                st.error("😞 NEGATIVE")

            elif overall == "MIXED":
                st.warning("⚖️ MIXED")

            else:
                st.info("😐 NEUTRAL")

            # Aspect analysis
            st.subheader("Aspect-Level Sentiment")

            if result["aspects"]:

                table_data = []

                for item in result["aspects"]:

                    table_data.append(
                        {
                            "Aspect": item["aspect"],
                            "Sentiment": item["sentiment"],
                            "Confidence": round(
                                item["confidence"],
                                3,
                            ),
                            "Text": item["text"],
                        }
                    )

                aspect_df = pd.DataFrame(
                    table_data
                )

                st.dataframe(
                    aspect_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No automotive aspects detected."
                )


# =========================================================
# BATCH CSV ANALYSIS
# =========================================================

elif page == "Batch CSV Analysis":

    st.header("Batch CSV Analysis")

    st.write(
        "Upload a CSV containing automotive reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )

        if "review" not in df.columns:

            st.error(
                "CSV must contain a 'review' column."
            )

        else:

            if st.button(
                "🚀 Analyze Dataset",
                type="primary",
            ):

                results = []

                progress = st.progress(0)

                for index, row in df.iterrows():

                    review = str(row["review"])

                    result = analyze_review(
                        review,
                        analyzer,
                    )

                    if result["aspects"]:

                        for item in result["aspects"]:

                            results.append(
                                {
                                    "review_id": row.get(
                                        "review_id",
                                        index + 1,
                                    ),
                                    "vehicle": row.get(
                                        "vehicle",
                                        "",
                                    ),
                                    "review": review,
                                    "aspect": item[
                                        "aspect"
                                    ],
                                    "sentiment": item[
                                        "sentiment"
                                    ],
                                    "confidence": round(
                                        item[
                                            "confidence"
                                        ],
                                        4,
                                    ),
                                    "matched_text": item[
                                        "text"
                                    ],
                                    "overall_sentiment":
                                        result[
                                            "overall_sentiment"
                                        ],
                                }
                            )

                    progress.progress(
                        (index + 1) / len(df)
                    )

                result_df = pd.DataFrame(
                    results
                )

                st.session_state[
                    "batch_results"
                ] = result_df

                st.success(
                    f"Analysis complete: "
                    f"{len(df)} reviews processed."
                )

    if "batch_results" in st.session_state:

        result_df = st.session_state[
            "batch_results"
        ]

        st.subheader("Results")

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_data,
            file_name=(
                "automotive_sentiment_results.csv"
            ),
            mime="text/csv",
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.header("Automotive Sentiment Analytics")

    output_file = Path(
        "outputs/automotive_sentiment_results.csv"
    )

    if not output_file.exists():

        st.warning(
            "Run batch analysis first."
        )

    else:

        df = pd.read_csv(output_file)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Reviews",
            df["review"].nunique(),
        )

        col2.metric(
            "Aspect Records",
            len(df),
        )

        col3.metric(
            "Positive",
            (df["sentiment"] == "POSITIVE").sum(),
        )

        col4.metric(
            "Negative",
            (df["sentiment"] == "NEGATIVE").sum(),
        )

        # Sentiment distribution
        st.subheader(
            "Sentiment Distribution"
        )

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

        # Aspect distribution
        st.subheader(
            "Aspect Distribution"
        )

        aspect_counts = (
            df["aspect"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            aspect_counts
        )

        # Average confidence
        st.subheader(
            "Average Confidence by Sentiment"
        )

        confidence = (
            df.groupby("sentiment")[
                "confidence"
            ]
            .mean()
        )

        st.bar_chart(
            confidence
        )

# =========================================================
# VEHICLE COMPARISON
# =========================================================

elif page == "Vehicle Comparison":

    st.header("🚗 Vehicle Comparison")

    st.write(
        "Compare aspect-level sentiment patterns between "
        "two vehicles based on analyzed automotive reviews."
    )

    output_file = Path(
        "outputs/automotive_sentiment_results.csv"
    )

    if not output_file.exists():

        st.warning(
            "No analyzed dataset found. "
            "Run Batch CSV Analysis first."
        )

    else:

        df = pd.read_csv(output_file)

        if "vehicle" not in df.columns:

            st.error(
                "The results dataset does not contain "
                "a vehicle column."
            )

        else:

            vehicles = sorted(
                df["vehicle"]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(vehicles) < 2:

                st.warning(
                    "At least two vehicles are required "
                    "for comparison."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    vehicle_a = st.selectbox(
                        "Vehicle A",
                        vehicles,
                        index=0,
                    )

                with col2:
                    vehicle_b = st.selectbox(
                        "Vehicle B",
                        vehicles,
                        index=1,
                    )

                if vehicle_a == vehicle_b:

                    st.warning(
                        "Please select two different vehicles."
                    )

                else:

                    df_a = df[
                        df["vehicle"] == vehicle_a
                    ].copy()

                    df_b = df[
                        df["vehicle"] == vehicle_b
                    ].copy()

                    st.divider()

                    # -------------------------------------------------
                    # REVIEW COUNTS
                    # -------------------------------------------------

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        f"{vehicle_a} Reviews",
                        df_a["review"].nunique(),
                    )

                    col2.metric(
                        f"{vehicle_b} Reviews",
                        df_b["review"].nunique(),
                    )

                    col3.metric(
                        f"{vehicle_a} Aspect Records",
                        len(df_a),
                    )

                    col4.metric(
                        f"{vehicle_b} Aspect Records",
                        len(df_b),
                    )

                    # -------------------------------------------------
                    # SENTIMENT DISTRIBUTION
                    # -------------------------------------------------

                    st.subheader(
                        "Sentiment Distribution"
                    )

                    sentiment_order = [
                        "POSITIVE",
                        "NEGATIVE",
                        "NEUTRAL",
                    ]

                    sentiment_a = (
                        df_a["sentiment"]
                        .value_counts()
                        .reindex(
                            sentiment_order,
                            fill_value=0,
                        )
                    )

                    sentiment_b = (
                        df_b["sentiment"]
                        .value_counts()
                        .reindex(
                            sentiment_order,
                            fill_value=0,
                        )
                    )

                    sentiment_comparison = pd.DataFrame(
                        {
                            vehicle_a: sentiment_a,
                            vehicle_b: sentiment_b,
                        }
                    )

                    st.bar_chart(
                        sentiment_comparison
                    )

                    # -------------------------------------------------
                    # ASPECT-LEVEL COMPARISON
                    # -------------------------------------------------

                    st.subheader(
                        "Aspect-Level Sentiment Comparison"
                    )

                    common_aspects = sorted(
                        set(df_a["aspect"])
                        | set(df_b["aspect"])
                    )

                    comparison_rows = []

                    for aspect in common_aspects:

                        a_rows = df_a[
                            df_a["aspect"] == aspect
                        ]

                        b_rows = df_b[
                            df_b["aspect"] == aspect
                        ]

                        def dominant_sentiment(rows):

                            if rows.empty:
                                return "N/A"

                            return (
                                rows["sentiment"]
                                .value_counts()
                                .index[0]
                            )

                        comparison_rows.append(
                            {
                                "Aspect": aspect,
                                vehicle_a: dominant_sentiment(
                                    a_rows
                                ),
                                vehicle_b: dominant_sentiment(
                                    b_rows
                                ),
                                f"{vehicle_a} Records":
                                    len(a_rows),
                                f"{vehicle_b} Records":
                                    len(b_rows),
                            }
                        )

                    comparison_df = pd.DataFrame(
                        comparison_rows
                    )

                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # -------------------------------------------------
                    # ASPECT RECORD COUNTS
                    # -------------------------------------------------

                    st.subheader(
                        "Aspect Coverage"
                    )

                    aspect_a = (
                        df_a["aspect"]
                        .value_counts()
                        .rename(vehicle_a)
                    )

                    aspect_b = (
                        df_b["aspect"]
                        .value_counts()
                        .rename(vehicle_b)
                    )

                    aspect_comparison = pd.concat(
                        [
                            aspect_a,
                            aspect_b,
                        ],
                        axis=1,
                    ).fillna(0)

                    st.bar_chart(
                        aspect_comparison
                    )

                    # -------------------------------------------------
                    # CONFIDENCE
                    # -------------------------------------------------

                    st.subheader(
                        "Average Sentiment Confidence"
                    )

                    confidence_df = pd.DataFrame(
                        {
                            vehicle_a: [
                                df_a["confidence"].mean()
                            ],
                            vehicle_b: [
                                df_b["confidence"].mean()
                            ],
                        },
                        index=["Average Confidence"],
                    )

                    st.bar_chart(
                        confidence_df
                    )

                    st.caption(
                        "The comparison presents model-generated "
                        "sentiment patterns and confidence values. "
                        "It does not determine an overall winner "
                        "between vehicles."
                    )