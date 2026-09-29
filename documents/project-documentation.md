# AutoSense-NLP — Project Documentation

## AI-Based Automotive Review and Customer Sentiment Analytics

This directory contains the complete supporting documentation and visual
materials for the **AutoSense-NLP** project.

AutoSense-NLP is an NLP-based automotive review analytics system that
extracts automotive-related aspects from customer reviews and determines
the sentiment associated with those aspects using a transformer-based
sentiment model.

---

# 1. Project Overview

Automotive customer reviews contain opinions about several different
vehicle characteristics such as:

- Engine
- Battery
- Mileage
- Safety
- Comfort
- Service
- Infotainment
- Price
- Design
- Performance
- Reliability
- Charging
- Interior
- Exterior
- Vehicle

A single review can contain multiple aspects and different sentiments
towards those aspects.

For example:

> "The battery range is excellent but charging takes too long."

AutoSense-NLP identifies:

| Aspect | Sentiment |
|---|---|
| Battery | POSITIVE |
| Charging | NEGATIVE |

The overall sentiment is:

**MIXED**

---

# 2. Project Objectives

The main objectives of AutoSense-NLP are:

1. Preprocess automotive review text.
2. Detect automotive-specific aspects.
3. Classify sentiment using a transformer model.
4. Perform aspect-level sentiment analysis.
5. Detect mixed sentiment within reviews.
6. Process multiple reviews using CSV files.
7. Generate structured sentiment-analysis results.
8. Provide an interactive Streamlit dashboard.
9. Provide analytics and vehicle comparison.
10. Evaluate the sentiment component on automotive-domain data.
11. Validate the implementation using automated tests.

---

# 3. System Architecture

The complete processing pipeline is:

```text
                AUTOMOTIVE REVIEW
                       |
                       v
              TEXT PREPROCESSING
                       |
                       v
              ASPECT EXTRACTION
                       |
                       v
              CLAUSE SEGMENTATION
                       |
                       v
          TRANSFORMER SENTIMENT MODEL
                       |
                       v
           ASPECT-SENTIMENT INTEGRATION
                       |
                       v
               OVERALL SENTIMENT
                       |
              +--------+--------+
              |                 |
              v                 v
        BATCH ANALYSIS     STREAMLIT DASHBOARD
                                |
                    +-----------+-----------+
                    |           |           |
                    v           v           v
                 Analytics   Vehicle    Single Review
                            Comparison    Analysis