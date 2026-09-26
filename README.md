# AutoSense-NLP

## AI-Based Automotive Review and Customer Sentiment Analytics

AutoSense-NLP is an NLP-based application for analyzing automotive customer reviews, identifying automotive aspects, and determining sentiment associated with those aspects.

The project uses a hybrid aspect-detection approach and transformer-based sentiment classification, followed by batch analytics and an interactive Streamlit dashboard.

---

## Project Status

Current development phase:

**Phase 1 — Project Setup and Data Preprocessing**

Completed:

- Python virtual environment
- Git repository
- Project directory structure
- Synthetic automotive development dataset
- CSV loading and validation
- Basic text preprocessing
- Missing-review handling
- Duplicate-review handling
- Processed CSV generation
- Unit tests for preprocessing

---

## Development Environment

- Operating System: Windows 11
- Language: Python 3.12
- IDE: Visual Studio Code
- Environment: Python virtual environment (`.venv`)
- Version Control: Git
- Primary execution target: CPU
- Optional GPU acceleration: Planned

---

## Project Structure

```text
AutoSense-NLP/
│
├── data/
│   ├── raw/
│   │   ├── synthetic/
│   │   │   └── automotive_reviews.csv
│   │   │
│   │   └── muse_caraste/
│   │       ├── train.csv
│   │       ├── devel.csv
│   │       └── FinalFullAnnotationsAllLabelsCompleted.csv
│   │
│   └── processed/
│       └── synthetic_reviews_cleaned.csv
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── tests/
│   └── test_preprocessing.py
│
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt