

## AI-Based Automotive Review and Customer Sentiment Analytics

AutoSense-NLP is an NLP-based application that analyzes automotive reviews, identifies important vehicle-related aspects, and determines the sentiment associated with each aspect.

The system combines rule-based aspect extraction with a pretrained transformer-based sentiment model and provides an interactive Streamlit dashboard for single-review analysis, batch processing, analytics, and vehicle comparison.

---

## 🎯 Project Overview

Automotive reviews often contain opinions about multiple parts of a vehicle in a single sentence.

For example:

> "The battery range is excellent but charging takes too long."

AutoSense-NLP can identify:

| Aspect | Sentiment |
|---|---|
| Battery | POSITIVE |
| Charging | NEGATIVE |

Overall sentiment: **MIXED**

This aspect-level approach provides more detailed information than assigning a single sentiment to the entire review.

---

## ✨ Key Features

- 📝 Automotive review preprocessing
- 🚗 15-category automotive aspect detection
- 🤖 Transformer-based sentiment classification
- 🔍 Aspect-level sentiment analysis
- 🔀 Mixed sentiment detection
- 📊 Batch CSV analysis
- 📈 Sentiment and aspect analytics
- ⚖️ Vehicle comparison
- 🖥️ Interactive Streamlit dashboard
- 🧪 Automated testing with Pytest
- 📋 Model evaluation on MuSe-CarASTE
- 📄 Complete project documentation and academic report

### Automotive Aspect Categories

The system currently detects:

`Vehicle` · `Engine` · `Battery` · `Mileage` · `Safety`

`Comfort` · `Service` · `Infotainment` · `Price` · `Design`

`Performance` · `Reliability` · `Charging` · `Interior` · `Exterior`

---

## 🏗️ System Architecture

```text
Automotive Review
       ↓
Text Preprocessing
       ↓
Aspect Extraction
       ↓
Clause Segmentation
       ↓
Transformer Sentiment Model
       ↓
Aspect-Sentiment Integration
       ↓
Overall Sentiment
       ↓
Streamlit Dashboard
       ├── Single Review Analysis
       ├── Batch CSV Analysis
       ├── Analytics
       └── Vehicle Comparison
Architecture Diagram

🧠 Sentiment Model

The project uses:

cardiffnlp/twitter-roberta-base-sentiment-latest

The model predicts three classes:

POSITIVE
NEGATIVE
NEUTRAL

The model is used as a pretrained transformer and was not fine-tuned on automotive reviews in this project.

Therefore, the reported evaluation represents a zero-shot baseline rather than an automotive-specific trained model.

📊 Dataset
Synthetic Development Dataset

A synthetic dataset containing 30 automotive reviews is used for development, testing, and dashboard demonstration.

It contains examples covering:

Positive sentiment
Negative sentiment
Neutral sentiment
Mixed sentiment
Multiple aspects within a review

The synthetic dataset is not presented as real customer data.

MuSe-CarASTE

MuSe-CarASTE is used for dataset inspection and evaluation.

The evaluated labeled data contains:

14,574 labeled annotations
Positive: 9,999
Negative: 2,578
Neutral: 1,997

The raw MuSe-CarASTE dataset files are not included in the public repository.

📈 Results
Batch Analysis

Final development run:

Metric	Result
Reviews analyzed	30
Aspect-level records	67
Positive	48
Negative	13
Neutral	6
Zero-Shot Sentiment Evaluation

Evaluation was performed on 14,574 labeled MuSe-CarASTE opinion annotations.

Metric	Score
Accuracy	49.98%
Weighted Precision	82.19%
Weighted Recall	49.98%
Weighted F1	55.51%

These results should be interpreted as a baseline because the pretrained model was not fine-tuned for automotive reviews.

🧪 Testing

The project includes automated tests covering preprocessing, dataset handling, aspect extraction, aspect sentiment, and transformer sentiment prediction.

Final result:

32 passed

Test breakdown:

Module	Tests
Aspect Extraction	7
Aspect Sentiment	7
Dataset	6
Preprocessing	7
Sentiment	5
Total	32
🖥️ Streamlit Dashboard

The application provides four modules:

1. Single Review Analysis

Analyze an individual automotive review and view detected aspects, sentiment, confidence, and overall sentiment.

2. Batch CSV Analysis

Upload multiple reviews and generate structured aspect-level sentiment results.

3. Analytics

Visualize sentiment distribution, aspect distribution, and confidence statistics.

4. Vehicle Comparison

Compare sentiment patterns and aspect-level records for two vehicles.

🛠️ Technology Stack
Category	Technology
Language	Python
NLP	Hugging Face Transformers
Deep Learning	PyTorch
Data Processing	Pandas
Machine Learning	Scikit-learn
Dashboard	Streamlit
Testing	Pytest
Development	VS Code
Version Control	Git / GitHub
📁 Project Structure
AutoSense-NLP/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── synthetic/
│   │       └── automotive_reviews.csv
│   └── processed/
│       └── synthetic_reviews_cleaned.csv
│
├── documents/
│   ├── 1archi.png
│   ├── 2archi.png
│   ├── 4archi.png
│   ├── 5archi.png
│   ├── 6archi.png
│   ├── 7archi.png
│   ├── AutoSense_NLP_Project_Report_10_Pages.pdf
│   └── project-documentation.md
│
├── notebooks/
├── outputs/
├── src/
├── tests/
│
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
📚 Documentation

The documents/ directory contains the supporting materials for the project.

project-documentation.md

Detailed technical documentation covering:

Project objectives
Dataset
Methodology
Architecture
Aspect extraction
Sentiment analysis
Dashboard
Evaluation
Testing
Limitations
Future scope
Running instructions
AutoSense_NLP_Project_Report_10_Pages.pdf

The formal academic project report containing the complete project description, methodology, implementation, results, discussion, conclusion, references, and appendix.

Architecture & Supporting Images

The documents/ directory also contains project-related architecture and visual documentation images used in the report and presentation.

🚀 Installation & Usage
1. Clone the repository
git clone https://github.com/Annwesha2024/AutoSense-NLP.git
cd AutoSense-NLP
2. Create a virtual environment
python -m venv .venv
3. Activate the environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run tests
pytest -q

Expected result:

32 passed
6. Run batch analysis
python src\batch_analysis.py
7. Start the Streamlit dashboard
streamlit run app.py

The application will normally open at:

http://localhost:8501
⚠️ Limitations
Aspect extraction is currently rule-based.
Complex language may lead to incorrect aspect-sentiment associations.
Some automotive terms can be ambiguous.
The transformer model is not fine-tuned for automotive reviews.
The current evaluation is a sentiment-only zero-shot evaluation.
The synthetic development dataset is small and is not representative of the automotive market.
🔮 Future Scope

Future improvements could include:

Fine-tuning transformer models on automotive-domain data
Training a dedicated aspect-based sentiment model
Improved contextual aspect detection
Better clause-level sentiment association
Multilingual automotive review analysis
Larger real-world datasets
Explainable AI
Temporal sentiment analysis
Cloud deployment
Benchmarking multiple transformer models
👤 Author

Annwesha Das

B.Tech — Computer Science & Engineering
Institute of Engineering & Management
Enrolment Number: 12023002001366
Class Roll Number: 8
Section: CSE C