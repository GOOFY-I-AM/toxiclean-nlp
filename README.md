# 🧹 ToxiClean — Toxic Speech Neutralizer

An NLP-powered web application that detects toxic language and
rewrites it into neutral, respectful alternatives in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red)
![NLTK](https://img.shields.io/badge/NLTK-3.7+-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.1+-orange)

## Features
- Detects 6 toxicity categories simultaneously
- Identifies exact toxic words with category labels
- Neutralizes toxic text using 3 strategies
- Batch analysis for multiple texts
- Clean professional Streamlit UI

## Project Structure
\```
NLP project/
├── app/
│   └── streamlit_app.py      # Streamlit web UI
├── modules/
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline
│   ├── preprocessor.py       # Text cleaning
│   ├── classifier.py         # ML model
│   ├── word_detector.py      # Toxic word detection
│   └── neutralizer.py        # Text neutralization
├── notebooks/
│   └── train_model.py        # Model training script
├── data/                     # Place train.csv here
├── models/                   # Saved models go here
├── requirements.txt
└── README.md
\```

## Quick Start
\```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"

# Run the app
python -m streamlit run app/streamlit_app.py
\```

## Dataset
Uses the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset.
Download `train.csv` and place it in the `data/` folder.

## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| NLP | NLTK, scikit-learn |
| ML Model | Logistic Regression + TF-IDF |
| UI | Streamlit |

## Author
Your Name — [github.com/yourusername](https://github.com/yourusername)