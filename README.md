# Speech Classifier NLP

Two NLP projects in one repo: analysing the syntax and style of 19th century novels, and building classifiers to predict political party from UK parliamentary speeches.

## Part One: Novel Analysis

Explored the language of a set of 19th century novels using NLP tools and techniques. The goal was to measure readability, identify stylistic patterns, and extract syntactic relationships from the texts.

### What I Built

- A text reader that parses novel files into a structured DataFrame with title, author, year, and full text.
- Type token ratio (TTR) calculation for each novel using NLTK tokenisation.
- Flesch Kincaid readability scoring using the CMU Pronouncing Dictionary for syllable counts.
- Full spaCy parsing pipeline that processes each novel into tokenised Doc objects, stored as pickle files for reuse.
- Functions to extract the most common adjectives, most frequent subjects of a given verb, and subject verb pairs ranked by Pointwise Mutual Information (PMI).

### Key Findings

- Flesch Kincaid scores ranged from 4.65 (The Secret Garden) to 14.68 (Erewhon), but the scores don't always reflect true difficulty. Texts from different time periods can score similarly despite very different language styles.
- PMI based subject verb analysis revealed more meaningful associations than raw frequency counts, surfacing contextually interesting patterns that frequency alone would miss.

## Part Two: Parliamentary Speech Classification

Built machine learning classifiers to predict political party (Labour, Conservative, Liberal Democrat, Independent) from the text of UK parliamentary speeches using the Hansard dataset (40,000 speeches).

### What I Built

- Data cleaning pipeline: filtered by party, speech class, and minimum length (1000 characters). Merged Labour (Co-op) into Labour.
- TF-IDF vectorisation with stopword removal, max 3000 features, and stratified train/test split.
- Two classification approaches:
  - **Standard ML:** TfidfVectorizer with Random Forest (n_estimators=300) and linear SVM. Tested with unigrams, bigrams, and trigrams.
  - **Pipeline with SMOTE:** Custom tokenizer using regex cleaning and spaCy stopword removal, combined with SMOTE oversampling to handle class imbalance, and the same classifiers.

### Results

| Approach                            | Classifier    | Macro F1 |
| ----------------------------------- | ------------- | -------- |
| Standard ML (ngram 1,2)             | SVM           | 0.5116   |
| Standard ML (ngram 1,2)             | Random Forest | 0.3462   |
| Pipeline + SMOTE + Custom Tokenizer | SVM           | 0.4898   |
| Pipeline + SMOTE + Custom Tokenizer | Random Forest | 0.3653   |

SVM with linear kernel consistently outperformed Random Forest across all configurations. The custom tokenizer made text features cleaner but didn't significantly improve classification. N-gram coverage and class frequency mattered more than token level cleaning. The severe class imbalance (Independent and Liberal Democrat had very few speeches) pulled down the macro F1 scores, with those classes scoring close to 0 for Random Forest.

## Project Structure

```
speech-classifier-nlp/
├── PartOne.py              # Novel analysis: readability, parsing, syntactic extraction.
├── PartTwo.py              # Speech classification: cleaning, vectorisation, classifiers.
├── answers.txt             # Text answers for discussion questions.
├── texts/
│   ├── novels/             # 19th century novel text files.
│   └── hansard40000.csv    # UK parliamentary speech dataset.
└── pickles/                # Serialised parsed DataFrames.
```

## What I Learned

- How to use spaCy's dependency parser to extract syntactic relationships (subjects, objects, verbs) from text.
- Calculating PMI to find statistically meaningful word associations rather than just frequent ones.
- The limitations of readability formulas like Flesch Kincaid when comparing texts across different time periods.
- Building and comparing ML pipelines for text classification with TF-IDF, n-grams, and custom tokenisers.
- Handling class imbalance with SMOTE and understanding why macro F1 can be misleading with severely imbalanced datasets.
- That sometimes simpler approaches (standard TF-IDF + SVM) outperform more complex pipelines, and that's a valid finding too.

## Tech Stack

- Python, NLTK, spaCy
- scikit-learn, imbalanced-learn
- Pandas, NumPy
