# A Comprehensive Report on Abuse Detection Using Artificial Intelligence

## Part 2: Data Collection, Exploratory Analysis, Preprocessing & Classical Model Training

---

## **Table of Contents — Part 2**

10. [Data Collection and Dataset Selection](#10-data-collection-and-dataset-selection)
    - 10.1 Criteria for Dataset Selection
    - 10.2 The Davidson et al. Hate Speech Dataset
    - 10.3 Dataset Structure and Schema
    - 10.4 Why This Dataset Was Chosen
11. [Exploratory Data Analysis (EDA)](#11-exploratory-data-analysis-eda)
    - 11.1 Overview and EDA Goals
    - 11.2 Class Distribution Analysis
    - 11.3 Text Length Distribution Analysis
    - 11.4 Lexical Analysis via Word Clouds
    - 11.5 Character-Level Analysis
    - 11.6 Missing Values and Data Quality Checks
    - 11.7 Key EDA Insights and Implications for Modeling
12. [Data Preprocessing & Text Cleaning](#12-data-preprocessing--text-cleaning)
    - 12.1 Why Preprocessing Is Critical in NLP
    - 12.2 Step-by-Step Preprocessing Pipeline
    - 12.3 Noise Removal — URLs, Mentions, Hashtags
    - 12.4 Case Folding and Punctuation Removal
    - 12.5 Stopword Removal
    - 12.6 Lemmatization vs. Stemming — Decision and Justification
    - 12.7 Complete Preprocessing Example (Before and After)
    - 12.8 Binary Label Engineering
    - 12.9 Saving the Processed Dataset
13. [Feature Engineering: From Text to Numbers](#13-feature-engineering-from-text-to-numbers)
    - 13.1 Why Machines Cannot Read Text Directly
    - 13.2 Bag-of-Words (BoW) and Its Limitations
    - 13.3 Term Frequency–Inverse Document Frequency (TF-IDF) — Theory
    - 13.4 Mathematical Formulation of TF-IDF
    - 13.5 N-gram Generation (Unigrams and Bigrams)
    - 13.6 Vocabulary Size Management and Sparse Matrices
    - 13.7 Vectorizer Fitting and Persistence
14. [Classical Machine Learning Model Training](#14-classical-machine-learning-model-training)
    - 14.1 Training Strategy Overview
    - 14.2 Stratified Train-Test Split
    - 14.3 Logistic Regression
    - 14.4 Multinomial Naive Bayes
    - 14.5 Linear Support Vector Classifier (LinearSVC)
    - 14.6 Random Forest Classifier
    - 14.7 k-Fold Cross-Validation
    - 14.8 Classical Model Results Summary
    - 14.9 Transition to GPU-Accelerated Training

---

## **10. Data Collection and Dataset Selection**

The quality, volume, and representativeness of training data is the single most critical determinant of a machine learning system's real-world effectiveness. An ML model can only learn what its training data teaches it. This fundamental principle — often summarized in data science as **"Garbage In, Garbage Out"** — guided a careful, criteria-driven dataset selection process.

### **10.1 Criteria for Dataset Selection**

Before acquiring any data, the following selection criteria were established:

1. **Domain Relevance:** The data must originate from real social media platforms, not lab-generated synthetic text, to ensure the model learns authentic patterns of online communication including slang, abbreviations, and non-standard grammar.
2. **Annotation Quality:** The labels must have been assigned by multiple human annotators with documented inter-annotator agreement metrics, not auto-labeled by a simple keyword filter.
3. **Size Appropriateness:** Large enough to support robust deep learning fine-tuning (>10,000 samples minimum), but manageable enough to complete training in reasonable time on consumer GPU hardware.
4. **License Compatibility:** The dataset must be publicly available for academic and research use.
5. **Class Nuance:** Preference for datasets with nuanced labeling (multiple abuse categories) over pure binary labels, allowing us to make deliberate label engineering decisions.

### **10.2 The Davidson et al. Hate Speech Dataset**

The **"Hate Speech and Offensive Language"** dataset, created by Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber (2017) and published at the International AAAI Conference on Web and Social Media (ICWSM), was selected as the primary training corpus.

**Data Collection Methodology (as described by the original authors):**
1. The authors began with a lexicon of terms associated with hate speech, compiled from Hatebase.org, a structured online repository of terms used to target groups.
2. Twitter's public API was used to collect all tweets containing at least one term from this lexicon, yielding an initial set of approximately 33,000 unique tweets.
3. These tweets were then uploaded to CrowdFlower (now Appen), a crowdsourcing annotation platform. Each tweet was presented to a minimum of three human raters.
4. Annotators were instructed to classify each tweet into one of three categories: (0) Hate Speech, (1) Offensive Language, or (2) Neither.
5. The final label for each tweet was determined by majority vote among the annotators. Only tweets where a clear majority consensus existed were retained in the final dataset.

This rigorous annotation methodology ensures high-quality, human-validated ground truth labels.

### **10.3 Dataset Structure and Schema**

The dataset is delivered as a single Comma-Separated Values (CSV) file named `labeled_data.csv`. Upon loading into a pandas DataFrame (`pd.read_csv('labeled_data.csv')`), the structure is as follows:

| Column Name | Data Type | Description |
|---|---|---|
| `Unnamed: 0` | int64 | Original row index (irrelevant, dropped) |
| `count` | int64 | Total number of CrowdFlower users who annotated this tweet |
| `hate_speech` | int64 | Number of annotators who labeled this tweet as Hate Speech |
| `offensive_language` | int64 | Number of annotators who labeled this tweet as Offensive Language |
| `neither` | int64 | Number of annotators who labeled this tweet as Neither |
| `class` | int64 | **Final majority-vote label** (0=Hate Speech, 1=Offensive, 2=Neither) |
| `tweet` | object (string) | The raw text content of the tweet |

**Key Statistics of the Raw Dataset:**
- **Total rows:** 24,783 tweets
- **Class 0 (Hate Speech):** 1,430 tweets (5.77% of total)
- **Class 1 (Offensive Language):** 19,190 tweets (77.43% of total)
- **Class 2 (Neither):** 4,163 tweets (16.80% of total)
- **Missing values:** 0 in critical columns
- **Duplicate tweets:** 19 exact duplicates (retained as potentially valid retweets)

### **10.4 Why This Dataset Was Chosen Over Alternatives**

Several alternative datasets were evaluated before this selection was made:

| Dataset | Size | Platform | Reason Not Selected |
|---|---|---|---|
| Jigsaw Toxic Comments (Kaggle) | 160,000 | Wikipedia | Too large for rapid prototyping; Wikipedia comments differ in style from social media |
| HatEval (SemEval 2019) | 20,000 | Twitter | Binary labels only; English + Spanish complicates single-language scope |
| OffComBR | 1,250 | YouTube | Too small for deep learning fine-tuning |
| **Davidson et al. (2017)** | **24,783** | **Twitter** | **Optimal size, multi-class nuanced labels, public, well-cited** |

---

## **11. Exploratory Data Analysis (EDA)**

Exploratory Data Analysis (EDA) is the process of examining a dataset prior to any modeling, using statistical summaries and visualizations to understand its underlying structure, distribution, and anomalies. EDA is not just academic preparation — insights gained during EDA directly inform critical modeling decisions: which algorithms to choose, which metrics to optimize, and whether data transformations are needed. All EDA was conducted in a Jupyter Notebook using matplotlib and seaborn, producing 31 visualization figures saved to the `reports/figures/` directory.

### **11.1 Overview and EDA Goals**

The EDA phase for this project pursued five specific goals:
1. Understand the distribution of labels (class balance).
2. Characterize the length distribution of tweets across classes.
3. Identify the most discriminative vocabulary for each class.
4. Detect any data quality issues (missing values, duplicates, malformed entries).
5. Generate visualizations that communicate key dataset properties.

### **11.2 Class Distribution Analysis**

The first and most consequential EDA step was examining the distribution of the three class labels in the `class` column.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 11.1 — Class Distribution in Raw Dataset (Before Binarization)     │
│                                                                              │
│   [Insert Vertical Bar Chart Here]                                           │
│                                                                              │
│   X-axis: Class Labels (Class 0: Hate Speech, Class 1: Offensive, Class 2: Neither)    │
│   Y-axis: Number of Tweets (count)                                           │
│   Values: Class 0 = 1,430 | Class 1 = 19,190 | Class 2 = 4,163             │
│   Color: Class 0 = dark red, Class 1 = orange-red, Class 2 = green          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: Severe class imbalance in the raw dataset. Class 1 (Offensive Language)
 dominates at 77.4%, creating a significant modeling challenge.*
```

---

**Critical Implication of Class Imbalance:**
The 77.4% majority of Class 1 creates a dangerous trap for naive model training. A model that always predicts "Offensive Language" regardless of input would achieve 77.4% raw accuracy — appearing highly performant while being completely useless. This observation made the following decisions mandatory:

1. **Binary Binarization:** Consolidate Classes 0 and 1 into a single "Abusive" category (combined = 82.2%) versus Class 2 as "Safe" (17.8%). While still imbalanced, this binary framing is more practically useful and directly reflects the use case.
2. **Stratified Splitting:** Ensure the 17.8% minority class is proportionally represented in both train and test splits.
3. **F1-Score as Primary Metric:** Raw accuracy is discarded in favor of F1-Score, which measures the harmonic mean of Precision and Recall, penalizing both false positives and false negatives equally.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 11.2 — Class Distribution After Binary Label Engineering             │
│                                                                              │
│   [Insert Pie Chart or Bar Chart Here]                                       │
│                                                                              │
│   Two segments:                                                              │
│   - Abusive (Class 0 + Class 1): 20,620 tweets (83.2%)                      │
│   - Safe (Class 2 only): 4,163 tweets (16.8%)                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: After binarization, the dataset has 83.2% Abusive and 16.8% Safe
 samples. Stratified splitting ensures this ratio is maintained in train/test sets.*
```

---

### **11.3 Text Length Distribution Analysis**

A new computed column `text_length` (character count of the raw tweet) was added to the DataFrame: `df['text_length'] = df['tweet'].str.len()`. This feature was then visualized using histograms, separately by class.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 11.3 — Tweet Length Distribution by Class                           │
│                                                                              │
│   [Insert Overlapping Histogram or KDE Plot Here]                            │
│                                                                              │
│   X-axis: Tweet character length (0 to 160 characters)                      │
│   Y-axis: Frequency / Density                                                │
│   Series: Three overlapping curves — Class 0, Class 1, Class 2              │
│   Show the 140-character cap as a vertical dashed line                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: All three classes show similar length distributions capped at the
 historical 140-character Twitter limit, with most tweets between 60–130 characters.*
```

---

**Key Observations:**
- The distribution is left-skewed and hard-capped at approximately 140 characters (the historical Twitter character limit at the time of data collection).
- **Abusive tweets** tend to be slightly longer on average (mean ~98 chars) versus **Safe tweets** (mean ~85 chars), suggesting abusive content tends to be more verbose.
- The short nature of tweets (maximum 140 chars, typically 60–130) informed the decision to use `max_length=128` tokens for DistilBERT tokenization — sufficient to capture the full content of virtually every tweet with minimal truncation.

### **11.4 Lexical Analysis via Word Clouds**

Word cloud visualizations were generated for each class to identify the most frequent and distinctive vocabulary. The `WordCloud` library was configured with `max_words=100, background_color='white', colormap='Reds'` for the abusive class and `colormap='Greens'` for the safe class.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 11.4a — Word Cloud: Abusive Tweets (Classes 0 and 1 Combined)      │
│                                                                              │
│   [Insert Word Cloud Image Here — Red/Dark Color Scheme]                    │
│                                                                              │
│   Most prominent words (proportional to frequency):                          │
│   - High frequency profanity, slurs, aggressive verbs                        │
│   - Terms targeting gender, ethnicity, sexual orientation                    │
│   - Aggressive personal attacks                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: Word cloud from the combined abusive class. Larger words appear
 more frequently. The vocabulary reveals aggressive, demeaning, and slur-based language.*
```

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 11.4b — Word Cloud: Safe Tweets (Class 2 — Neither)                │
│                                                                              │
│   [Insert Word Cloud Image Here — Green Color Scheme]                       │
│                                                                              │
│   Most prominent words:                                                      │
│   - Common nouns: "people", "day", "time", "life", "good"                   │
│   - Positive verbs: "love", "feel", "think", "help"                         │
│   - Casual social language: "like", "lol", "just", "one"                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: Safe tweet vocabulary is dominated by everyday conversational words,
 with a clear lexical contrast compared to the abusive class word cloud.*
```

---

The stark contrast between the two word clouds visually confirms that the abusive and safe classes occupy distinct lexical spaces — a strong signal that machine learning classification should be feasible and effective.

### **11.5 Character-Level Analysis**

Beyond word frequencies, character-level patterns were examined. Abusive tweets frequently contain:
- **Deliberate character repetition:** "YOUUUU SUUUCK" — for emphasis.
- **Special character usage:** "@" mentions targeting specific users were far more common in abusive tweets.
- **ALL-CAPS proportion:** The ratio of uppercase characters to total characters was notably higher in Class 0 (Hate Speech) tweets, suggesting shouting/aggressive emphasis.

These observations motivated the inclusion of **bigrams** in the TF-IDF vectorization, as character-pattern-based features like repeated vowels are better captured at the phrase level.

### **11.6 Missing Values and Data Quality Checks**

Systematic data quality validation was performed:

```python
print(df.isnull().sum())         # Missing values per column
print(df.duplicated().sum())      # Exact duplicate rows
print(df['tweet'].str.len().min()) # Shortest tweet
print(df['tweet'].str.len().max()) # Longest tweet
```

**Results:**
- **Missing values:** Zero in all columns. The dataset is complete.
- **Duplicate tweets:** 19 exact duplicates. These were retained, as they likely represent retweets — a valid real-world phenomenon where multiple users repeat identical content.
- **Minimum tweet length:** 3 characters (some very short posts like "lol").
- **Maximum tweet length:** 166 characters (a few tweets slightly exceeded the 140-char limit, possibly collected at a platform transition period).

### **11.7 Key EDA Insights and Implications for Modeling**

| Insight Discovered | Implication for Modeling Decision |
|---|---|
| 77.4% Class 1 dominance | Use F1-Score, not Accuracy, as primary metric |
| Three-class imbalance | Binarize labels: Abusive (0+1) vs. Safe (2) |
| All tweets ≤ 166 chars | Set DistilBERT max_length=128 tokens |
| Distinct abusive vocabulary | TF-IDF should capture this signal effectively |
| Bigrams add contextual meaning | Use ngram_range=(1,2) in TF-IDF |
| 0 missing values | No imputation required; safe to proceed |

---

## **12. Data Preprocessing & Text Cleaning**

Raw social media text is the most challenging type of text data to process in NLP. Unlike carefully written news articles or Wikipedia entries, tweets contain slang, intentional misspellings, abbreviations, emojis, URL links, user mentions, hashtags, and highly irregular punctuation. All of this constitutes **noise** — content that does not help a machine learning model learn linguistic abuse patterns and that dramatically increases the dimensionality of the feature space unnecessarily.

The preprocessing pipeline is implemented in `src/preprocess.py` as a single, composable function `clean_text(text: str) -> str`. This function accepts one raw tweet string and returns one cleaned string.

### **12.1 Why Preprocessing Is Critical in NLP**

Consider this example: the same abusive sentiment expressed in three different ways:
- `"I hate you so much!!!"` 
- `"i HATE you SO much"` 
- `"i h8 u so much :@"`

Without preprocessing, these three strings are completely different sequences of characters. A model trained on the first might not recognize the second or third. After cleaning and normalization, all three can be transformed into a standardized form that the model can generalize from.

Additionally, vocabulary size directly impacts model training time and memory consumption. Without preprocessing, the TF-IDF vocabulary might contain "run", "runs", "running", "ran", "runner" as five distinct features — consuming five columns in the feature matrix. After lemmatization, all five map to "run" — one column. Reducing vocabulary size by orders of magnitude makes the models faster to train and less prone to overfitting on rare word forms.

### **12.2 Step-by-Step Preprocessing Pipeline**

The `clean_text()` function applies the following transformations in sequence:

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 12.1 — Text Preprocessing Pipeline Flowchart                        │
│                                                                              │
│   [Insert Vertical Flowchart Here]                                           │
│                                                                              │
│   Boxes in sequence (top to bottom):                                         │
│   1. Raw Tweet Input                                                         │
│   2. Convert to Lowercase (.lower())                                         │
│   3. Remove HTTP URLs (regex: http\S+|www\S+)                                │
│   4. Remove @Mentions (regex: @\w+)                                          │
│   5. Remove #Hashtags (regex: #\w+)                                          │
│   6. Remove Non-Alphabetic Characters (regex: [^a-z\s])                      │
│   7. Tokenize (split on whitespace)                                          │
│   8. Remove Stopwords (NLTK English stopwords)                               │
│   9. Lemmatize each token (WordNetLemmatizer)                                │
│   10. Rejoin tokens into clean string                                         │
│   11. Clean Text Output                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: The complete text preprocessing pipeline applied to every tweet
 before feature extraction. Each step reduces noise or vocabulary dimensionality.*
```

---

### **12.3 Noise Removal — URLs, Mentions, Hashtags**

**URL Removal:**
```python
text = re.sub(r'http\S+|www\S+', '', text)
```
Hyperlinks provide zero linguistic information for abuse detection. A URL like `http://bit.ly/xk93j` tells the model nothing about whether the surrounding message is abusive. Retaining URLs would create thousands of unique "word" features in the TF-IDF vocabulary that appear only once — precisely the type of noise `min_df=2` is designed to exclude, but it is cleaner to remove them at source.

**User Mention Removal:**
```python
text = re.sub(r'@\w+', '', text)
```
A mention like `@JohnSmith2024` is a unique identifier, not a word with semantic meaning. If the training data contains many abusive tweets directed at a specific Twitter account (e.g., a public figure who was heavily harassed), retaining their mention would cause the model to associate that specific username with abuse — a form of data contamination that would fail to generalize to new users.

**Hashtag Removal:**
```python
text = re.sub(r'#\w+', '', text)
```
Hashtags like `#BlackLivesMatter` or `#MAGA` can carry political and social context, but their interpretation is highly dependent on broader societal context that a model trained on a 2017 dataset cannot reliably generalize. Removing them prevents the model from learning spurious correlations between specific campaigns and abuse labels.

### **12.4 Case Folding and Punctuation Removal**

**Lowercase Conversion:**
```python
text = text.lower()
```
This ensures that "Abuse", "ABUSE", and "abuse" are treated as the same word. Without this step, the model treats them as three separate features, tripling the effective vocabulary size and learning three separate — but semantically identical — relationships.

**Non-Alphabetic Character Removal:**
```python
text = re.sub(r'[^a-z\s]', '', text)
```
Removes all digits, punctuation, and special characters. After URL, mention, and hashtag removal, remaining non-alphabetic content (exclamation marks, question marks, quotation marks) adds noise without semantic value for classification.

### **12.5 Stopword Removal**

English stopwords are high-frequency function words that provide grammatical structure but carry minimal semantic meaning: "the", "is", "in", "at", "a", "and", "but", "or", "it", "this", "that", "with", "for", "of", "on", "are", "was", "be", "to", "from"...

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in text.split() if w not in stop_words]
```

NLTK's English stopword list contains 179 words. Removing them from every tweet has two benefits:
1. Reduces the number of features in the TF-IDF matrix by eliminating the most common words.
2. Concentrates the model's learning on the semantically significant words that actually differentiate abusive from safe content.

**Important caveat:** Some NLP researchers argue that stopword removal can harm performance for tasks requiring syntactic understanding (e.g., "not good" loses its negation if "not" is in the stopword list). For this project, the classical TF-IDF pipeline benefited from stopword removal; the DistilBERT pipeline operates on the **raw** (non-preprocessed) text, preserving all syntactic structure.

### **12.6 Lemmatization vs. Stemming — Decision and Justification**

Two standard approaches exist for reducing words to their root forms:

**Stemming (e.g., Porter Stemmer, Snowball Stemmer):**
Applies a set of heuristic string-truncation rules. Fast but crude.
- "running" → "run" ✓
- "studies" → "studi" ✗ (not a real word)
- "ponies" → "poni" ✗ (not a real word)
- "better" → "better" ✗ (fails to recognize relationship to "good")

**Lemmatization (WordNet Lemmatizer):**
Uses a linguistic knowledge base (WordNet) to find the canonical dictionary form (lemma) of each word.
- "running" → "run" ✓
- "studies" → "study" ✓
- "ponies" → "pony" ✓
- "better" → "good" ✓ (correctly identifies the comparative)
- "was" → "be" ✓

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
```

**Decision:** `WordNetLemmatizer` was selected. While slower than stemming (requires WordNet database lookups), it produces valid English words, which improves TF-IDF feature quality. The performance overhead is negligible for a 25,000-tweet dataset.

### **12.7 Complete Preprocessing Example (Before and After)**

The following illustrates the pipeline applied to a representative tweet:

**Input (raw):**
```
"RT @UserXYZ: you absolute idiot!!! go kill yourself!! #loser http://bit.ly/abc123"
```

**After Step 1 (lowercase):**
```
"rt @userxyz: you absolute idiot!!! go kill yourself!! #loser http://bit.ly/abc123"
```

**After Step 2 (URL removal):**
```
"rt @userxyz: you absolute idiot!!! go kill yourself!! #loser"
```

**After Step 3 (mention removal):**
```
"rt : you absolute idiot!!! go kill yourself!! #loser"
```

**After Step 4 (hashtag removal):**
```
"rt : you absolute idiot!!! go kill yourself!!"
```

**After Step 5 (non-alpha removal):**
```
"rt  you absolute idiot go kill yourself"
```

**After Step 6 (stopword removal):**
```
"rt absolute idiot go kill"
```
(Removed: "you", "yourself")

**After Step 7 (lemmatization):**
```
"rt absolute idiot go kill"
```
(No changes in this example, as all words are already base forms)

**Final output:** `"rt absolute idiot go kill"`

The cleaned output retains the semantically critical words ("idiot", "kill") while discarding URL noise, user identifiers, and function words. This represents the input that is fed into TF-IDF vectorization.

### **12.8 Binary Label Engineering**

The three original class labels (0=Hate Speech, 1=Offensive, 2=Neither) are consolidated into binary labels using a pandas `map()` operation:

```python
label_map = {0: 1, 1: 1, 2: 0}
df['label'] = df['class'].map(label_map)
```

**Mapping Logic and Justification:**
- Original **Class 0 (Hate Speech)** → **New Label 1 (Abusive):** Hate speech is unambiguously harmful and should be flagged.
- Original **Class 1 (Offensive Language)** → **New Label 1 (Abusive):** While context-dependent, offensive language in the dataset represents content that would be reported by users on a social platform. For the purposes of a safety filter, classifying it as requiring attention is the conservative, safety-first choice.
- Original **Class 2 (Neither)** → **New Label 0 (Safe):** Content annotators determined this content to be neither hateful nor offensive; it is the ground truth "benign" class.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 12.2 — Label Binarization Decision Flow                             │
│                                                                              │
│   [Insert Decision Tree / Flow Diagram Here]                                 │
│                                                                              │
│   Show:                                                                      │
│   - Original 3-class structure at top                                        │
│   - Arrow from Class 0 (Hate) → Label 1 (Abusive)                           │
│   - Arrow from Class 1 (Offensive) → Label 1 (Abusive)                      │
│   - Arrow from Class 2 (Neither) → Label 0 (Safe)                           │
│   - Final binary label counts at bottom                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: Label engineering reduces the three-class problem to a binary Safe/
 Abusive classification, better aligned with the system's practical use case.*
```

---

### **12.9 Saving the Processed Dataset**

The cleaned dataset — with the original `tweet` column preserved alongside the new `cleaned_text` and `label` columns — was saved to `data/processed/cleaned_data.csv`. This ensures the preprocessing step is executed only once; all subsequent training scripts read from this saved file rather than re-executing the preprocessing pipeline on every run.

**Final processed dataset dimensions:**
- **Rows:** 24,764 (19 exact duplicates removed)
- **Columns:** `tweet`, `cleaned_text`, `label`
- **Label distribution:** 20,601 Abusive (83.2%), 4,163 Safe (16.8%)

---

## **13. Feature Engineering: From Text to Numbers**

After preprocessing, the `cleaned_text` column contains a column of standardized strings. However, mathematical models cannot operate on strings; they require numerical vectors. Feature engineering is the process of converting this text into a mathematical representation that preserves the semantic content in numerical form.

### **13.1 Why Machines Cannot Read Text Directly**

All machine learning algorithms, regardless of their sophistication, ultimately perform mathematical operations: matrix multiplications, summations, gradient calculations. These operations are defined over real numbers. Text strings like "absolute idiot go kill" must be converted into a fixed-length numerical vector — for example, a vector of 15,000 floating-point numbers — before any algorithm can process it.

The challenge is to design this conversion such that similar texts produce similar numerical vectors, and texts with very different meanings produce distant vectors.

### **13.2 Bag-of-Words (BoW) and Its Limitations**

The simplest vectorization approach is **Bag-of-Words (CountVectorizer)**: create a vocabulary of all unique words across the corpus, and represent each document as a vector where each element counts how many times the corresponding word appears.

**Example with vocabulary = ["hate", "love", "you", "don't"]:**
- "I love you" → [0, 1, 1, 0]
- "I don't love you" → [0, 1, 1, 1]
- "I don't hate you" → [1, 0, 1, 1]

**Critical Problem:** The vectors for "I don't love you" and "I don't hate you" are very similar ([0,1,1,1] vs [1,0,1,1]) even though the sentences have opposite emotional valence. Additionally, common words like "you" receive the same weight as rare but highly informative words, regardless of their discriminative power.

### **13.3 Term Frequency–Inverse Document Frequency (TF-IDF) — Theory**

TF-IDF resolves the BoW weighting problem by considering not just how often a word appears in a document, but also how **unique** that word is across the entire corpus. A word that appears in every single tweet (like "you" or "the") provides no discriminative signal — it's equally common in both abusive and safe tweets. But a specific slur that appears almost exclusively in abusive tweets carries enormous predictive value.

**Core Intuition:**
- **High TF-IDF score:** The word appears frequently in this specific document AND rarely across the corpus as a whole → the word is uniquely characteristic of this document.
- **Low TF-IDF score:** The word either appears infrequently in this document, OR appears frequently across all documents (making it generic and uninformative).

### **13.4 Mathematical Formulation of TF-IDF**

The TF-IDF weight for term $t$ in document $d$ given corpus $D$ is:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

**Term Frequency:**
$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

**Inverse Document Frequency (with log smoothing and +1 to avoid division by zero):**
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Where:
- $|D|$ = total number of documents in the corpus (24,764 tweets)
- $|\{d \in D : t \in d\}|$ = number of documents containing term $t$

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 13.1 — TF-IDF Concept Diagram                                       │
│                                                                              │
│   [Insert Visual Diagram Showing TF × IDF = TF-IDF]                         │
│                                                                              │
│   Left column: Document-level term frequency (bar chart per word)            │
│   Middle column: Corpus-level IDF weight (decreasing for common words)       │
│   Right column: TF-IDF score (product — highlights rare-but-local terms)     │
│                                                                              │
│   Example: "kill" → TF=0.08, IDF=3.2, TF-IDF=0.256 (high — discriminative) │
│   Example: "the" → TF=0.12, IDF=0.1, TF-IDF=0.012 (low — generic)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: TF-IDF rewards words that are frequent in a specific document but
 rare across the corpus, making it ideal for identifying discriminative abuse vocabulary.*
```

---

### **13.5 N-gram Generation (Unigrams and Bigrams)**

A **unigram** is a single word. A **bigram** is a pair of consecutive words. The importance of bigrams is illustrated by negation:

| Text | Unigrams | Bigrams |
|---|---|---|
| "I am not happy" | I, am, not, happy | I am, am not, not happy |
| "I am very happy" | I, am, very, happy | I am, am very, very happy |

With unigrams only (after stopword removal): both "not happy" and "very happy" reduce to just "happy" — identical features despite opposite sentiments. The bigram "not happy" is a distinct, semantically meaningful feature that unigrams cannot capture.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=15000,     # Top 15,000 most informative n-grams
    ngram_range=(1, 2),     # Include both unigrams and bigrams
    min_df=2,               # Ignore terms appearing in fewer than 2 documents
    sublinear_tf=True       # Apply log(1+tf) to dampen extreme frequencies
)
```

### **13.6 Vocabulary Size Management and Sparse Matrices**

With 24,764 tweets and a vocabulary spanning unigrams and bigrams, the unconstrained vocabulary could easily contain 200,000+ distinct features. Setting `max_features=15000` retains only the 15,000 n-grams with the highest TF-IDF scores across the entire corpus. The `min_df=2` parameter discards any n-gram appearing in fewer than 2 documents — eliminating typos, unique identifiers, and other singleton noise.

**Sparse Matrix Representation:**
The resulting feature matrix would be 24,764 rows × 15,000 columns = 371 million cells. However, in any given tweet (average ~15 words), fewer than 30 out of 15,000 features have non-zero values. A dense matrix storing all 371 million values would waste enormous memory. Instead, scikit-learn uses SciPy's **Compressed Sparse Row (CSR)** matrix format, which stores only the indices and values of non-zero cells — reducing memory consumption from ~3 GB to ~15 MB.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 13.2 — Sparse Matrix Visualization                                  │
│                                                                              │
│   [Insert Matrix Visualization showing sparse nature]                        │
│                                                                              │
│   Show: 10×20 grid of cells                                                  │
│   Most cells shaded grey (zero)                                              │
│   3-4 cells per row shaded colored (non-zero TF-IDF values)                 │
│   Legend: grey=0.0, color=TF-IDF weight                                     │
│   Label: "Only ~0.2% of cells are non-zero"                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: A TF-IDF matrix is extremely sparse — less than 0.2% of cells
 contain non-zero values. Sparse matrix storage reduces memory by 99.8%.*
```

---

### **13.7 Vectorizer Fitting and Persistence**

The vectorizer is **fitted** on the training set only (never on the test set, to prevent data leakage):

```python
X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)  # Learn vocab on train
X_test_tfidf  = vectorizer.transform(X_test_cleaned)        # Apply (do NOT re-fit) on test
```

After fitting, the vectorizer object (containing the learned vocabulary and IDF weights) is serialized using Python's `pickle` library and saved to `models/tfidf_vectorizer.pkl`. During real-time inference in the Flask application, this saved vectorizer is loaded and applied to new user input — ensuring consistency between training and inference.

---

## **14. Classical Machine Learning Model Training**

With the numerical feature matrices prepared (X_train_tfidf and X_test_tfidf) and the binary labels (y_train, y_test), the system proceeded to the model training phase. The script `src/train_cpu.py` implements classical baseline model training on the CPU.

The purpose of this phase is specifically to establish **quantified baseline performance** before moving to more computationally expensive approaches. By measuring classical model performance, the incremental value of XGBoost GPU and DistilBERT can be empirically demonstrated.

### **14.1 Training Strategy Overview**

All classical models share the same protocol:
1. Receive the identical `X_train_tfidf` (sparse, 15,000 features) and `y_train` arrays.
2. Fit (train) the model on these arrays.
3. Generate predictions on the held-out `X_test_tfidf`.
4. Compute classification metrics: Accuracy, Precision, Recall, F1-Score (weighted).
5. Generate a confusion matrix.
6. Record all metrics for the comparative analysis table.

### **14.2 Stratified Train-Test Split**

Before training any model, the processed dataset was partitioned into training and testing subsets using scikit-learn's `train_test_split`:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean,           # Cleaned text strings
    y,                 # Binary labels (0=Safe, 1=Abusive)
    test_size=0.20,    # 20% held out for testing
    random_state=42,   # Fixed seed for reproducibility
    stratify=y         # Maintain class ratio in both splits
)
```

**Dataset partition sizes:**
- **Training set:** 19,811 tweets (80%) → Used to train all models
- **Testing set:** 4,953 tweets (20%) → Used only for final evaluation (never seen during training)
- Both splits maintain the 83.2% / 16.8% Abusive / Safe ratio due to stratification.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 14.1 — Stratified Train-Test Split Diagram                          │
│                                                                              │
│   [Insert Split Diagram Here]                                                │
│                                                                              │
│   Full dataset (24,764 samples)                                              │
│      │                                                                       │
│      ├── Training Set (80% = 19,811 samples)                                 │
│      │       ├── Abusive: 16,500 (83.2%)                                     │
│      │       └── Safe: 3,311 (16.8%)                                         │
│      │                                                                       │
│      └── Test Set (20% = 4,953 samples)                                      │
│              ├── Abusive: 4,121 (83.2%)                                      │
│              └── Safe: 832 (16.8%)                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: Stratified splitting ensures the minority Safe class (16.8%) is
 proportionally represented in both training and test sets.*
```

---

### **14.3 Logistic Regression**

Despite its name, Logistic Regression is a **classification** algorithm, not a regression one. It models the probability of belonging to a class using the logistic (sigmoid) function, which maps any real-valued input to a probability in [0, 1].

**Mathematical Model:**
$$P(\text{Abusive} | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where $\mathbf{x}$ is the 15,000-dimensional TF-IDF feature vector, $\mathbf{w}$ is the learned weight vector, and $b$ is the bias term.

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(
    max_iter=1000,   # Maximum iterations for solver convergence
    C=1.0,           # Regularization strength (inverse)
    solver='lbfgs',  # Limited-memory BFGS solver for sparse data
    n_jobs=-1        # Use all CPU cores
)
lr_model.fit(X_train_tfidf, y_train)
```

**Why Logistic Regression is a Strong Baseline for NLP:**
For sparse, high-dimensional text features (like TF-IDF), logistic regression is particularly effective because each feature dimension (word) is relatively independent. The linear decision boundary in 15,000-dimensional feature space is surprisingly expressive for text classification. Its learned coefficients are also directly interpretable: the most positive weight coefficients correspond to the words most predictive of the Abusive class.

### **14.4 Multinomial Naive Bayes**

The Naive Bayes classifier applies Bayes' Theorem under the "naive" assumption that all features are conditionally independent given the class label:

$$P(\text{Abusive} | \mathbf{x}) \propto P(\text{Abusive}) \times \prod_{i=1}^{n} P(x_i | \text{Abusive})$$

**Implementation:**
```python
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB(alpha=0.1)  # Laplace smoothing parameter
nb_model.fit(X_train_tfidf, y_train)
```

The `alpha=0.1` parameter implements Laplace smoothing — preventing zero-probability estimates for words never seen in a particular class during training (which would make the entire product zero).

**Characteristics:** Despite its simplistic independence assumption (clearly violated in natural language — words are not independent), Naive Bayes performs remarkably well on text classification tasks. It is extremely fast to train and predict, requiring a single pass over the data to compute the conditional probability tables.

**Limitation:** Because it treats each word independently, it cannot capture phrase-level patterns. "Not good" is treated identically to seeing "not" and "good" separately — the negation relationship is lost.

### **14.5 Linear Support Vector Classifier (LinearSVC)**

Support Vector Machines seek the **optimal separating hyperplane** — the decision boundary in feature space that maximizes the margin between the two classes. A larger margin generally means better generalization to unseen data.

For the high-dimensional sparse TF-IDF feature space (15,000 dimensions), the hyperplane that separates Abusive from Safe tweets must be found in a 15,000-dimensional space. `LinearSVC` uses the `liblinear` library to efficiently solve this optimization problem for sparse matrices.

**Implementation:**
```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
svc_model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))
svc_model.fit(X_train_tfidf, y_train)
```

Note: `CalibratedClassifierCV` wraps `LinearSVC` to enable probability output (LinearSVC itself produces only binary decisions, not probabilities). This wrapper applies Platt scaling post-hoc.

**Why LinearSVC Performs Well on NLP Tasks:** With sufficient features (15,000) and a linear kernel, SVM can effectively separate the high-dimensional sparse vectors that represent abusive vs. safe text. LinearSVC is specifically designed for sparse data and is often the fastest of the classical models to train on large TF-IDF matrices.

### **14.6 Random Forest Classifier**

Random Forest is an ensemble method that builds many independent decision trees in parallel, each trained on a random bootstrap sample of the data and a random subset of features. The final prediction is determined by majority vote across all trees.

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,    # Number of trees
    max_features='sqrt', # Each tree uses sqrt(15000) ≈ 122 features
    n_jobs=-1,           # Parallel training on all CPU cores
    random_state=42
)
rf_model.fit(X_train_tfidf, y_train)
```

**Important Note on Random Forest Performance with TF-IDF:**
Decision trees and random forests generally perform **worse** than linear models on sparse TF-IDF features. The reason: each individual tree can only use `sqrt(15000) ≈ 122` features per split — a tiny fraction of the informative vocabulary. TF-IDF features are highly correlated (many abuse-related words appear together), but random subsampling breaks these correlations. This hypothesis was confirmed in our results: Random Forest underperformed Logistic Regression on this dataset, which is the expected and documented finding in the NLP literature.

### **14.7 k-Fold Cross-Validation**

To provide a more reliable performance estimate than a single train-test split, **5-fold cross-validation** was applied to the training set:

```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    model,
    X_train_tfidf,
    y_train,
    cv=5,           # 5 folds
    scoring='f1',   # Primary metric
    n_jobs=-1
)
print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

Cross-validation divides the training set into 5 equal "folds." In each round, 4 folds are used for training and 1 fold for validation. The model is trained and evaluated 5 times, and the average and standard deviation of F1-Scores across all 5 rounds are reported. A low standard deviation (< 0.01) indicates stable, consistent performance that generalizes well.

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 14.2 — 5-Fold Cross-Validation Process Diagram                     │
│                                                                              │
│   [Insert Horizontal Strip Diagram Showing Fold Rotation]                    │
│                                                                              │
│   Show 5 rows, each representing one fold iteration:                         │
│   - Fold 1: [VAL][TRAIN][TRAIN][TRAIN][TRAIN] → F1 score                    │
│   - Fold 2: [TRAIN][VAL][TRAIN][TRAIN][TRAIN] → F1 score                    │
│   - Fold 3: [TRAIN][TRAIN][VAL][TRAIN][TRAIN] → F1 score                    │
│   - Fold 4: [TRAIN][TRAIN][TRAIN][VAL][TRAIN] → F1 score                    │
│   - Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][VAL] → F1 score                    │
│   - Final: Average ± Std Dev of all 5 scores                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: In 5-fold cross-validation, each data point is used for validation
 exactly once, providing a robust, low-variance performance estimate.*
```

---

### **14.8 Classical Model Results Summary**

After training all four classical models and evaluating them on the held-out 20% test set, the following performance was recorded:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---|---|---|---|---|
| Logistic Regression | 92.8% | 0.931 | 0.928 | 0.929 | ~45 sec (CPU) |
| Multinomial Naive Bayes | 89.4% | 0.897 | 0.894 | 0.893 | ~2 sec (CPU) |
| LinearSVC | 93.2% | 0.934 | 0.932 | 0.933 | ~38 sec (CPU) |
| Random Forest | 88.9% | 0.891 | 0.889 | 0.888 | ~8 min (CPU) |
| **XGBoost (GPU)** | **94.1%** | **0.942** | **0.941** | **0.936** | **~90 sec (GPU)** |

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   FIGURE 14.3 — Classical Model F1-Score Comparison Bar Chart                │
│                                                                              │
│   [Insert Grouped Bar Chart Here]                                            │
│                                                                              │
│   X-axis: Model Name (LR, NB, SVC, RF, XGBoost GPU)                         │
│   Y-axis: F1-Score (0.86 to 0.96)                                            │
│   Color: Each model a different color bar                                    │
│   Add horizontal dashed line at DistilBERT's 0.9642 for comparison          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
*Caption: F1-Score comparison across all classical models and XGBoost GPU.
 DistilBERT (horizontal line) provides the target benchmark for deep learning.*
```

---

**Key Findings from Classical Training:**
1. LinearSVC and Logistic Regression both achieved excellent baseline performance (~93% F1), confirming the expectation that linear models are strong on sparse TF-IDF feature spaces.
2. Naive Bayes, despite its speed advantage, falls short due to its independence assumption and inability to handle feature correlations.
3. Random Forest underperformed as predicted — decision tree ensembles are not well-suited to very sparse, high-dimensional text features.
4. XGBoost GPU, while operating on the same TF-IDF features as the classical models, surpassed them by 0.3–4.8% F1 through its iterative error-correction boosting mechanism.

### **14.9 Transition to GPU-Accelerated Training**

The classical models have established that TF-IDF feature representations can achieve ~93% F1. However, all these models share a fundamental architectural limitation: they cannot understand word context. The word "sick" has one TF-IDF representation regardless of whether the surrounding sentence reads "That performance was sick!" (positive slang) or "That behavior makes me sick" (disgust/criticism).

To surpass this ceiling, the project moves to **transformer-based deep learning** in Part 3, where contextual word representations are computed through multi-head self-attention — a qualitatively different approach to text understanding.

The best XGBoost GPU model was saved (`models/xgboost_gpu.json`) and deployed in the Flask application as the "Classical" model option, providing a fast, lightweight alternative to DistilBERT for users who prioritize inference speed over maximum accuracy.

---

*End of Part 2. Part 3 covers the advanced GPU training stages: XGBoost detailed configuration, DistilBERT fine-tuning with PyTorch, the complete model evaluation framework with confusion matrix analysis, the multimodal extensions (Whisper voice, CLIP image analysis, emoji detection), and the full Flask web application deployment as an Instagram-like social media clone.*
