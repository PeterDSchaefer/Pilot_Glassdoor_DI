########################################################################################################################
########################################################################################################################
#### CALCULATE DIVERSITY AND INCLUSION MEASURES FROM GLASSDOOR REVIEWS
########################################################################################################################
########################################################################################################################
#### AUTHOR: Peter Schaefer
#### THIS VERSION: 2024, July 23rd
#### OVERVIEW OF STEPS:
#### (1) Load all Glassdoor reviews available in main folder
#### (2) Preprocess and parse the reviews using the Stanford CoreNLP Package: Segmentation & tokenization, Lemmatization, Named Entity Recognition, Dependency Parsing
#### (3) Cleaning parsed text and learning phrases: phraser module of the gensim library
#### (4) Word embedding, word2vec, and model training
#### (5) From the trained model, generate the DI dictionary
#### (6) Scoring DI


########################################################################################################################
### STEP 0: IMPORT PACKAGES AND DEFINE PARAMETERS
########################################################################################################################
import os
import stanza
import spacy
import string
import pandas as pd
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ensure that Stanza models are downloaded
stanza.download('en')
# Load SpaCy model
spacy_nlp = spacy.load("en_core_web_sm")

# Setting up the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative file paths
RAW_DATA_FOLDER = os.path.join(BASE_DIR, '04_processed_gd_reviews', '01_txt')
OUTPUT_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '01_joint_gdreviews.txt')
PREPROCESSED_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '02_preprocessed_gdreviews.txt')
CLEANED_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '03_cleaned_gdreviews.txt')
WORD2VEC_MODEL_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps','04_word2vec_model')
DICTIONARY_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '05_di_dictionary.txt')
DICTIONARY_REVISED_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '06_di_dictionary_revised.txt')
CSV_OUTPUT_FILE = os.path.join(BASE_DIR, '04_processed_gd_reviews', '03_text_processing_steps', '07_di_measures.csv')

# Steps to perform
PERFORM_COMPILE_REVIEWS = False
PERFORM_PREPROCESSING = False
PERFORM_CLEANING = False
PERFORM_TRAINING = True
PERFORM_DICTIONARY_CREATION = True
PERFORM_DI_MEASUREMENT = False

# Parameters for the steps
PREPROC_NER = True          # Named Entity Recognition - yes or no?
PREPROC_DP = True           # Dependency parsing - yes or no?
PROC_SPACY = False          # Use SpaCy if True, else use Stanza

# Seed words for DI dictionary
SEED_WORDS = ["diversity", "inclusion", "equal_opportunity", "belonging", "engagement", "representation", "inclusive_culture", "diverse_skills"]

########################################################################################################################
#### STEP 1: COMPILE REVIEWS AVAILABLE IN THE RAW DATA FOLDER
########################################################################################################################
def compile_reviews(root_folder, output_file):
    """
    Compiles the content of all files ending with '_all_reviews.txt' in the root folder into a single output file.

    Arguments:
    root_folder (str): The root directory containing the text files.
    output_file (str): The path to the output file where compiled content will be saved.
    """
    all_reviews_content = []

    # Traverse the directory structure
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_all_reviews.txt'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_reviews_content.append(f.read())

    # Write the compiled content to the output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(all_reviews_content))

    print(f'Compiled reviews have been saved to {output_file}')


########################################################################################################################
#### STEP 2: PREPROCESS THE TEXT USING STANZA
########################################################################################################################
def preprocess_text(input_file, output_file, chunk_size=10000):
    """
    Preprocesses the text in the input file using the chosen NLP tool for sentence segmentation, tokenization, lemmatization, NER, and dependency parsing.

    Arguments:
    input_file (str): The path to the input text file.
    output_file (str): The path to the output file where preprocessed text will be saved.
    chunk_size (int): The number of characters to process at a time.
    """
    if PROC_SPACY:
        preprocess_text_spacy(input_file, output_file, chunk_size)
    else:
        preprocess_text_stanza(input_file, output_file, chunk_size)

def preprocess_text_spacy(input_file, output_file, chunk_size):
    """
    Preprocesses the text using SpaCy for sentence segmentation, tokenization, lemmatization, NER, and dependency parsing.

    Arguments:
    input_file (str): The path to the input text file.
    output_file (str): The path to the output file where preprocessed text will be saved.
    chunk_size (int): The number of characters to process at a time.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    num_chunks = len(text) // chunk_size + 1
    preprocessed_sentences = []

    for i in tqdm(range(num_chunks), desc="Processing chunks with SpaCy"):
        chunk = text[i * chunk_size: (i + 1) * chunk_size]
        doc = spacy_nlp(chunk)

        for sentence in doc.sents:
            tokens = []
            for token in sentence:
                lemma = token.lemma_

                if PREPROC_NER and token.ent_type_:
                    lemma = f'<{token.ent_type_}>'

                tokens.append(lemma)

            if PREPROC_DP:
                for token in sentence:
                    if token.dep_ in ('compound', 'mwe'):
                        head = token.head
                        compound = f'{head.text}_{token.text}'
                        tokens = [compound if w == head.text or w == token.text else w for w in tokens]

            preprocessed_sentences.append(' '.join(tokens))

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(preprocessed_sentences))

def preprocess_text_stanza(input_file, output_file, chunk_size):
    """
    Preprocesses the text using Stanza for sentence segmentation, tokenization, lemmatization, NER, and dependency parsing.

    Arguments:
    input_file (str): The path to the input text file.
    output_file (str): The path to the output file where preprocessed text will be saved.
    chunk_size (int): The number of characters to process at a time.
    """
    processors = 'tokenize,lemma,pos'
    if PREPROC_NER:
        processors += ',ner'
    if PREPROC_DP:
        processors += ',depparse'

    nlp = stanza.Pipeline('en', processors=processors, use_gpu=False)

    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    num_chunks = len(text) // chunk_size + 1
    preprocessed_sentences = []

    for i in tqdm(range(num_chunks), desc="Processing chunks with Stanza"):
        chunk = text[i * chunk_size: (i + 1) * chunk_size]
        doc = nlp(chunk)

        for sentence in doc.sentences:
            tokens = []
            for token in sentence.tokens:
                word = token.words[0]

                lemma = word.lemma

                if PREPROC_NER and token.ner != 'O':
                    lemma = f'<{token.ner}>'

                tokens.append(lemma)

            if PREPROC_DP:
                for word in sentence.words:
                    if word.deprel in ('compound', 'mwe'):
                        head = sentence.words[word.head - 1]
                        compound = f'{head.text}_{word.text}'
                        tokens = [compound if w == head.text or w == word.text else w for w in tokens]

            preprocessed_sentences.append(' '.join(tokens))

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(preprocessed_sentences))


########################################################################################################################
#### STEP 3: CLEAN THE TEXT AND LEARN PHRASES USING GENSIM
########################################################################################################################
#### Cleaning parsed text and learning phrases: phraser module of the gensim library

def clean_and_learn_phrases(input_file, output_file):
    """
    Cleans the text and learns two- and three-word phrases using Gensim's Phrases module.

    Arguments:
    input_file (str): The path to the preprocessed text file.
    output_file (str): The path to the output file where cleaned and phrase-learned text will be saved.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in tqdm(lines, desc="Cleaning text"):
        tokens = line.strip().split()
        cleaned_tokens = [
            token for token in tokens
            if token not in string.punctuation and
               token.lower() not in STOPWORDS and
               len(token) > 1
        ]
        cleaned_lines.append(cleaned_tokens)

    # Learn two-word phrases
    phrases = Phrases(cleaned_lines, min_count=1, threshold=10)
    bigram = Phraser(phrases)
    bigram_lines = [bigram[line] for line in cleaned_lines]

    # Learn three-word phrases
    phrases = Phrases(bigram_lines, min_count=1, threshold=10)
    trigram = Phraser(phrases)
    trigram_lines = [trigram[line] for line in bigram_lines]

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for line in trigram_lines:
            out_file.write(' '.join(line) + '\n')

########################################################################################################################
#### STEP 4: MODEL TRAINING WITH WORD EMBEDDING (WORD2VEC)
########################################################################################################################
#### Training a Word2Vec model with the cleaned text
def train_word2vec_model(input_file, model_file):
    """
    Trains a Word2Vec model using the cleaned text.

    Arguments:
    input_file (str): The path to the cleaned text file.
    model_file (str): The path to the output file where the trained Word2Vec model will be saved.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file]

    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=5,
        workers=4,
        sg=1,  # Use skip-gram
        negative=5,  # Negative sampling
        epochs=20
    )

    model.save(model_file)
    print(f'Trained Word2Vec model has been saved to {model_file}')


########################################################################################################################
### STEP 5: CREATE DICTIONARY FOR MEASURING "DIVERSITY AND INCLUSION"
########################################################################################################################
### Creating a context-specific dictionary using the trained Word2Vec model

def create_di_dictionary(model_file, seed_words, dictionary_file, top_n=100):
    """
    Creates a context-specific dictionary for measuring "diversity and inclusion" using the trained Word2Vec model.

    Arguments:
    model_file (str): The path to the trained Word2Vec model file.
    seed_words (list): The list of seed words for DI dictionary.
    dictionary_file (str): The path to the output file where the DI dictionary will be saved.
    top_n (int): The number of top associated words to include for each seed word.
    """
    model = Word2Vec.load(model_file)
    di_words = set()

    for seed in seed_words:
        if seed in model.wv.key_to_index:
            similar_words = model.wv.most_similar(seed, topn=top_n)
            di_words.update([word for word, _ in similar_words])
        else:
            print(f'Seed word "{seed}" not found in the vocabulary.')

    with open(dictionary_file, 'w', encoding='utf-8') as file:
        for word in di_words:
            file.write(f'{word}\n')

    print(f'DI dictionary has been saved to {dictionary_file}')

########################################################################################################################
### STEP 6: MEASURE DIVERSITY AND INCLUSION
########################################################################################################################
### Measuring diversity using the revised DI dictionary and TF_IDF
def load_phrases(cleaned_file):
    """
    Loads the bigram and trigram phrasers from the cleaned text file.

    Arguments:
    cleaned_file (str): The path to the cleaned text file.

    Returns:
    tuple: A tuple containing the bigram and trigram phrasers.
    """
    with open(cleaned_file, 'r', encoding='utf-8') as file:
        lines = [line.strip().split() for line in file]

    bigram_phrases = Phrases(lines, min_count=1, threshold=10)
    trigram_phrases = Phrases(bigram_phrases[lines], min_count=1, threshold=10)
    bigram_phraser = Phraser(bigram_phrases)
    trigram_phraser = Phraser(trigram_phrases)

    return bigram_phraser, trigram_phraser

def measure_diversity(root_folder, revised_dictionary_file, cleaned_file, output_csv_file):
    """
    Measures the diversity using the revised DI dictionary.

    Arguments:
    root_folder (str): The root directory containing subfolders with text files.
    revised_dictionary_file (str): The path to the revised DI dictionary file.
    cleaned_file (str): The path to the cleaned text file used to train the model.
    output_csv_file (str): The path to the output CSV file where the diversity measures will be saved.
    """
    # Load the revised dictionary
    with open(revised_dictionary_file, 'r', encoding='utf-8') as file:
        di_words = set(line.strip().lower() for line in file.readlines())

    # Load bigram and trigram phrasers
    bigram_phraser, trigram_phraser = load_phrases(cleaned_file)

    # Initialize data structures
    company_year_measures = {}

    # Traverse the directory structure
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.txt') and '_all_reviews.txt' not in file:
                parts = file.split('_')
                if len(parts) == 2 and parts[1].endswith('.txt'):
                    name = parts[0]
                    year = parts[1].split('.')[0]
                    file_path = os.path.join(subdir, file)

                    # Preprocess the text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()

                    # Tokenize and remove stopwords
                    tokens = [token for token in text.split() if token not in STOPWORDS and token not in string.punctuation]

                    # Apply bigram and trigram models
                    bigram_tokens = bigram_phraser[tokens]
                    trigram_tokens = trigram_phraser[bigram_tokens]

                    # Join tokens to create a document
                    document = ' '.join(trigram_tokens)

                    # Calculate tf-idf scores
                    vectorizer = TfidfVectorizer(vocabulary=di_words, lowercase=True, ngram_range=(1, 3))  # Include bi- and trigrams
                    tfidf_matrix = vectorizer.fit_transform([document])
                    tfidf_scores = tfidf_matrix.sum(axis=1).tolist()[0][0]

                    # Update company-year measures
                    if (year, name) not in company_year_measures:
                        company_year_measures[(year, name)] = []

                    company_year_measures[(year, name)].append(tfidf_scores)

    # Calculate average measures for each company-year
    average_measures = {
        (year, name): np.mean(scores)
        for (year, name), scores in company_year_measures.items()
    }

    # Save the results to a CSV file
    with open(output_csv_file, 'w', encoding='utf-8') as csv_file:
        csv_file.write('Year,CompanyID,DiversityMeasure\n')
        for (year, name), measure in average_measures.items():
            csv_file.write(f'{year},{name},{measure}\n')

    print(f'Diversity measures have been saved to {output_csv_file}')


if __name__ == '__main__':
    if PERFORM_COMPILE_REVIEWS:
        # Compile reviews from the specified directory
        compile_reviews(RAW_DATA_FOLDER, OUTPUT_FILE)
        print(f'Compiled Q&A parts have been saved to {OUTPUT_FILE}')

    if PERFORM_PREPROCESSING:
        # Preprocess the compiled Q&A file
        preprocess_text(OUTPUT_FILE, PREPROCESSED_FILE)
        print(f'Preprocessed text has been saved to {PREPROCESSED_FILE}')

    if PERFORM_CLEANING:
        # Clean the preprocessed text and learn phrases
        clean_and_learn_phrases(PREPROCESSED_FILE, CLEANED_FILE)
        print(f'Cleaned text and learned phrases have been saved to {CLEANED_FILE}')

    if PERFORM_TRAINING:
        # Train the Word2Vec model using the cleaned text
        train_word2vec_model(CLEANED_FILE, WORD2VEC_MODEL_FILE)
        print(f'Trained Word2Vec model has been saved to {WORD2VEC_MODEL_FILE}')

    if PERFORM_DICTIONARY_CREATION:
        # Create DI dictionary using the trained Word2Vec model
        create_di_dictionary(WORD2VEC_MODEL_FILE, SEED_WORDS, DICTIONARY_FILE)
        print(f'DI dictionary has been saved to {DICTIONARY_FILE}')

    if PERFORM_DI_MEASUREMENT:
        # Measure diversity using the revised DI dictionary
        measure_diversity(RAW_DATA_FOLDER, DICTIONARY_REVISED_FILE, CLEANED_FILE, CSV_OUTPUT_FILE)
        print(f'Diversity measures have been saved to {CSV_OUTPUT_FILE}')
