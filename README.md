# AINA (Antibacterial Inorganic Nanoparticles) Dataset

This repository contains a novel dataset for automated screening in the domain of Antibacterial Nanoparticles and Python scripts for the best performing two models from Automated Screening of Antibacterial Nanoparticle Literature: Dataset Curation and Model Evaluation. You can find the repository structure below:

```
├── envs/
│   ├── bertenv.yml             
│   └── svm_tfidf_env.yml      
├── models/
│   ├── biobert.py             
│   └── linear_svm_tfidf.py   
├── data/
│   ├── antimicrobial_nanoparticles_train_data.csv   # Train data
│   └── antimicrobial_nanoparticles_test_data.csv    # Test data
├── retrieve_abstracts.py       
└── README.md                
```

## Dataset


Due to licensing restrictions, abstracts are not included in the repository. You must retrieve them yourself using the provided retriever script before running any models.

## Reproducing Abstracts

We provide `retrieve_abstracts.py` to automatically fetch abstracts for all papers using their DOIs.

### Requirements

```bash
pip install requests pandas tqdm
```

### Usage

Run the retriever on both the train and test sets:

```bash
python retrieve_abstracts.py --input data/antimicrobial_nanoparticles_train_data.csv
python retrieve_abstracts.py --input data/antimicrobial_nanoparticles_test_data.csv
```

This will update the CSV files in place, adding an `Abstract` column as required by the models.

You can get better API rate limits from Crossref and Pubmed by providing your email but it is optional.
```bash
python retrieve_abstracts.py --input data/antimicrobial_nanoparticles_train_data.csv --email you@example.com
python retrieve_abstracts.py --input data/antimicrobial_nanoparticles_test_data.csv --email you@example.com
```

## Running the Models

Once abstracts have been retrieved, create the conda environment for your preferred model and run it:

```bash
# BioBERT
conda env create -f envs/bertenv.yml
conda activate bertenv
python models/biobert.py

# SVM with TF-IDF
conda env create -f envs/svm_tfidf_env.yml
conda activate svm_tfidf_env
python models/linear_svm_tfidf.py
```

## Credits

The **AINA dataset** was developed under the *BlueSky Initiative* at the **University of Michigan College of Engineering**. Annotation team:

Brendan Knittle, Luke Wesseln, Tianjie Qiu, Alienna Glenn, Maxwell Topping, Emine Sümeyra Turalı-Emre, Sergio Q. Sanchez, Shivani Kozaker, Nicole Sorensen, William Brown, Shamalee Goonetilleke, Christopher Altheim, J. Scott VanEpps
