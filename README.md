# AINA (Antibacterial Inorganic Nanoparticles) Dataset

This repository contains a novel dataset for automated screening in the domain of Antibacterial Nanoparticles and Python scripts for the best performing two models from #Paper Name#. You can find repository structure below:
```
├── env/
│   ├── bertenv.yml      # Conda environment for Biobert model
│   └── svm_tfidf_env.yml   # Conda environment for SVM with TF-IDF
├── models/
│   ├── biobert.py          # Python script for running Biobert model
│   └── linear_svm_tfidf.py # Python script for running SVM-TF-IDF model
├── data/
│   ├── antimicrobial_nanoparticles_test_data.csv    # Test data
│   └── antimicrobial_nanoparticles_train_data.csv   # Train data
└── README.md               # Project documentation

--
```
You can find packages and libraries required for running a model in the env files. If you want to run the model please create the corresponding conda environment:
```
conda env create -f env/model_preferred_env.yml
conda activate model_preferred_env
```
Please note that data files do not have a column for abstract. Due to licensing restrictions, we had to remove the abstract data. Thus, you have to find abstracts, if you want to make use of the models. Please attach abstracts to the main datasets under a column named 'Abstract'.

After activating environments and attaching abstracts, you can run the corresponding model. 
```
python model_preferred.py
```

## Credits

The **AINA dataset** was developed under the *BlueSky Initiative* at the **University of Michigan College of Engineering**. Annotation Team is as follows:

Brendan Knittle, Luke Wesseln, Tianjie Qiu, Alienna Glenn, Maxwell Topping, Emine Sümeyra Turalı-Emre, Sergio Q. Sanchez, Shivani Kozaker, Nicole Sorensen, William Brown, Shamalee Goonetilleke, Christopher Altheim, J. Scott VanEpps





