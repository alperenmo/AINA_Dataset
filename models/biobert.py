
import os
from huggingface_hub import login
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import EarlyStoppingCallback
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(MODELS_DIR,'..'))
DATA_DIR = os.path.join(REPO_DIR,'data')
TEST_PATH = os.path.join(DATA_DIR,'antimicrobial_nanoparticles_test_data.csv')
TRAIN_PATH = os.path.join(DATA_DIR,'antimicrobial_nanoparticles_train_data.csv')


# Please enter your token
HUGGINGFACE_TOKEN = ""

login(token=HUGGINGFACE_TOKEN)

def load_data():
    df_train = pd.read_csv('TRAIN_PATH')  
    df_test = pd.read_csv('TEST_PATH')    

    train_texts = df_train['Title'] + " " + df_train['Abstract']
    test_texts = df_test['Title'] + " " + df_test['Abstract']
    
    train_labels = df_train['Decision'].apply(lambda x: 1 if x == 'include' else 0)
    test_labels = df_test['Decision'].apply(lambda x: 1 if x == 'include' else 0)

    return train_texts, test_texts, train_labels, test_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != 'cuda':
    raise RuntimeError("Cuda Not Available")
else:
    print(f"Cuda Available")

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

train_texts, test_texts, train_labels, test_labels = load_data()

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

train_labels = torch.tensor(train_labels.values)
test_labels = torch.tensor(test_labels.values)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  
    greater_is_better=False,
    report_to="none"
)

def compute_metrics(p):
    preds = p.predictions
    labels = p.label_ids
    preds = torch.argmax(torch.tensor(preds), axis=1).numpy()
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

for i in range(5):
    print(f"\n===== Training Run {i+1} =====")

    model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print(f"\n--- Evaluation after run {i+1} ---")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    predictions_output = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions_output.predictions), axis=1).numpy()
    true_labels = predictions_output.label_ids
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=["exclude", "include"]))

    results_df = pd.DataFrame({
    'Index': range(len(true_labels)),
    'Expected': ['include' if label == 1 else 'exclude' for label in true_labels],
    'Predicted': ['include' if pred == 1 else 'exclude' for pred in preds]
})
    results_df['Match'] = results_df['Expected'] == results_df['Predicted']

    xlsx_path = REPO_DIR
    results_df.to_excel(xlsx_path, index=False)