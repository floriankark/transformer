#%%
import matplotlib.pyplot as plt
import numpy as np


# Laden Sie die Daten
train_loss = np.load('/Users/floriankark/Downloads/loss_list.npy')
valid_loss = np.load('/Users/floriankark/Downloads/valid_loss_list.npy')

# Erstellen Sie ein Array mit der gleichen Länge wie train_loss für die x-Werte der Validierungsverluste
x_valid = np.linspace(0, len(train_loss)-1, num=len(valid_loss))

# Strecken Sie die Validierungsverluste auf die Länge der Trainingsverluste
valid_loss_stretched = np.interp(np.arange(len(train_loss)), x_valid, valid_loss)

# Erstellen Sie ein Diagramm
plt.figure(figsize=(10, 6))

# Zeichnen Sie die Daten
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss_stretched, label='Validation loss')

# Achsenbeschriftungen
plt.xlabel('Steps')
plt.ylabel('Loss')

# Legende
plt.legend()

# Diagramm anzeigen
plt.show()

# %%
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Laden Sie die Daten
with open('/Users/floriankark/Downloads/translations_base_test.json') as f:
    data = json.load(f)

# Liste zur Speicherung der BLEU-Scores
scores = []

# Berechnen Sie den BLEU-Score für jeden Eintrag
for entry in data:
    source = word_tokenize(entry['source'])
    correct = [word_tokenize(entry['correct'])]  # Die Referenz muss eine Liste von Listen sein
    generated = word_tokenize(entry['generated'])

    score = sentence_bleu(correct, generated)
    scores.append(score)

# Berechnen Sie den Durchschnitt der BLEU-Scores
average_score = sum(scores) / len(scores)

print(f"Durchschnittlicher BLEU-Score: {average_score}")
# %%
