from transformers import BertTokenizer, BertForPreTraining
import torch

# Charger le tokenizer et le modèle pré-entraîné
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForPreTraining.from_pretrained(model_name)

# Prétraiter le texte en utilisant le tokenizer 
text = "Votre texte à encoder"
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor([input_ids])

# Utiliser le modèle BERT pour apprendre à représenter le texte
outputs = model(input_ids)
encoded_text = outputs[0]