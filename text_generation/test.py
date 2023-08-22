from transformers import BertTokenizer, BertForMaskedLM
import torch

# chargé le modèle pré-entrainé et le tokenizer
model_name = 'bert-base-uncased'
tokenizer =  BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# préparer le texte d'entré
text = "[CLS] I want to generate some text [SEP]"

# tokenizer le texte
tokenized_text = tokenizer.tokenize(text)

# Convertir les tokens en identifiants numériques 
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Générer des prédictions de texte masqué

segments_ids = [0] * len(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    predictions = outputs[0]

# Extraire les prédictions pour le texte masqué 
masked_index = tokenized_text.index("[MASK]")
predicted_token = torch.argmax(predictions[0, masked_index]).item()
predicted_text = tokenizer.convert_ids_to_tokens([predicted_token])[0]

# Afficher le texte généré 
generated_text = text.replace("[MASK]", predicted_text)
print(generated_text)

