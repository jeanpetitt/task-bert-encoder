from transformers import BertModel, BertTokenizer
import torch


# chargé le modèle pré-entrainé
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)


# préparer les données à compresser en les encodant avec le tokenizer
text = "voici mon text"
encoded_input = tokenizer(text, return_tensors='pt')

# passez les données encodées au modèle BERT pour obtenir la
# obtenir les caractéristiques du texte
with torch.no_grad():
    output = model(**encoded_input)
    features = output.last_hidden_state
    