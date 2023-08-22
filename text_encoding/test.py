from transformers import BertTokenizer, BertModel
import torch

# Charger le modèle pré-entraîné et le tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Prétraiter le texte en utilisant le tokenizer 
text = "Votre texte à encoder"
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor([input_ids])

# Encoder le texte en utilisant le modèle BERT
with torch.no_grad():
    outputs = model(input_ids)
    encoded_layers = outputs[0]
    
# Récupérer la représentation vectorielle du texte encodé
encoded_text = encoded_layers.squeeze(0)
