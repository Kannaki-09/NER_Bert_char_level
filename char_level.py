import string
from transformers import AutoTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
import torch.nn as nn

label_list = ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-SIZE", "I-SIZE"]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}


all_chars = list(string.ascii_letters + string.digits + "_")
char_to_id = {c: i + 1 for i, c in enumerate(all_chars)}
char_to_id["<pad>"] = 0
MAX_CHAR_LEN = 15

def char_encode_token(token):
    token_lower = token.lower()
    char_ids = [char_to_id.get(c, 0) for c in token_lower [: MAX_CHAR_LEN]]
    char_ids += [0] * (MAX_CHAR_LEN - len(char_ids))
    return char_ids

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast = True)

data = [
    {"tokens": ["1", "masala", "vada", "with", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["2", "idlis", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "pongal", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plate", "of", "puri", "with", "chutney"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "idli", "with", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["two", "pooris", "no", "masala"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "vada", "with", "less", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "coffee"], "labels": ["O", "B-FOOD"]},
    {"tokens": ["one", "filter", "coffee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["get", "me", "2", "masala", "dosas"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["i", "want", "hot", "pongal"], "labels": ["O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["4", "idlis", "and", "1", "vada"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["single", "vada", "with", "no", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plate", "of", "samosa", "with", "sauce"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["one", "hot", "idli"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["no", "spicy", "in", "dosa"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["extra", "sambar", "for", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["one", "pav", "bhaji"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["need", "strong", "tea"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["3", "medu", "vada", "with", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["no", "oil", "in", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["give", "me", "1", "poori", "with", "less", "salt"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["hot", "coffee", "please"], "labels": ["B-CUSTOMIZATION", "B-FOOD", "O"]},
    {"tokens": ["order", "one", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "hot", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["just", "one", "vada", "with", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "masala", "dosas", "less", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "no", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "hot", "coffee"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["please", "give", "me", "1", "samosa", "with", "extra", "chutney"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "pooris", "without", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "filter", "coffee", "no", "sugar"], "labels": ["O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "and", "idli"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["a", "plate", "of", "pongal", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "spicy", "masala", "dosa"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["i", "want", "one", "vada", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "half", "plate", "poori", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "three", "idlis", "with", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "a", "cup", "of", "strong", "coffee"], "labels": ["O", "O", "B-QUANTITY", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["i", "will", "take", "two", "masala", "dosas"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["get", "me", "a", "hot", "vada"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["no", "oil", "in", "my", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["can", "i", "have", "one", "poori", "without", "salt"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "with", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "no", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "a", "cup", "of", "tea"], "labels": ["O", "O", "B-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["get", "me", "strong", "filter", "coffee"], "labels": ["O", "O", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["two", "samosas", "no", "mint", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "hot", "puri"], "labels": ["O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["order", "4", "idlis", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "cold", "coffee"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["one", "plate", "medu", "vada", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "hot", "idlis", "no", "onion"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "cup", "strong", "tea"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["two", "dosas", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "coffee"], "labels": ["O", "B-FOOD"]},
    {"tokens": ["need", "one", "plate", "pongal", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "pooris", "no", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["four", "idlis", "with", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "hot", "masala", "dosa"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["a", "plate", "of", "samosa", "with", "sweet", "chutney"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "dosas", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "a", "bowl", "of", "pongal", "with", "ghee", "on", "top"], "labels": ["O", "O", "O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "poori", "less", "masala"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plain", "dosa", "no", "butter"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "and", "a", "vada", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "set", "dosa", "with", "no", "salt"], "labels": ["O", "O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["full", "meals", "with", "extra", "gravy"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["small", "portion", "of", "upma", "no", "ginger"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "single", "vada", "without", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "onion", "dosas", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "poha", "no", "mustard", "seeds"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "cup", "of", "coffee", "with", "less", "sugar"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "four", "idlis", "and", "no", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["medium", "size", "masala", "dosa", "without", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "a", "small", "portion", "of", "kesari"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["serve", "two", "vada", "pavs", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "get", "me", "three", "chapatis", "no", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "bowl", "of", "curd", "rice", "extra", "curd"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "double", "egg", "dosa", "with", "extra", "onions"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "samosa", "no", "mint", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "plate", "of", "lemon", "rice", "without", "nuts"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plain", "parathas", "with", "butter"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "vada", "no", "coconut", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "glass", "of", "lassi", "less", "sweet"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "pani", "puris", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "would", "like", "two", "biryani", "packets", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "single", "sandwich", "no", "sauce"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "half", "plate", "idiyappam", "extra", "coconut", "milk"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "with", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["two", "vada", "no", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "masala", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "plate", "of", "pongal", "without", "cashews"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "pav", "bhaji", "less", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "four", "veg", "rolls", "extra", "cheese"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["five", "jalebi", "with", "less", "sugar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "bowl", "of", "curd", "rice", "with", "no", "pickle"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "mango", "lassi", "not", "too", "sweet"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "plain", "roti", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["six", "chicken", "momoes", "extra", "dip"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "veg", "thali", "without", "curd"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "a", "half", "plate", "noodles", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "coffee", "less", "sugar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "glass", "of", "buttermilk", "no", "salt"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["seven", "paneer", "tikka", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "sandwiches", "without", "sauce"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "bottle", "water", "cold"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION"]},
    {"tokens": ["three", "pani", "puri", "no", "onions"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "cup", "tea", "no", "milk"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "vada", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "no", "spice", "please"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["half", "dosa", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "biryani", "without", "onions"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "samosas", "extra", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "cup", "of", "tea", "with", "more", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "sambar", "combo"], "labels": ["O", "B-FOOD", "I-FOOD", "I-FOOD"]},
    {"tokens": ["plain", "dosa", "with", "no", "butter"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["single", "vada", "with", "less", "salt"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "curd", "rice", "without", "mustard"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "veg", "rolls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["tea", "no", "milk"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "half", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["three", "vada", "no", "chilli"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plain", "uttapam", "extra", "chutney"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "no", "ghee"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "masala", "dosa", "without", "potato"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "vada", "with", "less", "chilli"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "extra", "coconut", "chutney"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "no", "onions"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "less", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["masala", "dosa", "without", "spice"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "cups", "coffee", "extra", "hot"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "plain", "dosa", "with", "no", "salt"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "tea", "with", "less", "sugar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["four", "vada", "extra", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "chutney", "with", "no", "coriander"], "labels": ["O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "five", "samosas", "without", "mint", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plain", "dosa", "add", "extra", "chutney"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "filter", "coffees"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["idli", "with", "extra", "gunpowder"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "dosa", "with", "no", "chutney"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "cup", "coffee", "no", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "three", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["tea", "without", "milk"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "me", "pongal", "extra", "pepper"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "vada", "less", "spicy"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["single", "plate", "idli", "without", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "masala", "dosa", "no", "onions"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "three", "vada", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "coffee", "no", "milk"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "a", "dosa", "extra", "crisp"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "idli", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["filter", "coffee", "with", "more", "sugar"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "half", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["three", "vada", "without", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "less", "salt"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "dosa", "extra", "butter"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["tea", "with", "less", "sugar"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "more", "chutney"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "dosa", "without", "ghee"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "four", "vada", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "cup", "coffee", "with", "no", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["dosa", "with", "extra", "chutney", "and", "sambar"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "plain", "idli"], "labels": ["O", "O", "B-FOOD", "I-FOOD"]},
    {"tokens": ["two", "vada", "without", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["filter", "coffee", "no", "sugar"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "2", "masala", "dosa", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "one", "plate", "pongal", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "five", "vada", "with", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "idli", "and", "more", "sambar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "i", "have", "a", "crispy", "dosa"], "labels": ["O", "O", "O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["please", "add", "two", "plates", "of", "poori"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["one", "pongal", "with", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["extra", "chutneys", "and", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["make", "that", "2", "crispy", "dosa"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["i", "want", "a", "large", "plate", "of", "pongal"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["one", "vada", "no", "chilli"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "you", "send", "idli", "x", "3", "with", "less", "oil"], "labels": ["O", "O", "O", "B-FOOD", "B-QUANTITY", "I-QUANTITY", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "extra", "sambar", "for", "pongal"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["please", "include", "2", "idlis", "and", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["need", "four", "vada", "no", "ghee", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "masala", "dosa", "with", "extra", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "some", "hot", "pongal", "and", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["i", "want", "idli", "without", "spicy", "chutney"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "idlis", "and", "2", "vada"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["please", "send", "2", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "idlis", "and", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "pongal", "without", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "dosa", "with", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "4", "idlis", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "one", "plate", "of", "poori", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "vadais", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "plates", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "idlis", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "masala", "dosa"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "I-FOOD"]},
    {"tokens": ["one", "plate", "idli", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "4", "dosa", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "vada", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "2", "idlis", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "3", "idlis", "with", "extra", "butter"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "pongal", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "dosas", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},{"tokens": ["order", "two", "plates", "of", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "get", "one", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "three", "idlis", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "vada", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "plates", "pongal", "without", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "dosa", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "idlis", "less", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "poori", "extra", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "three", "vadais", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "masala", "dosa", "without", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "two", "dosas", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "vada", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "poori", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "three", "idlis", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "pongal", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "dosa", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "idlis", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "with", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "three", "plates", "poori", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "masala", "dosa", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "plate", "vada", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "dosas", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "idli", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "idlis", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "vada", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "poori", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "with", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "vada", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "three", "dosas", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "idli", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "plates", "pongal", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "three", "idlis", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "have", "two", "masala", "dosas", "with", "extra", "butter"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "idli", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "plates", "of", "vada", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "plate", "pongal", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "idlis", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "dosas", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "idli", "no", "onion"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "vadais", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "sambar", "to", "my", "idli"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["I", "need", "two", "pooris", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "three", "vada", "with", "extra", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "1", "plate", "pongal", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "idlis", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "add", "less", "oil", "to", "vada"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "extra", "crispy", "poori"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["two", "idli", "no", "salt"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "one", "plate", "pongal", "with", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "three", "vadais", "with", "no", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "two", "dosa", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "idli", "with", "no", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "masala", "dosa", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "two", "idlis", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "need", "vada", "with", "extra", "sambar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "plates", "of", "pongal", "no", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "idli", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "two", "poori", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "idli", "with", "more", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "dosa", "without", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "vada", "with", "extra", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "one", "plate", "idli", "no", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "four", "idlis", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "masala", "dosa", "with", "no", "oil"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "idli", "with", "less", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "poori", "no", "oil"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "two", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "need", "three", "idlis", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "pongal", "extra", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "less", "oil", "to", "my", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["give", "four", "idlis", "without", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "three", "plates", "of", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "idli", "with", "no", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "one", "poori", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "plates", "pongal", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "chutney", "to", "vada"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["I", "want", "masala", "dosa", "with", "extra", "butter"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "poori", "without", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "idlis", "with", "no", "salt"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "vada", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "two", "masala", "dosa", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "idli", "no", "onion"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "pongal", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "three", "poori", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "vada", "without", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "idly", "with", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "send", "1", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "3", "pooriis", "less", "oil", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["can", "i", "have", "extra", "chutney", "and", "2", "idlis"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["send", "me", "one", "plate", "of", "masala", "dosaa"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-CUSTOMIZATION", "I-FOOD"]},
    {"tokens": ["i", "need", "small", "vada", "with", "extra", "spicy", "chutni"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "dossas", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "idly", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "plates", "pooris", "with", "extra", "sabji"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "2", "idly", "with", "more", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "dosa", "no", "ghee", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "3", "idlis", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "give", "1", "plate", "pongal", "with", "extra", "chutni"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "dossas", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "small", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "idly", "and", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "poori", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "2", "idlys", "with", "extra", "spicy", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "order", "one", "plate", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "extra", "chutney", "and", "2", "idly"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["give", "me", "3", "plates", "dosa", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "poori", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "2", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "one", "plate", "vada", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "give", "one", "plate", "pongal", "extra", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "dosa", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "small", "vada", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "idly", "with", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "send", "2", "masala", "dosas", "wit", "extra", "cheese"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "i", "have", "1", "idly", "with", "less", "spice"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "plate", "poori", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "4", "vada", "wth", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "dosa", "no", "ghee", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["want", "2", "idli", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "a", "smal", "vada", "no", "chilly"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "3", "masala", "dosa", "extra", "chutni"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1plate", "idly", "with", "no", "ginger"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "4", "dosaa", "less", "salt"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "and", "extra", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "1", "poori", "with", "extra", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "plates", "vada", "less", "spicy"], "labels": ["B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "one", "masaladosa", "with", "extra", "cheese"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "idly", "with", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "dossas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "idlies", "with", "little", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "1", "plate", "poory", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "vadais", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "3", "masala", "dossas", "pls"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "O"]},
    {"tokens": ["extra", "chutney", "and", "2", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["1", "plate", "poori", "with", "less", "salt"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "2", "small", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["want", "one", "masala", "dos", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "3", "idli", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "vada", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "plates", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "1", "masala", "dosa", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "2", "idli", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "poori", "less", "oil"], "labels": ["B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "3", "dosa", "with", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "idli", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "1", "vada", "with", "less", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "dosa", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "2", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "one", "poori", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "vada", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "idlies", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "masala", "dosa", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "2", "idlis", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "dosa", "less", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "extra", "chutni"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "pongal", "with", "no", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "masla", "dosa", "no", "oil"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "4", "idlies", "extra", "chutney", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["get", "me", "poori", "without", "onion"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["2", "dosas", "less", "oil", "more", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "1", "idli", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "vada", "extra", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "3", "idly", "wit", "no", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "pongal", "with", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["order", "4", "idlis", "wit", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "two", "plates", "of", "idli"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["get", "me", "dosa", "with", "extra", "sambhar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "3", "masala", "dosa", "no", "onions"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["4", "idlis", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "pooris", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["dosa", "without", "spice"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "poori", "add", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "small", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["add", "chutni", "for", "my", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["need", "5", "idlis", "extra", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "masala", "dosa", "no", "spicy"], "labels": ["O", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "2", "idlis", "wit", "sambhar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "idli", "less", "salt"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "vada", "no", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "masala", "dosaa", "xtra", "chutney"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "idli", "with", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "5", "poori", "no", "oil", "please"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["send", "me", "3", "idlis", "extra", "sambar"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["give", "one", "dosa", "wit", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["make", "dosa", "extra", "crisp"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "sambar", "on", "top"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O"]},
    {"tokens": ["send", "me", "big", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["one", "idli", "with", "xtra", "chutni"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "poori", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "dosaa", "no", "salt"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "four", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "2", "vada", "with", "extra", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "pongal", "with", "chutney"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["please", "add", "chutney", "extra"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "idli", "extra", "crisp"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "idlis", "no", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "pongal", "witout", "onion"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "1", "dosa", "less", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "one", "idli", "no", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "idly", "no", "onion", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["pls", "giv", "me", "3", "idlies", "and", "xtra", "chuttney"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "1", "dosa", "widout", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "poori", "no", "ghee", "pls"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["send", "one", "pongal", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["xtra", "crispy", "dosa", "plz"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-FOOD", "O"]},
    {"tokens": ["i", "need", "4", "idli", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["masal", "dos", "1", "plate", "no", "ghee"], "labels": ["B-FOOD", "I-FOOD", "B-QUANTITY", "I-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "vada", "no", "coconut", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3x", "idlis", "wid", "spicy", "chutny"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "1", "pongal", "n", "2", "vada"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "1", "idli", "wit", "xtra", "sambhar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "2", "idlis", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "take", "3", "idly", "more", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["hey", "1", "masala", "dosa", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "witout", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "chutny", "to", "my", "1", "idli"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["2", "pooris", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "dosa", "more", "crispy", "n", "spicy"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "I-CUSTOMIZATION"]},
    {"tokens": ["xtra", "sambhar", "and", "no", "onion"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "vada", "plz"], "labels": ["B-QUANTITY", "B-FOOD", "O"]},
    {"tokens": ["get", "me", "2", "idlis", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["chutney", "extra", "pls", "and", "3", "idlis"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "dosa", "x1", "widout", "spicy", "chutney"], "labels": ["O", "B-FOOD", "B-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "2x", "vada", "more", "crisp"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "4", "idlis", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "poori", "with", "xtra", "sambhar"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "pongal", "no", "onion", "n", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "masala", "dosa", "no", "butter"], "labels": ["O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "idli", "x3", "extra", "chutney"], "labels": ["O", "B-FOOD", "B-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "plain", "dosa", "no", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "1", "idly", "xtra", "chuttni", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["give", "me", "dosa", "less", "ghee"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "pongal", "no", "onion"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "poori", "less", "oil", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["xtra", "hot", "chutny", "with", "my", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["pls", "send", "4", "idlies", "n", "vada"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["idli", "x2", "wit", "no", "coconut"], "labels": ["B-FOOD", "B-QUANTITY", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "witout", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "extra", "sambhar", "and", "1", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["can", "i", "get", "3x", "idlis", "xtra", "crispy"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
{"tokens": ["give", "me", "two", "plates", "masala dosa", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "plain dosa"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD"]},
                {"tokens": ["give", "me", "two", "plates", "rava dosa", "with", "chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "onion dosa", "without","oil"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Idli", "with","chutney","and","samabar"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION","O","I-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "vegetable Uttapam", "with","chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "onion Uttapam", "with","chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "tomato Uttapam", "with","chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Poha", "with","samabr"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Aloo Paratha", "with","sabji"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Puri Bhaji", "with","samabr","and","chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION","O","I-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Thepla", "with","samabr"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Sabudana Khichdi", "with","chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plate", "Chole Bhature", "little","spicy"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Chole Bhature", "little","spicy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Vegetable Sandwich", "little","crispy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Bread Butter", "little","crispy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Bread Toast", "little","crispy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Pongal", "little","splicy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "karaPongal", "little","splicy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "sihiPongal", "little","sweet"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},    
                {"tokens": ["need", "Three", "plate", "Misal Pav", "little","splicy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Misal Pav", "little","splicy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Samosa", "little","sauces"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "Three", "plate", "Kachori", "little","crispy"], "labels": ["O","B-QUANTITY","O", "B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Pakora", "onion", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","I-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Pakora", "potato", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","I-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Pakora", "paneer", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","I-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Dhokla", "little","sweet"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Veg Cutlet", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Spring Rolls", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Bhel Puri", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Sev Puri", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Pani Puri", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Golgappa", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Dahi Puri", "little","salt"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Vada Pav", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Batata Vada", "little","salt"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Chilli Paneer", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["need", "one", "Manchurian", "little","spicy"], "labels": ["O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Paneer Butter Masala", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kadai Paneer", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Palak Paneer", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Shahi Paneer", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Chole", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Rajma", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Dal Makhani", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Aloo Gobi", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Bhindi Masala", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Baingan Bharta", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Mix Veg Curry", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Malai Kofta", "with","Roti"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Jeera Aloo", "with","Chapati"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Stuffed Capsicum", "with","Naan"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Malai Kofta", "little","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "plain Naan", "without","oil"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "butter Naan", "without","oil"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "garlic Naan", "without","oil"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Tandoori Roti", "with","crspy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Gulab Jamun", "with","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Rasgulla", "with","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Jalebi", "with","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kheer", "with","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Payasam", "with","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Halwa", "with","sooji"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Halwa", "with","carrot"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Halwa", "with","moong dal"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ladoo", "with","besan"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ladoo", "with","kaju"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Barfi", "with","coconut"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Barfi", "with","pista"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Rasmalai", "with","pista"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kulfi", "with","pista"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Neer Dosa", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Set Dosa", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Rava Idli", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kesari Bath", "little","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Sooji Halwa", "little","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Bread Upma", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ragi Dosa", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Sheera", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ragi Mudde", "with","sambar"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Moong Dal Chilla", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Besan Chilla", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Moong Dal Khichdi","large", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Methi Thepla", "with","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Handvo", "with","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Vegetable Pulao", "with","spicy"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Pesarattu","large" "with","Raita"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Vegetable Pulao", "with","Raita"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Peas Pulao", "with","Papad"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ghee Rice", "with","Raita"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Ghee Rice", "with","Pickles"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Coconut Rice", "with","Coconut Chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kashmiri Pulao", "with","Green Chutney","large"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION","B-SIZE"]},
                {"tokens": ["give", "me","one", "Methi Rice", "with","Tamarind Chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Vangi Bath", "with","Sambar"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Raw Mango Rice", "with","Rasam"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Schezwan Fried Rice","large", "with","Tamarind Chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Methi Rice", "with","Tamarind Chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kesari", "with","Rava Kesari"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Kesari", "with","Pineapple Kesari"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Badam Halwa", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Peda", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Mysore Pak", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Khaja", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Basundi", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Phirni", "more","sweet"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Shrikhand", "with","Mango"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Shrikhand","large", "with","Elaichi"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Shrikhand", "with","Kesar"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Steamed Rice", "with","samabr"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Curd Rice", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Tamarind Rice", "with","samabr"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Bisibele Bath", "with","samabr"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Khichdi", "with","samabr"], "labels": ["O","O","B-QUANTITY","B-FOOD","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "Stuffed Capsicum","large" "with","samabr"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "veg biriyani","large", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "veg biriyani","small", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "veg biriyani","medium", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "veg biriyani","small", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me","one", "veg biriyani","raita", "with","chutney"], "labels": ["O","O","B-QUANTITY","B-FOOD","B-SIZE","O" ,"B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Rava Dosa", "Extra ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Onion Dosa", "Extra ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Sweet Pongal", "Extra Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Set Dosa", "ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Pesarattu", "ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Rava Upma", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Rava Idli", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Vegetable Uttapam", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "three", "plates", "Onion Uttapam", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "three", "plates", "Ragi Dosa", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "three", "plates", "Wheat Dosa", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "three", "plates", "Oats Dosa", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "three", "plates", "Corn Dosa", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Kharabath", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Masala Sandwich", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Cheese Dosa", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Paneer Dosa", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Aloo Paratha", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Paneer Paratha", "Spicy Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Methi Thepla", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Sabudana Khichdi", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Mini Idli", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Sheera", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Palak Dhokla", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Rava Kesari", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Puli Aval", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Lemon Aval", "Mild Chutney"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Coconut Aval", "With Butter"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Mangalore Buns", "With Butter"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plates", "Mysore Bonda", "With Butter"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plates", "Medu Vada", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "oue", "plates", "Masala Vada", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "four", "plates", "Urad Dal Vada", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plates", "Cornflakes Poha", "no ghee"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "four", "plates", "Stuffed Dosa", "With Cashews"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "one", "plates", "Khara Dosa", "With Cashews"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Roti Upma", "no sambar"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Mixed Dal Dosa", "no sambar"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Vegetable Poha", "with curd"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Kanda Poha", "with curd"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Batata Poha", "with curd"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Chutney Sandwich", "chuteny"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Idli Fry", "chuteny"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Poori Bhaji", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Chole Bhature", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Bread Upma", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Moong Dal Chilla", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Besan Chilla", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Sweet Dosa", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Sooji Upma", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Tomato Upma", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Lemon Upma", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
                {"tokens": ["give", "me", "two", "plates", "Gobi Manchurian", "No Garlic"], "labels": ["O", "O", "B-QUANTITY","O", "B-FOOD", "B-CUSTOMIZATION"]},
      
]

class FoodNERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        labels = self.data[idx]["labels"]
        encoding = self.tokenizer(tokens, is_split_into_words = True, return_tensors="pt", truncation =True, padding ='max_length', max_length = 32)

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[labels[word_idx]])
            else:
                lab = labels[word_idx]
                if lab.startswith("B-"):
                    lab = lab.replace("B-", "I-")
                label_ids.append(self.label_to_id.get(lab, self.label_to_id["O"]))
            previous_word_idx = word_idx
        char_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                char_ids.append([0]*MAX_CHAR_LEN)
            else:
                char_ids.append(char_encode_token(tokens[word_idx]))

        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids)
        item['char_ids'] = torch.tensor(char_ids)
        return item

# Assuming 'data' is already loaded as a list of dicts with 'tokens' and 'labels'
unique_labels = set()
for sample in data:
    unique_labels.update(sample["labels"])
print("Unique labels in dataset:", unique_labels)
print("Labels expected:", label_to_id.keys())

# Now create the dataset
dataset = FoodNERDataset(data, tokenizer, label_to_id)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

dataset = FoodNERDataset(data, tokenizer, label_to_id)
#print(dataset)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#print(dataloader)

bert_model_name = "bert-base-uncased"
num_labels = len(label_list)
#print(num_labels)
char_vocab_size = len(char_to_id)
#print(char_vocab_size)
char_emb_dim = 30
#print(char_emb_dim)
char_out_channels = 30
#print(char_out_channels)
char_kernel_size = 3
#print(char_kernel_size)
MAX_CHAR_LEN = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layers

bert = BertModel.from_pretrained(bert_model_name).to(device)
char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0).to(device)
char_cnn = nn.Conv1d(in_channels=char_emb_dim, out_channels=char_out_channels, kernel_size=char_kernel_size, padding=1).to(device)
relu = nn.ReLU().to(device)
dropout = nn.Dropout(0.1).to(device)
classifier = nn.Linear(bert.config.hidden_size + char_out_channels, num_labels).to(device)

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

# Forward function
def forward(input_ids, attention_mask, char_ids, labels=None):
    bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = bert_output.last_hidden_state 

    batch_size, seq_len, max_char_len = char_ids.size()
    char_emb = char_embedding(char_ids.view(-1, max_char_len))  
    char_emb = char_emb.transpose(1, 2)  
    char_cnn_out = char_cnn(char_emb)    
    char_cnn_out = relu(char_cnn_out)
    char_rep, _ = torch.max(char_cnn_out, dim=2)  
    char_rep = char_rep.view(batch_size, seq_len, -1)  

    combined = torch.cat([sequence_output, char_rep], dim=2)  
    combined = dropout(combined)
    logits = classifier(combined)  
    if labels is not None:
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss, logits
    else:
        return logits


# Optimizer

params = list(bert.parameters()) + list(char_embedding.parameters()) + \
         list(char_cnn.parameters()) + list(classifier.parameters()) + \
         list(dropout.parameters())
optimizer = AdamW(params, lr=5e-5)

# Training
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_loss = 0
    bert.train()
    char_embedding.train()
    char_cnn.train()
    classifier.train()

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = forward(input_ids, attention_mask, char_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average loss: {avg_loss:.4f}")

# Save Model

torch.save({
    "bert": bert.state_dict(),
    "char_embedding": char_embedding.state_dict(),
    "char_cnn": char_cnn.state_dict(),
    "classifier": classifier.state_dict(),
    "optimizer": optimizer.state_dict(),
    "label_to_id": label_to_id,
    "char_to_id": char_to_id
}, "food_ner_model.pt")
print("Model saved to food_ner_model.pt")

# LOAD MODEL (later, before inference)
checkpoint = torch.load("food_ner_model.pt", map_location=device)
bert.load_state_dict(checkpoint["bert"])
char_embedding.load_state_dict(checkpoint["char_embedding"])
char_cnn.load_state_dict(checkpoint["char_cnn"])
classifier.load_state_dict(checkpoint["classifier"])
optimizer.load_state_dict(checkpoint["optimizer"])
label_to_id = checkpoint["label_to_id"]
char_to_id = checkpoint["char_to_id"]
print("Model loaded from food_ner_model.pt")

# Inference

bert.eval()
char_embedding.eval()
char_cnn.eval()
classifier.eval()

sentence = "want masala but no butter"
tokens = sentence.split()

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
word_ids = encoding.word_ids(batch_index=0)

# Char IDs aligned with tokens
char_ids = []
for word_idx in word_ids:
    if word_idx is None:
        char_ids.append([0] * MAX_CHAR_LEN)
    else:
        char_ids.append(char_encode_token(tokens[word_idx]))

char_ids = torch.tensor([char_ids], device=device)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    logits = forward(input_ids, attention_mask, char_ids)

pred_ids = torch.argmax(logits, dim=-1)[0]

# Map predictions to labels
final_labels = []
previous_word_idx = None
for idx, word_id in enumerate(word_ids):
    if word_id is None:
        continue
    if word_id != previous_word_idx:
        final_labels.append(id_to_label[pred_ids[idx].item()])
    previous_word_idx = word_id

print("Tokens:         ", tokens)
print("Predicted labels:", final_labels)
