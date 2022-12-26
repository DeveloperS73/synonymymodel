# -*- coding: utf-8 -*-

pip install -U sentence-transformers

import sentence_transformers

sentences = ["Кошка ловит мышку", "Кошка ловит кайф"]

model = sentence_transformers.SentenceTransformer('inkoziev/sbert_synonymy')
embeddings = model.encode(sentences)

s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Люк вырос на Татуине", "Энакин родился на Татуине"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Гагарин - первый в Космосе", "Армстронг - первый на луне"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Кушать сушки", "Употреблять баранки"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Россия в беде", "У гитары порвалась струна"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Он же ребёнок", "Он жеребёнок"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Бери гитару", "Береги тару"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Ленин великий революционер", "Ленин предатель родины"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Сонзнание порождает материю", "Материя порождает сознание"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Сыр косичка", "Кыр сосичка"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Против войны", "За мир"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

sentences = ["Зов припяти", "Тень Чернобыля"]
embeddings = model.encode(sentences)
s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
print('Вероятность смыслового совпадения:', s)

"""Как модель переводит фразу в вектор?"""

import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
# model.cuda()  # uncomment it if you have a GPU

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

print(embed_bert_cls('Привет мир', model, tokenizer))
