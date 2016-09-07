# Nltk-Tagger-Portuguese
Tagger treinado para reconhecer palavras do Português

Utilização do Tagger (após gerar o tagger.pkl)
```python
import pickle
import nltk
text = ...
tagger = pickle.load(open("tagger.pkl"))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
sentences = portuguese_sent_tokenizer.tokenize(text)
tags = [tagger.tag(nltk.word_tokenize(sentence)) for sentence in sentences]
```

#Modelos

##DefaultTagger
Último fallback da cadeia, se ninguém resolver a palavra então ela deve ser um NOUN (substantivo).

## AffixTagger
Classifica a palavra de acordo com o prefixo/sufixo comum.

## UnigramTagger
Classifica a palavra de acordo com um dicionário de palavras vs classificação mais comum.

## RegexpTagger
Utilizado para corrigir alguns casos especiais em que os demais Taggers erram constantemente.

## BigramTagger
Classifica a palavra utilizando a palavra anterior para eliminar ambiguidades (ex: palavras comuns que podem ser tanto um verbo quanto um substantivo)

## BrillTagger
Treinado para reconhecer os padrões em que os demais taggers erram, e criar regras de substituição.
É a etapa mais demorada e que consome mais memória (por isto não esta sendo treinado com o dataset2), mas produz os maiores ganhos.
