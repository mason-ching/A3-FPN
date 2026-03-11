# from transformers import pipeline
#
# # Allocate a pipeline for sentiment-analysis
# classifier = pipeline('sentiment-analysis')
# classifier('We are very happy to introduce pipeline to the transformers repository.')

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
