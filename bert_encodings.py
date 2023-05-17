import transformers
import pickle

from workshop import *
from data import dataset

model = transformers.BertModel.from_pretrained("bert-base-cased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

mbpp = dataset()
sentences = [row["prompt"] for row in mbpp]

inputs = tokenizer(sentences, padding=True, return_tensors="pt")
outputs = model(**inputs)

masked_states = inputs.attention_mask[:, :, None] * outputs.last_hidden_state
sentence_encodings = masked_states.mean(axis=1)

tagged_sentence_embeddings = [
    {
        **mbpp_instance,
        "bert_encoding": encoding
    }
    for (encoding, mbpp_instance) in zip(sentence_encodings, mbpp)
]

root = output_directory("bert_encodings")
output_file = root / "mbpp_sanitized_with_bert_encodings.pickle"
with open(output_file, "wb") as file:
    pickle.dump(tagged_sentence_embeddings, file)