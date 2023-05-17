bert_encodings: results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle

results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle:
	PYTHONPATH=. python bert_encodings.py

code: bert_encodings
	PYTHONPATH=. python code_generation/bert_prompt_agnostic.py
	PYTHONPATH=. python code_generation/bert_prompt_aware.py
	PYTHONPATH=. python code_generation/code_only_prefixes.py
	PYTHONPATH=. python code_generation/no_prefixes.py
	PYTHONPATH=. python code_generation/random_prefixes.py

evaluation: code
	python evaluation.py

figures: evaluation
	python figure_1.py
	python figure_2.py
	python figure_3.py

clean:
	rm -r results/*