bert_encodings: results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle

results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle:
	PYTHONPATH=. python3 bert_encodings.py

code: bert_encodings
	PYTHONPATH=. python3 code_generation/bert_prompt_agnostic.py
	PYTHONPATH=. python3 code_generation/bert_prompt_aware.py
	PYTHONPATH=. python3 code_generation/code_only_prefixes.py
	PYTHONPATH=. python3 code_generation/no_prefixes.py
	PYTHONPATH=. python3 code_generation/random_prefixes.py

evaluation:
	python3 evaluation.py

figures:
	python3 figure_1.py
	python3 figure_2.py
	python3 figure_3.py

clean:
	rm -r results/*