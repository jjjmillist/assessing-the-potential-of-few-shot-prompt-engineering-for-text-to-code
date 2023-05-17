from   transformers import AutoTokenizer, AutoModelForCausalLM

from data import dataset
from predict import predict_batch
from util import *
from settings import *

mbpp = dataset()
prompts = []
for row in mbpp:    
    prompt = ""
    for line in row["prompt"].splitlines():
        prompt += "# " + line + "\n"

    for line in row["code"].splitlines():
        if line.startswith("def "):
            prompt += line.strip() + "\n"
            break
    
    prompts.append(prompt)

tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForCausalLM.from_pretrained(model_uri)

model.to("cuda:0")

root = output_directory("code/no_prefixes")

predict_batch(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    n_samples=n_samples,
    k=n_distribution_cutoff,
    output_path=root / "code.pickle"
)