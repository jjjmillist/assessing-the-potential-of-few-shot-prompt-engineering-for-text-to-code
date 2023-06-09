# Models
CODEPARROT = "codeparrot/codeparrot"
PYCODEGPT = "Daoguang/PyCodeGPT"

# General
model_uri = PYCODEGPT
n_samples = 20
n_contextual_examples = 3
n_distribution_cutoff = 10
n_data = None # use the first n_data rows of the dataset, for debugging - set to None to use everything

# Monte Carlo experiment
n_monte_carlo_prefixes = 30