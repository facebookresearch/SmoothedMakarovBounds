# Assumption-lean inference on treatment effect distributions

About
This repository provides the code for the paper "Assumption lean inference on treatment effect distributions". It provides methodology for estimating Makarov bounds on the c.d.f. of the treatment effect in A/B tests and reproduces the synthetic experiments in the paper.

#### Project structure
- *methodology.py* contains the core methodology and estimation algorithms
- *synthetic_exp.py* contains the code to run the synthetic experiments and the results
- *hyperparam* contains the tuning ranges and code for hyperparameter tuning
- *models.py* contains code for the nuisance models based on lightgbm
- *nuisances.py* contains the code for training the nuisance models and estimating empirical c.d.f.s (for marginal bounds)
- *run_config.py* executes *synthetic_exp.run* using the configuration in *configs.config_synthetic.json* (reproduces synthetic experiments from the paper)


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
The code is <MIT> licensed, as found in the LICENSE file.
