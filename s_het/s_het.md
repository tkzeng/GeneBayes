# Estimating $s_\text{het}$ using GeneBayes

We ran the following command to estimate $s_\text{het}$. GeneBayes returns posterior means and 95\% credible intervals for $s_\text{het}$, posterior densities, and feature importance metrics. Required input files and $s_\text{het}$ estimates can be found at: https://doi.org/10.5281/zenodo.7939768.


```
python genebayes.s_het.py \
gene_likelihoods.pkl gene_features_for_s_het.tsv genebayes_s_het 5e-08 0.99999995 \
--sg_err 34 \
--sd_err 41 \
--sa_err 42 \
--train True \
--lr 0.01 \
--max_depth 3 \
--min_child_weight 1 \
--subsample 0.8 \
--n_trees_per_iteration 4 \
--reg_alpha 2 \
--reg_lambda 1 \
--total_iterations 3
```

See below for definitions of command-line arguments.

```
positional arguments:
  likelihood            pkl file with pre-computed likelihoods for s_het.
  features              tsv containing gene features. Also known as X or independent variables.
  out_prefix            Prefix for the output files.
  integration_lb        Lower bound for numerical integration - the smallest value that you expect for the gene property of interest.
  integration_ub        Upper bound for numerical integration - the largest value you expect for the gene property of interest.

optional arguments:
  -h, --help            show this help message and exit
  --sg_err SG           Index for the error rate to use for stop gain variants. (default: 0)
  --sd_err SD           Index for the error rate to use for splice donor variants. (default: 0)
  --sa_err SA           Index for the error rate to use splice acceptor variants. (default: 0)
  --model MODEL         Load pretrained model to get estimates of the gene property. (default: None)
  --n_integration_pts N_INTEGRATION_PTS
                        Number of points for numerical integration. Larger values increase can improve accuracy but also increase training time and/or
                        numerical instability. (default: 1000)
  --total_iterations TOTAL_ITERATIONS
                        Maximum number of iterations. The actual number of iterations may be lower if using early stopping (see the 'train' option) (default:
                        1000)
  --train TRAIN         If True, then chromosomes 2, 4, 6 will be held out for validation and training will end when the loss on these chromosomes stops
                        decreasing. Otherwise, the model will train on all genes for the number of iterations specified in 'total_iterations'. (default: True)
  --lr LR               Learning rate for NGBoost. Smaller values may improve accuracy but also increase training time. Typical values: 0.01 to 0.2. (default:
                        0.05)
  --max_depth MAX_DEPTH
                        XGBoost parameter. See https://xgboost.readthedocs.io/en/stable/python/python_api.html. (default: 3)
  --n_trees_per_iteration N_ESTIMATORS
                        XGBoost parameter. (default: 1)
  --min_child_weight MIN_CHILD_WEIGHT
                        XGBoost parameter. (default: None)
  --reg_alpha REG_ALPHA
                        XGBoost parameter. (default: None)
  --reg_lambda REG_LAMBDA
                        XGBoost parameter. (default: None)
  --subsample SUBSAMPLE
                        XGBoost parameter. (default: None)
```