# Estimating $s_\text{het}$ using GeneBayes

We ran the following command to estimate $s_\text{het}$. GeneBayes returns posterior means and 95\% credible intervals for $s_\text{het}$, posterior densities, and feature importance metrics. Required input files and $s_\text{het}$ estimates can be found at: https://doi.org/10.5281/zenodo.7939767.


```
python genebayes.s_het.py \
  --response gene_likelihoods.pkl \
  --features gene_features_for_s_het.tsv.gz \
  --out s_het_genebayes \
  --sg_err 35 \
  --sd_err 40 \
  --sa_err 42 \
  --batch_size 100 \
  --n_integration_pts 10000 \
  --lr 0.04 \
  --total_iterations 1000 \
  --n_trees_per_iteration 1 \
  --reg_alpha 0 \
  --reg_lambda 2 \
  --subsample 0.8 \
  --min_child_weight 4 \
  --max_depth 3 \
  --early_stopping_iter 10 > train.log
```

See below for definitions of command-line arguments.

```
usage: genebayes.s_het.py [-h] [--gene_column GENE_COLUMN]
                          [--batch_size BATCH_SIZE]
                          [--n_integration_pts N_INTEGRATION_PTS]
                          [--total_iterations TOTAL_ITERATIONS]
                          [--early_stopping_iter EARLY_STOPPING_ITER]
                          [--lr LR] [--max_depth MAX_DEPTH]
                          [--n_trees_per_iteration N_ESTIMATORS]
                          [--min_child_weight MIN_CHILD_WEIGHT]
                          [--reg_alpha REG_ALPHA] [--reg_lambda REG_LAMBDA]
                          [--subsample SUBSAMPLE] [--sg_err SG] [--sd_err SD]
                          [--sa_err SA] [--hess_reg_1 HESS_REG_1]
                          [--hess_reg_2 HESS_REG_2] --response RESPONSE
                          --features FEATURES [--out OUT_PREFIX]
                          [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --gene_column GENE_COLUMN
                        Name of the column containing gene names/IDs.
                        (default: ensg)
  --batch_size BATCH_SIZE
                        Batch size for processing genes during training and
                        integration. (default: 1)
  --n_integration_pts N_INTEGRATION_PTS
                        Number of points for numerical integration. Larger
                        values increase can improve accuracy but also increase
                        training time and/or numerical instability. (default:
                        1001)
  --total_iterations TOTAL_ITERATIONS
                        Maximum number of iterations. The actual number of
                        iterations may be lower if using early stopping (see
                        the 'train' option) (default: 500)
  --early_stopping_iter EARLY_STOPPING_ITER
                        If >0, chromosomes 2, 4, 6 will be held out for
                        validation and training will end when the loss on
                        these chromosomes stops decreasing for the specified
                        number of iterations. Otherwise, the model will train
                        on all genes for the number of iterations specified in
                        'total_iterations'. (default: 10)
  --lr LR               Learning rate for NGBoost. Smaller values may improve
                        accuracy but also increase training time. Typical
                        values: 0.01 to 0.2. (default: 0.05)
  --max_depth MAX_DEPTH
                        XGBoost parameter. See https://xgboost.readthedocs.io/
                        en/stable/python/python_api.html. (default: 3)
  --n_trees_per_iteration N_ESTIMATORS
                        XGBoost parameter, n_estimators (default: 1)
  --min_child_weight MIN_CHILD_WEIGHT
                        XGBoost parameter. (default: None)
  --reg_alpha REG_ALPHA
                        XGBoost parameter. (default: None)
  --reg_lambda REG_LAMBDA
                        XGBoost parameter. (default: None)
  --subsample SUBSAMPLE
                        XGBoost parameter. (default: None)
  --sg_err SG           Index for the error rate to use for stop gain
                        variants. (default: 0)
  --sd_err SD           Index for the error rate to use for splice donor
                        variants. (default: 0)
  --sa_err SA           Index for the error rate to use splice acceptor
                        variants. (default: 0)
  --hess_reg_1 HESS_REG_1
                        Experimental regularization parameter for 2nd order
                        optimization. Unused in the current setup (where
                        natural_gradient=False). (default: 0.5)
  --hess_reg_2 HESS_REG_2
                        Experimental regularization parameter for 2nd order
                        optimization. Unused in the current setup (where
                        natural_gradient=False). (default: 0.5)
  --model MODEL         Provide a pretrained model to obtain predictions for
                        that model. (default: None)

required arguments:
  --response RESPONSE   tsv containing data that can be related to the gene
                        property of interest through a likelihood function.
                        Also known as y or dependent variable. (default: None)
  --features FEATURES   tsv containing gene features. Also known as X or
                        independent variables. (default: None)
  --out OUT_PREFIX      Prefix for the output files. (default: None)
```
