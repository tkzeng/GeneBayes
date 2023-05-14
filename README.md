# GeneBayes

GeneBayes is an Empirical Bayes framework that can be used to improve estimation of any gene property that one can relate to available data through a likelihood function. See XXX for an in-depth application of GeneBayes for gene constraint ($s_\text{het}) estimation, and XXX for additional example applications. 

To estimate a gene property of interest using GeneBayes, one needs to specify a prior distribution and likelihood. Then, GeneBayes trains a machine learning model (gradient-boosted trees) to predict the parameters of the prior distribution by maximizing the likelihood of the data. Finally, GeneBayes computes a per-gene posterior distribution for the gene property of interest, returning a posterior mean and 95% credible interval for each gene.

For details on estimating $s_\text{het}$ using GeneBayes, see the `s_het` directory. 

## Customizing GeneBayes for new gene properties

To use GeneBayes for a gene property of interest, please start with the file `examples/genebayes.o_over_e.py`, which contains an example prior and likelihood, and modify it to specify a custom prior/likelihood. See below for details.

### Prior

The `Prior` class defines the distribution $p(\theta_i)$, where $\theta_i$ represents the gene property of interest for gene $i$.

In the `Prior` class, specify the following:
* attribute: `n_params`
  * integer - the number of parameters in the distribution
* attribute: `positive`
  * boolean vector - denotes which of the params should be positive
* function: `distribution`
  * `PyTorch` distribution from `torch.distributions` or custom distribution that implements the methods `sample` and `log_prob`  

Example: Beta distribution
```
class Prior(RegressionDistn):
    n_params = 2
    
    # both params need to be >0
    positive = np.array([True, True])
   
    ...
    
    def distribution(params):
        alpha = params[0] 
        beta = params[1]
        return torch.distributions.Beta(alpha, beta)
```

### Likelihood

The `likelihood` function returns the probability $p(y_i|\theta_i)$, where $y_i$ represents available data for gene $i$ and $\theta_i$ is the gene property of interest.

The function can be defined using `PyTorch` distributions or custom code.

Example: Poisson likelihood, $\text{observed}_i \sim \text{Poisson}(\lambda_i\times\text{expected}_i)$, where $\text{observed}_i$ and $\text{expected}_i$ are data provided by the user, and the gene property of interest is $\lambda_i$.

```
def likelihood(i, gene_property, y):
    lambd = gene_property
    expected = torch.tensor(y[i, 1], device=device)
    observed = torch.tensor(y[i, 0], device=device)
    p = dist.Poisson(lambd * expected).log_prob(observed)
    return p
```

## Training a GeneBayes model

After specifying a prior and likelihood, one can train a GeneBayes model by specifying arguments via command line. The positional arguments described below are required.

```
usage: genebayes.custom_property.py [-h] [--n_integration_pts N_INTEGRATION_PTS] [--total_iterations TOTAL_ITERATIONS] [--train TRAIN] [--lr LR] [--max_depth MAX_DEPTH]
                                    [--n_estimators N_ESTIMATORS] [--min_child_weight MIN_CHILD_WEIGHT] [--reg_alpha REG_ALPHA] [--reg_lambda REG_LAMBDA] [--subsample SUBSAMPLE]
                                    response features out_prefix integration_lb integration_ub
                             
positional arguments:
  response              tsv containing data that can be related to the gene property of interest through a likelihood function. Also known as y or dependent variable. Other
                        information necessary for computing the likelihood can also be included in this file.
  features              tsv containing gene features. Also known as X or independent variables.
  out_prefix            Prefix for the output files.
  integration_lb        Lower bound for numerical integration - the smallest value that you expect for the gene property of interest.
  integration_ub        Upper bound for numerical integration - the largest value you expect for the gene property of interest.

optional arguments:
  -h, --help            show this help message and exit
  --n_integration_pts N_INTEGRATION_PTS
                        Number of points for numerical integration. Larger values increase can improve accuracy but also increase training time and/or numerical instability.
                        (default: 1000)
  --total_iterations TOTAL_ITERATIONS
                        Maximum number of iterations. The actual number of iterations may be lower if using early stopping (see the 'train' option) (default: 1000)
  --train TRAIN         If True, then chromosomes 2, 4, 6 will be held out for validation and training will end when the loss on these chromosomes stops decreasing. Otherwise, the
                        model will train on all genes for the number of iterations specified in 'total_iterations'. (default: True)
  --lr LR               Learning rate for NGBoost. Smaller values may improve accuracy but also increase training time. Typical values: 0.01 to 0.2. (default: 0.05)
  --max_depth MAX_DEPTH
                        XGBoost parameter. See https://xgboost.readthedocs.io/en/stable/python/python_api.html. (default: 3)
  --n_estimators N_ESTIMATORS
                        XGBoost parameter. (default: 1.0)
  --min_child_weight MIN_CHILD_WEIGHT
                        XGBoost parameter. (default: None)
  --reg_alpha REG_ALPHA
                        XGBoost parameter. (default: None)
  --reg_lambda REG_LAMBDA
                        XGBoost parameter. (default: None)
  --subsample SUBSAMPLE
                        XGBoost parameter. (default: None)
```
Next, we describe the required arguments in more detail.

* `response`: The format of this file will depend on the likelihood. See XXX for an example. In the simplest case, this file will contain a single column, corresponding to a scalar response variable for each gene.
* `features`: We support continuous and categorical features, including features with missing values for certain genes. Categorical features should be one-hot encoded. Features do not need to be scaled. In addition, we provide two pre-computed gene feature files:
  * XXX (XXX in total): All features - includes gene expression levels, Gene Ontology terms, conservation across species, neural network embeddings of protein sequences, gene regulatory features, co-expression and protein-protein interaction features, sub-cellular localization, and intolerance to missense mutations.
  * XXX (XXX in total): Subset of the first feature file - optimized for gene constraint estimation.
  * See XXX, XXX for a description of the pre-computed features.
* `integration_lb`, `integration_ub`: These arguments specify the range of values that GeneBayes will consider for the gene property of interest. We recommend for users to consider the range of values supported by their likelihood when choosing these parameters.
