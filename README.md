# GeneBayes

GeneBayes is an Empirical Bayes framework that can be used to improve estimation of any gene property that one can relate to available data through a likelihood function. See [Zeng et al. 2023](https://www.biorxiv.org/content/10.1101/2023.05.19.541520v1) for an in-depth application of GeneBayes for gene constraint ($s_\text{het}$) estimation. See below for additional example applications. 

To estimate a gene property of interest using GeneBayes, one needs to specify a prior distribution and likelihood. Then, GeneBayes trains a machine learning model (gradient-boosted trees) to predict the parameters of the prior distribution by maximizing the likelihood of the data. Finally, GeneBayes computes a per-gene posterior distribution for the gene property of interest, returning a posterior mean and 95% credible interval for each gene.

See the `s_het` directory for details on using GeneBayes to estimate $s_\text{het}$. 

## Setup

* Create a `python` environment:  
   ```
   conda create --name genebayes python=3.9
   ```
* Install `xgboost`: https://xgboost.readthedocs.io/en/stable/install.html
  * Installation with GPU support is recommended.
* Install `ngboost`:
   ```
   git clone https://github.com/tkzeng/ngboost.git
   pip install ngboost/
   ```
* Install `PyTorch`: https://pytorch.org/get-started/locally/
  * Installation with GPU support is recommended.
* Install `torchquad`:
   ```
   conda install torchquad -c conda-forge
   ```

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
usage: genebayes.o_over_e.py [-h] [--gene_column GENE_COLUMN] [--n_integration_pts N_INTEGRATION_PTS] [--total_iterations TOTAL_ITERATIONS]
                             [--early_stopping_iter EARLY_STOPPING_ITER] [--lr LR] [--max_depth MAX_DEPTH] [--n_trees_per_iteration N_ESTIMATORS]
                             [--min_child_weight MIN_CHILD_WEIGHT] [--reg_alpha REG_ALPHA] [--reg_lambda REG_LAMBDA] [--subsample SUBSAMPLE]
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
  --gene_column GENE_COLUMN
                        Name of the column containing gene names/IDs. If not specified, will look for 'ensg' and 'hgnc' columns. (default: None)
  --n_integration_pts N_INTEGRATION_PTS
                        Number of points for numerical integration. Larger values increase can improve accuracy but also increase training time and/or numerical instability.
                        (default: 1001)
  --total_iterations TOTAL_ITERATIONS
                        Maximum number of iterations. The actual number of iterations may be lower if using early stopping (see the 'train' option) (default: 500)
  --early_stopping_iter EARLY_STOPPING_ITER
                        If >0, chromosomes 2, 4, 6 will be held out for validation and training will end when the loss on these chromosomes stops decreasing for the specified number
                        of iterations. Otherwise, the model will train on all genes for the number of iterations specified in 'total_iterations'. (default: 10)
  --lr LR               Learning rate for NGBoost. Smaller values may improve accuracy but also increase training time. Typical values: 0.01 to 0.2. (default: 0.05)
  --max_depth MAX_DEPTH
                        XGBoost parameter. See https://xgboost.readthedocs.io/en/stable/python/python_api.html. (default: 3)
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
```
Next, we describe the required arguments in more detail.

`response`: The format of this file will depend on the likelihood. See `examples/number_of_obs_and_exp_lofs.tsv` for an example. In the simplest case, this file will contain a single column, corresponding to a scalar response variable for each gene.

`features`:
  * We provide two pre-computed gene feature files (see [feature_list.xlsx](https://zenodo.org/record/7939768/files/feature_list.xlsx?download=1) for a more detailed description):
    * [all_gene_features.tsv.gz](https://zenodo.org/record/7939768/files/all_gene_features.tsv.gz?download=1) (65,383 features): All features - includes gene expression levels, Gene Ontology terms, conservation across species, neural network embeddings of protein sequences, gene regulatory features, co-expression and protein-protein interaction features, sub-cellular localization, and intolerance to missense mutations. 
    * [gene_features_for_s_het.tsv.gz](https://zenodo.org/record/7939768/files/gene_features_for_s_het.tsv.gz?download=1) (1,248 features): Subset of the first feature file - optimized for gene constraint estimation.
  * Custom features 
    * GeneBayes supports continuous and categorical features, including features with missing values for certain genes. 
      Categorical features should be one-hot encoded. 
    * Features do not need to be scaled. 
    * In addition to a column containing gene names, please include a column named `chrom` with the chromosome for each gene.

`integration_lb`, `integration_ub`: These arguments specify the range of values that GeneBayes will consider for the gene property of interest. We recommend for users to consider the range of values supported by their likelihood when choosing these parameters.

### Runtime

Runtime primarily depends on  `total_iterations`, `lr`, and `n_integration_pts`. Generally, GeneBayes takes a few hours to train. Runtime is substantially lower when a GPU is available. 

## Outputs

First, GeneBayes outputs estimates of the gene property from the prior and posterior distributions.

`{out_prefix}.per_gene_estimates.tsv`
  * `gene_column`/`ensg`/`hgnc`: custom / Ensembl / HGNC gene IDs 
  * `prior_mean`: mean of the learned prior
  * `post_mean`: mean of the posterior
  * `post_lower_95`: lower bound of the 95% credible interval
  * `post_upper_95`: upper bound of the 95% credible interval

Second, GeneBayes outputs feature importance scores that represent the influence of each feature on the prior. The importance score of a feature is calculated as the number of times the feature is used to split the data across all trees. Since each parameter in the prior is predicted by a separate gradient-boosted tree model, importance scores are provided for each parameter.

`{out_prefix}.feature_importance.tsv`
* `gene_column`/`ensg`/`hgnc`: custom / Ensembl / HGNC gene IDs 
* `param0_importance`: feature importance score for the first parameter of the prior  
    ...
* `paramK_importance`: feature importance score for the last parameter of the prior, for K total parameters

## Example applications

### Differential expression

In this example, users have estimates of log-fold changes in gene expression between conditions and their standard errors from a differential expression workflow, and would like to estimate log-fold changes with greater power (e.g. for lowly-expressed genes with noisy estimates).

#### Likelihood

We define $\ell_\text{DE}^{(i)}$ and $\ell_i$ as the estimated and true log-fold change in expression respectively for gene $i$, and $s_i$ as the standard error for the estimate. Then, we define the likelihood for $\ell_i$ as

$$\ell_\text{DE}^{(i)} \mid \ell_i \sim \text{Normal}(\ell_i, s_i^2)$$

#### Prior

We describe two potential priors that one may choose to try. The first is a normal prior with parameters $\mu_i$ and $\sigma_i$:

$$\ell_i \sim \text{Normal}(\mu_i, \sigma_i^2)$$

The second is a spike-and-slab prior with parameters $\pi_i$, $\mu_i$, and $\sigma_i$, which assumes that gene $i$ only has a $\pi_i$ probability of being differentially expressed:

$$
\begin{split}
z_i &\sim \text{Bernoulli}(\pi_i) \\
\ell_i | z_i &\sim
\begin{cases}
0, & \text{if}\ z_i=0 \\
\text{Normal}(\mu_i, \sigma_i^2), & \text{if}\ z_i=1
\end{cases}
\end{split}
$$

### Variant burden tests

In this example, users have sequencing data from patients with a disease or (if calling *de novo* mutations) sequencing data from family trios, and would like to identify genes with excess mutational burden in patients (e.g. an excess of missense or LOF variants). One approach is to infer the relative risk for each gene (denoted as $\gamma_i$ for gene $i$), defined as the expected ratio of the number of variants in patients to the number of variants in healthy individuals.

#### Likelihood 

Let $E_i$ be the number of variants we expect to observe for gene $i$ given the study sample size and sequence-dependent mutation rates (e.g. expected counts obtained using the mutational model developed by [Samocha et al. 2014](https://www.nature.com/articles/ng.3050)). Next, let $O_i$ be the number of variants observed in patients for gene $i$. Then, we define the likelihood for $\eta_i$ as 

$$O_i\mid\eta_i \sim \text{Poisson}(\eta_i E_i)$$

#### Prior

Because $\eta_i$ is non-negative, one may want to choose a gamma prior with parameters $\alpha_i$ and $\beta_i$:

$$\eta_i \sim \text{Gamma}(\alpha_i, \beta_i)$$

