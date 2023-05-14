# GeneBayes

## Custom usage

To use GeneBayes for other applications, please start with the file `examples/genebayes.o_over_e.py`, which contains an example prior and likelihood, and modify the prior/likelihood as follows: 

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

