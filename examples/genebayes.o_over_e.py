import pickle
import numpy as np
import pandas as pd
import argparse
from functools import partial

import xgboost as xgb
from ngboost import NGBRegressor
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

import torch
import torch.distributions as dist

torch.set_default_dtype(torch.float64)
print = partial(print, flush=True)


class PriorScore(LogScore):
    def score(self, y):
        """
        log score, -log P(y), for prior distribution P
        """
        params = torch.tensor(self.params_transf, device=DEVICE)
        params = params.unsqueeze(dim=-1)
        params.requires_grad = True

        score = 0
        grad = torch.zeros(self.n_params, y.shape[0], device=DEVICE)

        for idx in torch.split(
            torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
        ):
            grid = torch.linspace(
                INTEGRATION_LB, INTEGRATION_UB, N_INTEGRATION_PTS, device=DEVICE
            )
            grid = grid.expand(len(idx), -1)

            integrand = log_prob(Prior.distribution, idx, params, y, grid)
            result = torch.trapezoid(integrand, grid, dim=1)  # [batch_size]

            result = -torch.sum(torch.log(result))
            result.backward()
            score += result.item()
            grad += torch.squeeze(params.grad, dim=-1)
            params.grad = None

        params = params.squeeze(dim=-1)
        grad[Prior.positive, :] *= params[Prior.positive, :]
        self.gradient = grad.T.cpu().detach().numpy()

        return score

    def d_score(self, data):
        """
        derivative of the score
        """
        for i in range(Prior.n_params):
            p0 = np.min(self.params_transf[i])
            p1 = np.quantile(self.params_transf[i], 0.01)
            p99 = np.quantile(self.params_transf[i], 0.99)
            p100 = np.max(self.params_transf[i])
            print(
                f"param #{i} - min: {p0}, 1st percentile: {p1}, "
                f"99th percentile: {p99}, max: {p100}"
            )

        return self.gradient


def log_prob(prior_dist, idx, params, y, gene_property):
    # log p(gene_property)
    prior_p = prior_dist([p[idx] for p in params]).log_prob(gene_property)

    # log p(y|gene_property)
    lambd = gene_property
    expected = y[idx, 1:]
    observed = y[idx, :1]

    lh = dist.Poisson(lambd * expected).log_prob(observed)

    return torch.exp(prior_p + lh)


class Prior(RegressionDistn):
    n_params = 2
    positive = np.array([True, True])
    scores = [PriorScore]

    def __init__(self, params):
        self._params = params
        self.params_transf = np.copy(params)
        self.params_transf[Prior.positive] = np.exp(self.params_transf[Prior.positive])

    @staticmethod
    def distribution(params):
        """
        Beta distribution
        alpha = params[0], beta = params[1]
        """
        return dist.Beta(params[0], params[1])

    @staticmethod
    def fit(y):
        """
        fit initial prior distribution for all genes
        """
        print("fitting initial distribution...")
        params = []
        for i in range(Prior.n_params):
            if Prior.positive[i]:
                params.append(torch.tensor(1.0, requires_grad=True, device=DEVICE))
            else:
                params.append(torch.tensor(0.0, requires_grad=True, device=DEVICE))
        optimizer = torch.optim.AdamW(params, lr=LR_INIT)

        for i in range(Prior.n_params):
            params[i] = params[i].expand(y.shape[0], 1)

        lr_stage = 0

        for i in range(MAX_EPOCHS_INIT):
            loss = 0

            for idx in torch.split(
                torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
            ):
                optimizer.zero_grad()

                grid = torch.linspace(
                    INTEGRATION_LB, INTEGRATION_UB, N_INTEGRATION_PTS, device=DEVICE
                ).expand(len(idx), -1)

                integrand = log_prob(Prior.distribution, idx, params, y, grid)
                result = torch.trapezoid(integrand, grid, dim=1)  # [batch_size]

                result = -torch.sum(torch.log(result))

                result.backward()

                optimizer.step()
                loss += result.item()

            print(
                "loss",
                loss,
                "params",
                [p[0].item() for p in params],
                "lr",
                optimizer.param_groups[0]["lr"],
            )

            if i == 0 or loss < min_loss:
                min_loss = loss
                min_epoch = i
            if i - min_epoch >= PATIENCE_INIT:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 10
                lr_stage += 1

            if lr_stage > 1:
                break

        params = [p[0].item() for p in params]
        print("initial params", params)
        params = np.array(params)
        params[Prior.positive] = np.log(params[Prior.positive])

        return params

    @property
    def params(self):
        return self.params_transf


def posterior(prob_y, idx, params, y, gene_property):
    """
    return posterior pdf, p(y,gene_property)/p(y)
    """
    prob_joint = log_prob(Prior.distribution, idx, params, y, gene_property)
    return torch.exp(torch.log(prob_joint) - torch.log(prob_y))


def output_prior_posterior(model, feature_table, y, max_iter=None):
    print("calculating posterior distributions...")
    X = feature_table.drop(
        [
            x
            for x in [GENE_COLUMN, "ensg", "hgnc", "chrom"]
            if x in feature_table.columns
        ],
        axis=1,
    )
    params = torch.tensor(
        model.pred_dist(X.to_numpy(), max_iter=max_iter)[:].params, device=DEVICE
    )
    # prior
    prior_mean = (
        torch.mean(Prior.distribution(params).sample(torch.Size([10000])), dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    # posterior
    post_mean = []
    lower_95, upper_95 = [], []

    for idx in range(params.shape[1]):
        idx = torch.tensor([idx], device=DEVICE)

        # compute p(y)
        grid = torch.linspace(
            INTEGRATION_LB, INTEGRATION_UB, N_INTEGRATION_PTS, device=DEVICE
        ).expand(len(idx), -1)

        integrand = log_prob(Prior.distribution, idx, params, y, grid)
        prob_y = torch.trapezoid(integrand, grid, dim=-1)

        # compute posterior pdf
        post_pdf = posterior(prob_y, idx, params, y, grid)  # [1, n_integration_pts]

        # expected posterior gene_property
        post = torch.trapezoid(post_pdf * grid, grid, dim=1)
        post_mean.append(post.item())

        # compute lower/upper bounds
        grid_size = (INTEGRATION_UB - INTEGRATION_LB) / N_INTEGRATION_PTS
        cdf = torch.cumulative_trapezoid(post_pdf.flatten(), dx=grid_size, dim=0)
        lower = (
            torch.argsort(torch.abs(cdf - 0.05))[0] + 1
        ).detach().cpu().numpy() * grid_size + INTEGRATION_LB
        upper = (
            torch.argsort(torch.abs(cdf - 0.95))[0] + 1
        ).detach().cpu().numpy() * grid_size + INTEGRATION_LB
        lower_95.append(lower)
        upper_95.append(upper)

    feature_reset = feature_table.reset_index()
    output = feature_reset[
        [x for x in [GENE_COLUMN, "ensg", "hgnc"] if x in feature_reset.columns]
    ].copy()
    output.loc[:, "prior_mean"] = prior_mean
    output.loc[:, "post_mean"] = post_mean
    output.loc[:, "post_lower_95"] = lower_95
    output.loc[:, "post_upper_95"] = upper_95
    output.to_csv(args.out_prefix + ".per_gene_estimates.tsv", sep="\t", index=None)

    ### feature importance metrics ###
    features = X.columns
    importance = {"feature": features}
    for i in range(model.feature_importances_.shape[0]):
        importance["param%s_importance" % i] = model.feature_importances_[i]
    importance = pd.DataFrame(importance)
    importance.to_csv(
        args.out_prefix + ".feature_importance.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--response",
        dest="response",
        required=True,
        help="tsv containing data that can be related to the gene property of interest through a likelihood function. Also known as y or dependent variable. Other information necessary for computing the likelihood can also be included in this file.",
    )
    required.add_argument(
        "--features",
        dest="features",
        required=True,
        help="tsv containing gene features. Also known as X or independent variables.",
    )
    required.add_argument(
        "--out", 
        dest="out_prefix", 
        required=True,
        help="Prefix for the output files."
    )
    required.add_argument(
        "--integration_lb",
        dest="integration_lb",
        type=float,
        required=True,
        help="Lower bound for numerical integration - the smallest value that you expect for the gene property of interest.",
    )
    required.add_argument(
        "--integration_ub",
        dest="integration_ub",
        type=float,
        required=True,
        help="Upper bound for numerical integration - the largest value you expect for the gene property of interest.",
    )
    parser.add_argument(
        "--gene_column",
        dest="gene_column",
        required=False,
        help="Name of the column containing gene names/IDs. If not specified, will look for 'ensg' and 'hgnc' columns.",
    )
    parser.add_argument(
        "--n_integration_pts",
        dest="n_integration_pts",
        type=int,
        default=1001,
        required=False,
        help="Number of points for numerical integration. Larger values can improve accuracy but also increase training time and/or numerical instability.",
    )
    parser.add_argument(
        "--total_iterations",
        dest="total_iterations",
        type=int,
        default=500,
        required=False,
        help="Maximum number of iterations. The actual number of iterations may be lower if using early stopping.",
    )
    parser.add_argument(
        "--early_stopping_iter",
        dest="early_stopping_iter",
        type=int,
        default=10,
        required=False,
        help="If >0, chromosomes 2, 4, 6 will be held out for validation and training will end when the loss on these chromosomes stops decreasing for the specified number of iterations. Otherwise, the model will train on all genes for the number of iterations specified in 'total_iterations'.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.05,
        required=False,
        help="Learning rate for NGBoost. Smaller values may improve accuracy but also increase training time. Typical values: 0.01 to 0.2.",
    )
    parser.add_argument(
        "--max_depth",
        dest="max_depth",
        type=int,
        default=3,
        required=False,
        help="XGBoost parameter. See https://xgboost.readthedocs.io/en/stable/python/python_api.html.",
    )
    parser.add_argument(
        "--n_trees_per_iteration",
        dest="n_estimators",
        type=int,
        default=1,
        required=False,
        help="XGBoost parameter, n_estimators",
    )
    parser.add_argument(
        "--min_child_weight",
        dest="min_child_weight",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--reg_alpha",
        dest="reg_alpha",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--reg_lambda",
        dest="reg_lambda",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--subsample",
        dest="subsample",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        required=False,
        help="Batch size for integration. Larger values can improve speed but use more memory.",
    )

    # parser.add_argument("--model", dest="model", default=None, description="Provide a pretrained model to obtain predictions for that model.")
    # parser.add_argument("--lr_init", dest="lr_init", type=float, default=1e-2, )
    # parser.add_argument("--patience_init", dest="patience_init", type=int, default=1)
    # parser.add_argument("--max_epochs_init", dest="max_epochs_init", type=int, default=100)

    args = parser.parse_args()
    print(vars(args))

    global LR_INIT, MAX_EPOCHS_INIT, PATIENCE_INIT, INTEGRATION_LB, INTEGRATION_UB, BATCH_SIZE, DEVICE

    MAX_EPOCHS_INIT = 100  # args.max_epochs_init
    LR_INIT = 1e-2  # args.lr_init
    PATIENCE_INIT = 1  # args.patience_init
    INTEGRATION_LB = args.integration_lb
    INTEGRATION_UB = args.integration_ub
    N_INTEGRATION_PTS = args.n_integration_pts
    GENE_COLUMN = args.gene_column
    BATCH_SIZE = args.batch_size

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEVICE == torch.device("cuda:0"):
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    ### Load data ###
    print("Loading data...")
    feature_table = pd.read_csv(args.features, sep="\t", index_col="hgnc")
    response_data = pd.read_csv(args.response, sep="\t", index_col="hgnc")

    # Filter to common genes
    common_genes = list(set(feature_table.index) & set(response_data.index))
    feature_table = feature_table.loc[common_genes]
    response_data = response_data.loc[common_genes]

    ### Prepare training/validation split ###
    val_chroms = ["chr2", "chr4", "chr6"]
    train_idx = ~feature_table["chrom"].isin(val_chroms)
    val_idx = feature_table["chrom"].isin(val_chroms)
    print(f"Training genes: {train_idx.sum()}, Validation genes: {val_idx.sum()}")

    ### Extract features and response ###
    # Define metadata columns to exclude from features
    meta_cols = [GENE_COLUMN, "ensg", "hgnc", "chrom"]
    feature_cols = [col for col in meta_cols if col in feature_table.columns]
    response_cols = [col for col in meta_cols if col in response_data.columns]

    X = feature_table.drop(feature_cols, axis=1).to_numpy()
    y = response_data.drop(response_cols, axis=1).to_numpy()

    # Split train/validation
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Convert to torch tensors
    y_train = torch.tensor(y_train, device=DEVICE)
    y_val = torch.tensor(y_val, device=DEVICE)
    y = torch.tensor(y, device=DEVICE)

    xgb_params = {
        "max_depth": args.max_depth,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "eta": 1.0,
        "subsample": args.subsample,
        "n_estimators": args.n_estimators,
    }
    xgb_params = {k: v for k, v in xgb_params.items() if v is not None}

    if not torch.cuda.is_available():
        learner = xgb.XGBRegressor(tree_method="hist", **xgb_params)
    else:
        learner = xgb.XGBRegressor(gpu_id=0, tree_method="gpu_hist", **xgb_params)

    if args.early_stopping_iter > 0:
        model = NGBRegressor(
            n_estimators=args.total_iterations,
            Dist=Prior,
            Base=learner,
            Score=PriorScore,
            verbose_eval=1,
            learning_rate=args.lr,
            natural_gradient=False,
        ).fit(
            X_train,
            y_train,
            X_val=X_val,
            Y_val=y_val,
            early_stopping_rounds=args.early_stopping_iter,
        )
    else:
        model = NGBRegressor(
            n_estimators=args.total_iterations,
            Dist=Prior,
            Base=learner,
            Score=PriorScore,
            verbose_eval=1,
            learning_rate=args.lr,
            natural_gradient=False,
        ).fit(X, y)

    f = open(args.out_prefix + ".model", "wb")
    pickle.dump(model, f)
    f.close()

    if model.best_val_loss_itr is not None:
        output_prior_posterior(model, feature_table, y, model.best_val_loss_itr)
    else:
        output_prior_posterior(model, feature_table, y)
