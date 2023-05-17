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

from torchquad import Boole, set_up_backend

torch.set_default_dtype(torch.float64)
set_up_backend("torch", data_type="float64")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print = partial(print, flush=True)


class PriorScore(LogScore):
    def score(self, y):
        """
        log score, -log P(y), for prior distribution P
        """
        params = torch.tensor(self.params_transf, device=device)
        params.requires_grad = True

        score = 0
        grad = torch.zeros(self.n_params, y.shape[0])

        for idx in torch.split(torch.randperm(y.shape[0]), split_size_or_sections=1):
            boole = Boole()
            domain = torch.tensor([[INTEGRATION_LB, INTEGRATION_UB]], device=device)
            result = boole.integrate(
                partial(log_prob, Prior.distribution, idx, params, y),
                dim=1, N=int(N_INTEGRATION_PTS),
                integration_domain=domain
            )
            assert result > 0
            result = -(torch.log(result) + torch.max(y[idx]))
            result.backward()
            score += result.item()
            grad += params.grad
            params.grad = None

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
            print(f"param #{i} - min: {p0}, 1st percentile: {p1}, "
                  f"99th percentile: {p99}, max: {p100}")

        return self.gradient

    def metric(self):
        all_grad = []
        for i in range(N_METRIC):
            params = torch.tensor(self.params_transf, device=device)
            params.requires_grad = True

            hs = Prior.distribution(params).sample()
            loss = Prior.distribution(params).log_prob(hs)
            loss = -torch.sum(loss)
            loss.backward()

            params.grad[Prior.positive, :] *= params[Prior.positive, :]
            all_grad.append(params.grad.T.detach().cpu())
            params.grad = None

        grad = np.stack(all_grad)
        grad = np.mean(np.einsum("sik,sij->sijk", grad, grad), axis=0)

        return grad


def log_prob(prior_dist, idx, params, gene_likelihoods, hs):
    hs = torch.squeeze(hs, dim=1)

    # p(hs)
    prior_p = torch.exp(
        prior_dist([p[idx] for p in params]).log_prob(hs))

    # p(y|hs)
    lh = likelihood(idx, hs, gene_likelihoods)

    return prior_p * lh


def likelihood(idx, hs, gene_likelihoods):
    '''
    p(y|parameter)
    '''
    S_GRID = torch.tensor([0.] + np.exp(np.linspace(np.log(1e-8), 0, num=100)).tolist(), device=device)
    likelihoods = gene_likelihoods[idx.expand(hs.shape[0])] - torch.max(gene_likelihoods[idx])

    with torch.no_grad():
        left_idx = torch.searchsorted(S_GRID, hs, right=True) - 1
    left_hs = S_GRID[left_idx]
    right_hs = S_GRID[left_idx + 1]

    idx0 = torch.arange(likelihoods.shape[0])
    left_likelihood = likelihoods[idx0, left_idx]
    right_likelihood = likelihoods[idx0, left_idx + 1]

    sp_1 = left_likelihood + torch.log((right_hs - hs) / (right_hs - left_hs))
    sp_2 = right_likelihood + torch.log((hs - left_hs) / (right_hs - left_hs))

    success_p = torch.exp(sp_1) + torch.exp(sp_2)

    return success_p


class Prior(RegressionDistn):
    n_params = 2
    positive = np.array([False, True])
    scores = [PriorScore]

    def __init__(self, params):
        self._params = params
        self.params_transf = np.copy(params)
        self.params_transf[Prior.positive] = np.exp(self.params_transf[Prior.positive])

    def distribution(params):
        """
        LogitNormal distribution
        https://en.wikipedia.org/wiki/Logit-normal_distribution
        """
        return dist.TransformedDistribution(
            dist.Normal(params[0], params[1]),
            transforms=[dist.SigmoidTransform()])

    def fit(y):
        """
        fit initial prior distribution for all genes
        """
        params = []
        for i in range(Prior.n_params):
            if Prior.positive[i]:
                params.append(torch.tensor(1., requires_grad=True, device=device))
            else:
                params.append(torch.tensor(0., requires_grad=True, device=device))
        optimizer = torch.optim.AdamW(params, lr=LR_INIT)

        for i in range(Prior.n_params):
            params[i] = params[i].expand(y.shape[0])

        lr_stage = 0

        for i in range(MAX_EPOCHS_INIT):
            loss = 0

            for idx in torch.split(torch.randperm(y.shape[0]), split_size_or_sections=1):
                optimizer.zero_grad()
                boole = Boole()
                domain = torch.tensor([[INTEGRATION_LB, INTEGRATION_UB]], device=device)

                result = boole.integrate(
                    partial(log_prob, Prior.distribution, idx, params, y),
                    dim=1, N=int(N_INTEGRATION_PTS),
                    integration_domain=domain
                )
                result = -(torch.log(result) + torch.max(y[idx]))
                result.backward()
                optimizer.step()
                loss += result.item()

            print("loss", loss, "params", [p[0] for p in params], "lr", optimizer.param_groups[0]['lr'])

            if i == 0 or loss < min_loss:
                min_loss = loss
                min_epoch = i
            if i - min_epoch >= PATIENCE_INIT:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                lr_stage += 1

            if lr_stage > 2:
                break

        params = np.array([p[0].item() for p in params])
        params[Prior.positive] = np.log(params[Prior.positive])

        return params

    @property
    def params(self):
        return self.params_transf


def posterior(prob_y, idx, params, y, hs):
    '''
    return posterior pdf, p(y,hs)/p(y)
    '''
    prob_y_hs = log_prob(Prior.distribution, idx, params, y, hs)
    return torch.exp(torch.log(prob_y_hs) - torch.log(prob_y))


def format_lh(GENE_LIKELIHOODS, data, args):
    lh = []
    for gene in data["hgnc"]:
        lh_gene = GENE_LIKELIHOODS[gene]['stop_gained'][args.sg][:] + GENE_LIKELIHOODS[gene]['splice_donor_variant'][
                                                                          args.sd][:] + \
                  GENE_LIKELIHOODS[gene]['splice_acceptor_variant'][args.sa][:]
        lh.append(lh_gene.to(device))
    lh = torch.stack(lh)
    return lh


def output_prior_posterior(model, feature_table, y, max_iter=None):
    print("calculating posterior distributions...")
    # score = model.score(X_val, y_val)
    X = feature_table.drop(["ensg", "hgnc", "chrom"], axis=1)
    params = torch.tensor(model.pred_dist(X.to_numpy(), max_iter=max_iter)[:].params,
                          device=device)
    # prior
    prior_mean = torch.mean(Prior.distribution(params).sample(torch.Size([10000])),
                            dim=0).detach().cpu().numpy()

    # posterior
    post_mean = []
    lower_95, upper_95 = [], []
    pdf = []
    # samples = []

    for idx in range(params.shape[1]):
        idx = torch.tensor([idx], device=device)

        # compute p(y)
        boole = Boole()
        domain = torch.tensor([[INTEGRATION_LB, INTEGRATION_UB]], device=device)

        prob_y = boole.integrate(
            partial(log_prob, Prior.distribution, idx, params, y),
            dim=1, N=int(N_INTEGRATION_PTS),
            integration_domain=domain
        )

        # compute posterior pdf
        boole = Boole()
        domain = torch.tensor([[INTEGRATION_LB, INTEGRATION_UB]], device=device)
        grid_points, h, n_per_dim = boole.calculate_grid(int(N_INTEGRATION_PTS), domain)
        post_pdf, _ = boole.evaluate_integrand(
            partial(posterior, prob_y, idx, params, y),
            grid_points
        )
        # PDF.append(post_pdf.detach().cpu().numpy())

        # expected posterior hs
        post = boole.calculate_result(post_pdf * torch.squeeze(grid_points), 1, n_per_dim, h)
        post_mean.append(post.item())

        # sample from posterior distribution
        # probabilities = post_pdf.detach().cpu().numpy()
        # probabilities = probabilities / np.sum(probabilities)
        # samps = np.random.choice(grid_points.detach().cpu().numpy().flatten(), p=probabilities, size=1000)
        # samples.append(samps)

        # nonuniform posterior pdf (log)
        grid_points_nonunif = torch.tensor(np.exp([np.linspace(
            np.log(INTEGRATION_LB), np.log(INTEGRATION_UB), num=500)]).tolist(), device=device)
        post_pdf_nonunif = posterior(prob_y, idx, params, y, grid_points_nonunif)
        post_pdf_nonunif = torch.squeeze(post_pdf_nonunif)
        pdf.append(post_pdf_nonunif.detach().cpu().numpy())

        # compute lower/upper bounds
        grid_size = (INTEGRATION_UB - INTEGRATION_LB) / N_INTEGRATION_PTS
        cdf = torch.cumulative_trapezoid(post_pdf, dx=grid_size, dim=0)
        lower = (torch.argsort(torch.abs(cdf - 0.05))[0] + 1).detach().cpu().numpy() * grid_size + INTEGRATION_LB
        upper = (torch.argsort(torch.abs(cdf - 0.95))[0] + 1).detach().cpu().numpy() * grid_size + INTEGRATION_LB
        lower_95.append(lower)
        upper_95.append(upper)

    # samples = pd.DataFrame(samples)
    # samples["ensg"] = feature_table["ensg"].tolist()
    # samples.to_csv(args.out_samp, sep='\t')

    pdf = pd.DataFrame(pdf,
                       columns=np.exp(np.linspace(np.log(INTEGRATION_LB), np.log(INTEGRATION_UB), num=500)))
    pdf = pd.concat([feature_table[["chrom", "ensg", "hgnc"]].copy(),
                     pdf], axis=1)
    pdf.to_csv(args.out_prefix + ".posterior_density.tsv", sep='\t', index=None)

    output = feature_table[["chrom", "ensg", "hgnc"]].copy()
    output.loc[:,"prior_mean"] = prior_mean
    output.loc[:,"post_mean"] = post_mean
    output.loc[:,"post_lower_95"] = lower_95
    output.loc[:,"post_upper_95"] = upper_95
    output.to_csv(args.out_prefix + ".per_gene_estimates.tsv", sep='\t', index=None)

    ### feature importance metrics ###
    features = X.columns
    importance = {"feature": features}
    for i in range(model.feature_importances_.shape[0]):
        importance["param%s_importance" % i] = model.feature_importances_[i]
    importance = pd.DataFrame(importance)
    importance.to_csv(args.out_prefix + ".feature_importance.tsv", sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="likelihood",
                        help="pkl file with pre-computed likelihoods for s_het.")
    parser.add_argument(dest="features",
                        help="tsv containing gene features. Also known as X or independent variables.")
    parser.add_argument(dest="out_prefix",
                        help="Prefix for the output files.")
    parser.add_argument(dest="integration_lb", type=float,
                        help="Lower bound for numerical integration - the smallest value that you expect for the gene property of interest.")
    parser.add_argument(dest="integration_ub", type=float,
                        help="Upper bound for numerical integration - the largest value you expect for the gene property of interest.")
    parser.add_argument("--sg_err", dest="sg", type=int, default=0,
                        help="Index for the error rate to use for stop gain variants.")
    parser.add_argument("--sd_err", dest="sd", type=int, default=0,
                        help="Index for the error rate to use for splice donor variants.")
    parser.add_argument("--sa_err", dest="sa", type=int, default=0,
                        help="Index for the error rate to use splice acceptor variants.")
    parser.add_argument("--model", dest="model", default=None,
                        help="Load pretrained model to get estimates of the gene property.")
    parser.add_argument("--n_integration_pts", dest="n_integration_pts", type=int, default=1000, required=False,
                        help="Number of points for numerical integration. Larger values increase can improve accuracy but also increase training time and/or numerical instability.")
    parser.add_argument("--total_iterations", dest="total_iterations", type=int, default=1000, required=False,
                        help="Maximum number of iterations. The actual number of iterations may be lower if using early stopping (see the 'train' option)")
    parser.add_argument("--train", dest="train", type=bool, default=True, required=False,
                        help="If True, then chromosomes 2, 4, 6 will be held out for validation and training will end when the loss on these chromosomes stops decreasing. Otherwise, the model will train on all genes for the number of iterations specified in 'total_iterations'.")
    parser.add_argument("--lr", dest="lr", type=float, default=0.05, required=False,
                        help="Learning rate for NGBoost. Smaller values may improve accuracy but also increase training time. Typical values: 0.01 to 0.2.")
    parser.add_argument("--max_depth", dest="max_depth", type=int, default=3, required=False,
                        help="XGBoost parameter. See https://xgboost.readthedocs.io/en/stable/python/python_api.html.")
    parser.add_argument("--n_trees_per_iteration", dest="n_estimators", type=int, default=1, required=False,
                        help="XGBoost parameter.")
    parser.add_argument("--min_child_weight", dest="min_child_weight", type=float, required=False,
                        help="XGBoost parameter.")
    parser.add_argument("--reg_alpha", dest="reg_alpha", type=float, required=False,
                        help="XGBoost parameter.")
    parser.add_argument("--reg_lambda", dest="reg_lambda", type=float, required=False,
                        help="XGBoost parameter.")
    parser.add_argument("--subsample", dest="subsample", type=float, required=False,
                        help="XGBoost parameter.")

    args = parser.parse_args()

    global LR_INIT, N_METRIC, MAX_EPOCHS_INIT, PATIENCE_INIT, INTEGRATION_LB, INTEGRATION_UB

    MAX_EPOCHS_INIT = 100 #args.max_epochs_init
    LR_INIT = 1e-2 #args.lr_init
    PATIENCE_INIT = 1 #args.patience_init
    N_METRIC = 1000 #args.n_metric
    INTEGRATION_LB = args.integration_lb
    INTEGRATION_UB = args.integration_ub
    N_INTEGRATION_PTS = args.n_integration_pts

    ### load likelihoods and training data ###
    feature_table = pd.read_csv(args.features, sep='\t')
    y = format_lh(pickle.load(open(args.likelihood, 'rb')), feature_table, args)

    ### train ###
    train_idx = ~feature_table["chrom"].isin(["chr1", "chr3", "chr5", "chr2", "chr4", "chr6"])
    val_idx = feature_table["chrom"].isin(["chr2", "chr4", "chr6"])
    train_idx = feature_table["chrom"].isin(["chr22"])
    val_idx = feature_table["chrom"].isin(["chr21"])

    X = feature_table.drop(["ensg", "hgnc", "chrom"], axis=1)
    X = X.to_numpy()

    X_train, y_train = X[train_idx], y[train_idx.to_numpy()]
    X_val, y_val = X[val_idx], y[val_idx.to_numpy()]

    if args.model is not None:  # already have pretrained model, want to obtain predictions
        model = pickle.load(open(args.model, 'rb'))
    else:  # train model and make predictions
        xgb_params = {"max_depth": args.max_depth,
                      "reg_alpha": args.reg_alpha,
                      "reg_lambda": args.reg_lambda,
                      "min_child_weight": args.min_child_weight,
                      "eta": 1.0,
                      "subsample": args.subsample,
                      "n_estimators": args.n_estimators}
        xgb_params = {k: v for k, v in xgb_params.items() if v is not None}

        if not torch.cuda.is_available():
            learner = xgb.XGBRegressor(
                tree_method="hist",
                **xgb_params
            )
        else:
            learner = xgb.XGBRegressor(
                gpu_id=0,
                tree_method="gpu_hist",
                **xgb_params
            )

        print("X.shape", X.shape)
        print("X_train.shape", X_train.shape)
        print("X_val.shape", X_val.shape)

        if args.train:
            model = NGBRegressor(n_estimators=args.total_iterations, Dist=Prior, Base=learner, Score=PriorScore,
                                 verbose_eval=1, learning_rate=args.lr, natural_gradient=True
                                 ).fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=10)
        else:
            model = NGBRegressor(n_estimators=args.total_iterations, Dist=Prior, Base=learner, Score=PriorScore,
                                 verbose_eval=1, learning_rate=args.lr, natural_gradient=True
                                 ).fit(X, y)

        f = open(args.out_prefix + ".model", 'wb')
        pickle.dump(model, f)
        f.close()

    if model.best_val_loss_itr is not None:
        output_prior_posterior(model, feature_table, y, model.best_val_loss_itr)
    else:
        output_prior_posterior(model, feature_table, y)
