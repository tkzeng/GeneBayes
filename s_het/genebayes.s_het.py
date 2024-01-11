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


def integrate(these_params, these_likelihoods, logit=True):
    result = 0

    grid = torch.linspace(-1., 1., N_INTEGRATION_PTS).expand(1, -1)
    means = Prior.get_mean(these_params)
    sds = Prior.get_sd(these_params)
    grid = means.expand(-1, 1) + grid * 8 * sds.expand(-1, 1)

    eval_at_pts, prior_p = log_prob(Prior.distribution,
                                    these_params,
                                    these_likelihoods,
                                    grid,
                                    logit=logit)

    m = torch.max(eval_at_pts, dim=1, keepdim=True)[0]
    pdf = eval_at_pts - m

    result += torch.exp(m.squeeze())*torch.trapezoid(
        torch.exp(pdf), grid, dim=1
    )

    m = torch.max(prior_p, dim=1, keepdim=True)[0]
    check = torch.exp(m.squeeze())*torch.trapezoid(
        torch.exp(prior_p - m), grid, dim=1
    )
    if np.any(np.abs(check.detach().cpu().numpy() - 1.) > 1e-5):
        bad_idx = np.argmax(np.abs(check.detach().cpu().numpy()-1.))
        print('Discrepancy of ',
              np.max(np.abs(check.detach().cpu().numpy() - 1.)),
              'found while integrating')
        print('params were', [p[bad_idx] for p in these_params])
        print(torch.exp(prior_p[bad_idx, 0]), torch.exp(prior_p[bad_idx, -1]))
        print('mean was', means[bad_idx])
        print('sd was', sds[bad_idx])
        print('grid was', grid[bad_idx])

    return result


class PriorScore(LogScore):
    hessian = None

    def score(self, y, compute_grad=False):
        """
        log score, -log P(y), for prior distribution P
        """
        params = torch.tensor(self.params_transf, device=DEVICE)
        params = params.unsqueeze(dim=-1)
        params.requires_grad = True

        score = 0
        if compute_grad:
            grad = np.zeros((self.n_params, y.shape[0]))

            # Just for debugging
            # grad_param_plus = np.zeros(
            #     (self.n_params, self.n_params, y.shape[0])
            # )

            hessian = np.zeros(
                (y.shape[0], self.n_params, self.n_params)
            )
        y = torch.tensor(y, device=DEVICE)

        for idx in torch.split(
            torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
        ):
            idx_numpy = idx.cpu().numpy()
            these_params = torch.tensor(
                self.params_transf[:, idx_numpy], device=DEVICE
            ).unsqueeze(dim=-1)

            if compute_grad:
                these_params.requires_grad = True

            result = integrate(these_params, y[idx])
            result = -torch.sum(torch.log(result)+torch.max(y[idx], dim=1)[0])

            if compute_grad:
                this_grad = torch.autograd.grad(
                    result,
                    these_params,
                    retain_graph=True,
                    create_graph=True
                )[0][:, :, 0]
                grad[:, idx_numpy] = [
                    tg.detach().squeeze().cpu().numpy() for tg in this_grad
                ]

            score += result.item()

            # This for loop is just for debugging and computes stuff needed to
            # numerically compute the hessian
            '''
            for perturb_idx in range(self.n_params):
                perturb_params = torch.tensor(
                    self.params_transf[:, idx_numpy], device=DEVICE
                ).unsqueeze(dim=-1)
                perturb_params[perturb_idx] += 1e-6
                perturb_params = perturb_params.detach().clone()
                perturb_params.requires_grad = True
                result = integrate(perturb_params, y[idx])
                result = -torch.sum(torch.log(result))
                result.backward()
                grad_param_plus[perturb_idx, :, idx_numpy] = (
                    perturb_params.grad.squeeze().T.cpu().numpy()
                )
            '''
            if compute_grad:
                for p_idx in range(self.n_params):
                    torch.sum(this_grad[p_idx, :]).backward(retain_graph=True)
                    this_hess_row = these_params.grad.squeeze().T.detach().cpu().numpy()
                    hessian[idx_numpy, p_idx, :] = this_hess_row
                    these_params.grad.zero_()

        # This is just for debugging, check that hessian is close to numerical
        # hessian
        '''
        numerical_hessian = np.zeros_like(hessian)
        for p_plus in range(self.n_params):
            for p in range(self.n_params):
                numerical_hessian[:, p_plus, p] = (
                    (grad_param_plus[p_plus, p, :] - grad[p, :])
                    / 1e-6
                )

        print('First gradient is')
        print(grad[:, 0])
        print('First hessian is')
        print(hessian[0])
        print('Numerical hessian is')
        print(numerical_hessian[0])

        abs_diff = np.abs(hessian - numerical_hessian)
        error = abs_diff.sum(axis=(1, 2))
        print('Worst hessians:', np.argmax(error))
        print(hessian[np.argmax(error)])
        print(numerical_hessian[np.argmax(error)])
        avg_hess = 0.5 * (np.abs(hessian) + np.abs(numerical_hessian))
        print('worst absolute difference is', np.max(abs_diff))
        print('worst relative difference is',
              np.max(abs_diff / (avg_hess + 1e-5)))
        '''
        # Ensure symmetry
        if compute_grad:
            if not np.allclose(hessian, np.transpose(hessian, [0, 2, 1])):
                print(
                    'Encountered asymmetric hessian with maximal deviance of',
                    np.max(np.abs(hessian - np.transpose(hessian, [0, 2, 1])))
                )

            hessian = 0.5 * (hessian + np.transpose(hessian, [0, 2, 1]))

            for p1, p1_pos in enumerate(Prior.positive):
                for p2, p2_pos in enumerate(Prior.positive):
                    if p1 == p2 and p1_pos:
                        hessian[:, p1, p2] = (
                            hessian[:, p1, p2] * self.params_transf[p1, :]**2
                            + grad[p1, :] * self.params_transf[p1, :]
                        )
                        continue
                    if p1_pos:
                        hessian[:, p1, p2] *= self.params_transf[p1, :]
                    if p2_pos:
                        hessian[:, p1, p2] *= self.params_transf[p2, :]

            assert not np.any(np.isnan(hessian))
            assert np.allclose(hessian, np.transpose(hessian, [0, 2, 1]))
            self.hessian = hessian

            untransf_grad = np.copy(grad)
            untransf_grad[Prior.positive] *= (
                self.params_transf[Prior.positive, :]
            )
            self.gradient = np.copy(untransf_grad.T)

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

        self.score(data, compute_grad=True)

        return np.copy(self.gradient)

    def metric(self):
        return self.regularized_hessian()

    def regularized_hessian(self):
        print('Mean, percentiles of grad norms',
              np.mean(np.sqrt((self.gradient**2).sum(axis=1))),
              np.percentile(np.sqrt((self.gradient**2).sum(axis=1)),
                            [0, 25, 50, 75, 100]))

        regularized_hessian = np.copy(self.hessian)
        evals, evecs = np.linalg.eigh(self.hessian)
        print('Mean, percentiles of eigenvalues', np.mean(np.abs(evals)),
              np.percentile(np.abs(evals), [0, 25, 50, 75, 100]))
        clipped_evals = np.clip(
            np.abs(evals),
            HESS_REG_1*np.mean(np.abs(evals)),
            # this_reg_val,
            float('inf')
        )
        print('After clipping, Mean, percentiles of eigenvalues',
              np.mean(np.abs(clipped_evals)),
              np.percentile(np.abs(clipped_evals), [0, 25, 50, 75, 100]))

        regularized_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            clipped_evals,
            evecs
        ) + HESS_REG_2*np.mean(np.abs(evals))*np.eye(self.n_params)[None, :, :]

        # The below is just for debugging and checking stuff out.
        '''
        diff = np.abs(evals.min(axis=1)) * (evals.min(axis=1) < 0)
        scale = np.abs(evals).max(axis=1)
        reg = diff + 1e-3*scale + 0.1
        reg = reg[:, None, None]
        print(self.hessian[0])
        print(self.gradient[0], 'Raw gradient')
        this_approx = np.linalg.solve(self.hessian, self.gradient)
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'raw_hessian')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        this_approx = np.linalg.solve(
            self.hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'Pure regularization')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        clipped_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            np.clip(np.abs(evals), 0., float('inf')),
            evecs
        )
        this_approx = np.linalg.solve(
            clipped_hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'Clip eigenvals + reg')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        clipped_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            np.clip(np.abs(evals), 0.1, float('inf')),
            evecs
        )
        this_approx = np.linalg.solve(
            clipped_hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'reg in eigenspace')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        this_approx = np.linalg.solve(
            regularized_hessian,
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'actual regularization')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        '''

        return regularized_hessian


def log_prob(prior_dist, these_params, these_likelihoods, gene_property, logit=True):
    prior_p = prior_dist(these_params, logit).log_prob(gene_property)

    if logit:
        hs = torch.sigmoid(gene_property)
    else: # for calculating the posterior only
        hs = gene_property
    hs = torch.clamp(hs,EPS,1-EPS)

    lh = likelihood(hs, these_likelihoods)

    return (prior_p+lh, prior_p)


def likelihood(hs, likelihoods):
    '''
    p(y|parameter)
    '''
    S_GRID = torch.tensor([0.] +
                          np.exp(np.linspace(np.log(1e-8), 0, num=100)).tolist(),
                          device=DEVICE)
    likelihoods -= torch.max(likelihoods, dim=1, keepdim=True)[0]

    with torch.no_grad():
        left_idx = torch.searchsorted(S_GRID, hs, right=True) - 1

    left_hs = S_GRID[left_idx]
    right_hs = S_GRID[left_idx + 1]

    idx0 = torch.arange(hs.shape[1])
    left_likelihood = []
    right_likelihood = []

    for i in range(likelihoods.shape[0]):
        likelihoods_curr = likelihoods[i].expand(hs.shape[1],-1)
        left_likelihood.append(likelihoods_curr[idx0, left_idx[i,:]])
        right_likelihood.append(likelihoods_curr[idx0, left_idx[i,:]+1])

    left_likelihood = torch.stack(left_likelihood)
    right_likelihood = torch.stack(right_likelihood)

    sp_1 = left_likelihood + torch.log((right_hs - hs) / (right_hs - left_hs))
    sp_2 = right_likelihood + torch.log((hs - left_hs) / (right_hs - left_hs))
    success_p = torch.exp(sp_1) + torch.exp(sp_2)

    return torch.log(torch.maximum(success_p,EPS))


class Prior(RegressionDistn):
    n_params = 2
    positive = np.array([False, True])
    scores = [PriorScore]

    def __init__(self, params):
        self._params = params
        self.params_transf = np.copy(params)
        self.params_transf[Prior.positive] = np.exp(
           self.params_transf[Prior.positive]
        )
        pass

    def distribution(params, logit=True):
        if logit:
            return dist.Normal(params[0], params[1])
        else:
            return dist.TransformedDistribution(
                dist.Normal(params[0], params[1]),
                transforms=[dist.SigmoidTransform()])

    def get_mean(params):
        return params[0]

    def get_sd(params):
        return params[1]

    def fit(y):
        """
        fit initial prior distribution for all genes
        """
        print("fitting initial distribution...")
        params = []
        for param in [0.,np.log(1.)]:
            params.append(
                torch.tensor(param, requires_grad=True, device=DEVICE)
            )
        optimizer = torch.optim.AdamW(params, lr=LR_INIT)

        for i in range(Prior.n_params):
            params[i] = params[i].expand(len(y), 1)

        lr_stage = 0
        min_loss = float('inf')

        for i in range(MAX_EPOCHS_INIT):
            loss = 0

            for idx in torch.split(
                torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
            ):
                optimizer.zero_grad()
                these_params = [torch.exp(p[idx]) if Prior.positive[p_idx]
                                else p[idx] for p_idx, p in enumerate(params)]
                result = integrate(these_params, y[idx])
                result = -torch.sum(torch.log(result)+torch.max(y[idx], dim=1)[0])

                result.backward()
                optimizer.step()
                loss += result.item()

            print(
                "loss",
                loss,
                "params",
                params[0][0].item(),
                params[1][0].item(),
                "lr",
                optimizer.param_groups[0]['lr']
            )

            if i == 0 or loss < min_loss:
                min_loss = loss
                min_epoch = i
            if i - min_epoch >= PATIENCE_INIT:
                optimizer.param_groups[0]['lr'] = (
                    optimizer.param_groups[0]['lr'] / 10
                )
                lr_stage += 1

            if lr_stage > 1:
                break

        params = [p[0].item() for p in params]
        print("initial params", params)
        params = np.array(params)

        return params

    @property
    def params(self):
        return self.params_transf


def compute_posterior(these_params, y, prob_y):
    result = 0
    cdf = 0

    grid = torch.linspace(-1., 1., N_INTEGRATION_PTS).expand(1, -1)
    means = Prior.get_mean(these_params)
    sds = Prior.get_sd(these_params)
    grid = means.expand(-1, 1) + grid * 8 * sds.expand(-1, 1)

    eval_at_pts, prior_p = log_prob(Prior.distribution,
                                    these_params,
                                    y,
                                    grid)

    eval_at_pts -= torch.log(prob_y)
    m = torch.max(eval_at_pts, dim=1, keepdim=True)[0]
    pdf = eval_at_pts - m

    # nonuniform posterior pdf (log)
    grid_points_nonunif = torch.exp(torch.linspace(torch.log(torch.tensor(5e-8)),
                                                   torch.log(torch.tensor(0.995)),
                                                   steps=500)).unsqueeze(dim=0)
    post_pdf_nonunif, _ = log_prob(Prior.distribution,
                                   these_params,
                                   y,
                                   grid_points_nonunif,
                                   logit=False)
    post_pdf_nonunif = post_pdf_nonunif.squeeze()
    post_pdf_nonunif -= torch.log(prob_y)

    # integral of g(x)f(x), where g = sigmoid
    result += torch.exp(m.squeeze())*torch.trapezoid(
        torch.exp(pdf)*torch.sigmoid(grid), grid, dim=1
    )

    cdf += torch.exp(m.squeeze())*torch.cumulative_trapezoid(
        torch.exp(pdf), grid, dim=1
    )

    lower = torch.argsort(torch.abs(cdf - 0.05))[0,0]
    upper = torch.argsort(torch.abs(cdf - 0.95))[0,0]
    lower = (grid[0,lower]+grid[0,lower+1])/2
    upper = (grid[0,upper]+grid[0,upper+1])/2

    lower = torch.sigmoid(lower)
    upper = torch.sigmoid(upper)

    m = torch.max(prior_p, dim=1, keepdim=True)[0]
    check = torch.exp(m.squeeze())*torch.trapezoid(
        torch.exp(prior_p - m), grid, dim=1
    )
    if np.any(np.abs(check.detach().cpu().numpy() - 1.) > 1e-5):
        bad_idx = np.argmax(np.abs(check.detach().cpu().numpy()-1.))
        print('Discrepancy of ',
              np.max(np.abs(check.detach().cpu().numpy() - 1.)),
              'found while integrating')
        print('params were', [p[bad_idx] for p in these_params])
        print(torch.exp(prior_p[bad_idx, 0]), torch.exp(prior_p[bad_idx, -1]))
        print('mean was', means[bad_idx])
        print('sd was', sds[bad_idx])
        print('grid was', grid[bad_idx])

    return result, post_pdf_nonunif, lower, upper


def output_prior_posterior(model, feature_table, y, max_iter=None):
    print("calculating posterior distributions...")

    X = feature_table.to_numpy()
    params = torch.tensor(model.pred_dist(X, max_iter=max_iter)[:].params,
                          device=DEVICE)

    # prior
    prior_mean = torch.mean(Prior.distribution(params, logit=False).sample(torch.Size([10000])),
                            dim=0).detach().cpu().numpy()

    # posterior
    pdf = []
    post_mean = []
    lower_95, upper_95 = [], []

    for idx in range(params.shape[1]):
        idx = torch.tensor([idx], device=DEVICE)

        # compute p(y)
        prob_y = integrate(params[:,idx].unsqueeze(dim=-1), y[idx])

        # compute posterior pdf + expected posterior gene_property
        post, post_pdf, lower, upper = compute_posterior(params[:,idx].unsqueeze(dim=-1), y[idx], prob_y)

        pdf.append(post_pdf.detach().cpu().numpy())
        post_mean.append(post.item())
        lower_95.append(lower.item())
        upper_95.append(upper.item())

    index = pd.DataFrame({"ensg":feature_table.index})
    pdf = pd.DataFrame(pdf, columns=torch.exp(torch.linspace(torch.log(torch.tensor(5e-8)),
                                                             torch.log(torch.tensor(0.995)),
                                                             steps=500)).detach().cpu().tolist())
    pdf = pd.concat([index,pdf], axis=1)
    pdf.to_csv(args.out_prefix + ".posterior_density.tsv", sep='\t', index=None, float_format='%g')

    params = params.detach().cpu().numpy()
    output = pd.DataFrame({"ensg":feature_table.index,
                           "param0":params[0],
                           "param1":params[1],
                           "prior_mean":prior_mean,
                           "post_mean":post_mean,
                           "post_lower_95":lower_95,
                           "post_upper_95":upper_95})
    output.to_csv(args.out_prefix + ".per_gene_estimates.tsv", sep='\t', index=None)

    ### feature importance metrics ###
    importance = {"feature": feature_table.columns}
    for i in range(model.feature_importances_.shape[0]):
       importance["param%s_importance" % i] = model.feature_importances_[i]
    importance = pd.DataFrame(importance)
    importance.to_csv(args.out_prefix + ".feature_importance.tsv", sep='\t', index=False)


def format_lh(GENE_LIKELIHOODS, data, args):
    lh = []
    for gene in data.index:
        if gene not in GENE_LIKELIHOODS.keys():
            lh.append(torch.zeros([101],device=DEVICE))
        else:
            lh_gene = GENE_LIKELIHOODS[gene]['stop_gained'][args.sg][:] + \
                      GENE_LIKELIHOODS[gene]['splice_donor_variant'][args.sd][:] + \
                      GENE_LIKELIHOODS[gene]['splice_acceptor_variant'][args.sa][:]
            lh.append(lh_gene.to(DEVICE))
    lh = torch.stack(lh)
    return lh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train_genes", dest="train_genes", required=False)
    parser.add_argument("--val_genes", dest="val_genes", required=False)
    parser.add_argument(
        "--gene_column",
        dest="gene_column",
        required=False,
        default="ensg",
        help="Name of the column containing gene names/IDs.",
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1, required=False
    )
    parser.add_argument(
        "--n_integration_pts",
        dest="n_integration_pts",
        type=int,
        default=1001,
        required=False,
        help="Number of points for numerical integration. Larger values "
             "increase can improve accuracy but also increase training "
             "time and/or numerical instability.",
    )
    parser.add_argument(
        "--total_iterations",
        dest="total_iterations",
        type=int,
        default=500,
        required=False,
        help="Maximum number of iterations. The actual number of iterations "
             "may be lower if using early stopping (see the 'train' option)",
    )
    parser.add_argument(
        "--early_stopping_iter",
        dest="early_stopping_iter",
        type=int,
        default=10,
        required=False,
        help="If >0, chromosomes 2, 4, 6 will be held out for validation "
             "and training will end when the loss on these chromosomes stops "
             "decreasing for the specified number of iterations. Otherwise, "
             "the model will train on all genes for the number of iterations "
             "specified in 'total_iterations'.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.05,
        required=False,
        help="Learning rate for NGBoost. Smaller values may improve accuracy "
             "but also increase training time. Typical values: 0.01 to 0.2.",
    )
    parser.add_argument(
        "--max_depth",
        dest="max_depth",
        type=int,
        default=3,
        required=False,
        help="XGBoost parameter. See https://xgboost.readthedocs.io/"
             "en/stable/python/python_api.html.",
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
        "--sg_err",
        dest="sg",
        type=int,
        default=0,
        help="Index for the error rate to use for stop gain variants.",
    )
    parser.add_argument(
        "--sd_err",
        dest="sd",
        type=int,
        default=0,
        help="Index for the error rate to use for splice donor variants.",
    )
    parser.add_argument(
        "--sa_err",
        dest="sa",
        type=int,
        default=0,
        help="Index for the error rate to use splice acceptor variants.",
    )
    parser.add_argument(
        "--hess_reg_1",
        dest="hess_reg_1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hess_reg_2",
        dest="hess_reg_2",
        type=float,
        default=0.5,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--response",
        dest="response",
        required=True,
        help="tsv containing data that can be related to the gene property of "
             "interest through a likelihood function. Also known as y or "
             "dependent variable.",
    )
    required.add_argument(
        "--features",
        dest="features",
        required=True,
        help="tsv containing gene features. Also known as X or independent "
             "variables.",
    )
    required.add_argument(
        "--out", dest="out_prefix", help="Prefix for the output files."
    )
    required.add_argument(
        "--integration_lb",
        dest="integration_lb",
        type=float,
        help="Lower bound for numerical integration - the smallest value that "
             "you expect for the gene property of interest.",
    )
    required.add_argument(
        "--integration_ub",
        dest="integration_ub",
        type=float,
        help="Upper bound for numerical integration - the largest value you "
             "expect for the gene property of interest.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Provide a pretrained model to obtain predictions for "
             "that model."
    )

    args = parser.parse_args()
    print(vars(args))

    global LR_INIT, N_METRIC, MAX_EPOCHS_INIT, PATIENCE_INIT
    global DEVICE, ZERO
    global HESS_REG_1, HESS_REG_2

    MAX_EPOCHS_INIT = 2# 500
    LR_INIT = 1e-3
    PATIENCE_INIT = 1
    N_METRIC = 1000
    N_INTEGRATION_PTS = args.n_integration_pts
    GENE_COLUMN = args.gene_column
    BATCH_SIZE = args.batch_size
    HESS_REG_1 = args.hess_reg_1
    HESS_REG_2 = args.hess_reg_2

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEVICE == torch.device("cuda:0"):
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    ZERO = torch.tensor(0., device=DEVICE)
    EPS = torch.tensor(1e-15, device=DEVICE)

    ### load likelihoods and training data ###
    y = pickle.load(open(args.response, 'rb'))
    feature_table = pd.read_csv(args.features, sep='\t', index_col="ensg")
    all_genes = list(set(feature_table.index))
    feature_table = feature_table.loc[all_genes]
    y = format_lh(y, feature_table, args)

    ### train ###
    train_idx = ~feature_table["chrom"].isin(["chr1", "chr3", "chr5", "chr2", "chr4", "chr6"])
    val_idx = feature_table["chrom"].isin(["chr2", "chr4", "chr6"])

    feature_table = feature_table.drop(["hgnc", "chrom"], axis=1)
    X = feature_table.to_numpy()

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
                device=DEVICE,
                tree_method="hist",
                **xgb_params
            )

        print("X.shape", X.shape)
        print("X_train.shape", X_train.shape)
        print("X_val.shape", X_val.shape)

        if args.early_stopping_iter>0:
            model = NGBRegressor(n_estimators=args.total_iterations, Dist=Prior, Base=learner, Score=PriorScore,
                                verbose_eval=1, learning_rate=args.lr, natural_gradient=False
                                ).fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=args.early_stopping_iter)
        else:
            print("no validation set")
            model = NGBRegressor(n_estimators=args.total_iterations, Dist=Prior, Base=learner, Score=PriorScore,
                                verbose_eval=1, learning_rate=args.lr, natural_gradient=False
                                ).fit(X, y)

        f = open(args.out_prefix + ".model", 'wb')
        pickle.dump(model, f)
        f.close()

    if model.best_val_loss_itr is not None:
        output_prior_posterior(model, feature_table, y, model.best_val_loss_itr)
    else:
        output_prior_posterior(model, feature_table, y)


