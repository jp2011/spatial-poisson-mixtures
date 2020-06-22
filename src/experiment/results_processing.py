import multiprocessing
import os
import pickle
import re
import sys
import traceback
import time
from multiprocessing.spawn import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.special import logsumexp
from scipy.stats import poisson
from numba import jit

from src.experiment.model_management import build_lgcp_uid, build_block_mixture_flat_uid, \
    build_block_mixture_gp_softmax_uid
from src.models.block_mixture_flat import FlatRegionMixture
from src.models.block_mixture_gp_softmax import BlockMixtureGpSoftmaxAllocation
from src.models.lgcp_matern import LgcpMatern


sys.path.append(str(Path.home() / "london-crime-mixtures"))

################################################################################
# GENERAL FUNCTIONS
################################################################################
def get_model_description(uid):
    """
    Retrieve the model context

    Every model has context associated with it. The context stores grid information, covariates, membership of grid
    cells to administrative units and various contants that are sueful throughout computation.
    :param uid: a unique identifier for the model
    :return: (:class:`GridContextGeo`: context, int K)
    """

    filepath = Path(os.getcwd()) / "models" / f"context--{uid}.pickle"

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")

    with open(filepath, 'rb') as ctx_file:
        ctx_map = pickle.load(ctx_file)
        context = ctx_map['context']

        all_data_gdf = context.full_dataset
        centroids_df = pd.DataFrame({'x': all_data_gdf.centroid.x, 'y': all_data_gdf.centroid.y})
        centroids_ordered = centroids_df.sort_values(by=['x', 'y'], ascending=True)
        context.full_dataset = all_data_gdf.loc[centroids_ordered.index]

        if 'K' in ctx_map:
            K = ctx_map['K']
            return context, K
        else:
            return context


def get_latest_snapshot_iteration(uid):
    """
    Our code makes regular snapshots. This function finds the latest snapshot for the given model UID.
    :param uid: unique ID for a model
    :return: the iteration number at which the latest snapshot was taken.
    """

    all_fnames = os.listdir(Path(os.getcwd()) / "models" / "snapshots")

    regex_match_string = r'(log-alpha|logalpha|hmc|F|logtheta|Z|alpha|beta)-samples--(.*)--([0-9]*)\.npy'
    matched_fnames = [re.match(regex_match_string, fname) for fname in all_fnames]
    matched_fnames = [x for x in matched_fnames if x is not None]

    decomposed_fnames = [(match.group(1), match.group(2), match.group(3)) for match in matched_fnames]

    max_samples = 0
    for _, _uid, _n in decomposed_fnames:
        if _uid == uid and int(_n) > max_samples:
            max_samples = int(_n)
    return max_samples


@jit(nopython=True)
def compute_joint_kernel_score(counts_predicted_samples, counts_observed, S, alpha=1, beta=1):
    """Computes the energy score as given by Gneiting and Raftery (2007) paper.
 
    Gneiting, Tilmann, and Adrian E Raftery. 2007. ‘Strictly Proper
    Scoring Rules, Prediction, and Estimation’. Journal of the
    American Statistical Association 102 (477):
    359–78. https://doi.org/10.1198/016214506000001437.


    :param counts_predicted_samples: S x N np.ndarray with the predicted counts.
    :param counts_observed: N vector np.ndarray with the observed counts.
    :param alpha: Norm order to use.
    :param beta: Power to which the norm is taken.
    :returns: Energy score
    :rtype: float

    """
    individual_error = 0
    all_permutations_error = 0
    for i in range(S):
        individual_error += np.linalg.norm(counts_predicted_samples[i, :] - counts_observed, ord=alpha) ** beta
        for j in range(S):
            prediction_diff = counts_predicted_samples[i, :] - counts_predicted_samples[j, :]
            all_permutations_error += np.linalg.norm(prediction_diff, ord=alpha) ** beta
            
    return (1 / S) * individual_error - 0.5 * (1 / (S ** 2)) * all_permutations_error


def compute_t_test(sample_1, sample_2):
    """Compute two-sample unpaired t-test between two samples

    :param sample_1: a vector containing samples 1
    :param sample_2: a vector containing samples 2
    :returns: t-statistic and the p-value for the test
    :rtype: (float, float)

    """
    t, p = ttest_ind(sample_1, sample_2, equal_var=False)
    return t, p


def compute_pai_pei(counts_predicted_samples, counts_observed, one_cell_area):
    """Compute PAI and PEI scores

    Compute PAI / PEI scores based on [Chainey_et_al_2008]_

    :param counts_predicted_samples: array of shape (S, N) with S samples of predicted counts for all N cells in the
        study area
    :param counts_observed: array of shape (N, 1) with observed coutns for all N cells in the study area
    :param one_cell_area: area of a single cell in the study area
    :returns: PAI, PEI score for the range of values 1 to N.
    :rtype: (1-D array of size N, 1-D array of size N)

    .. [Chainey_et_al_2008] Chainey, Spencer, Lisa Tompson, and Sebastian Uhlig. 2008. ‘The Utility of Hotspot Mapping
        for Predicting Spatial Patterns of Crime’. Security Journal 21 (1–2): 4–28. https://doi.org/10.1057/palgrave.sj.8350066.
    """

    (S, N) = counts_predicted_samples.shape

    cum_sum_area = one_cell_area * (np.arange(N) + 1)
    total_area = cum_sum_area[-1]

    desc_sorted_observed_counts = np.sort(counts_observed)[::-1]
    total_realised = np.sum(counts_observed)

    counts_observed_tiled = np.tile(counts_observed.reshape(1, -1), (S, 1))
    cum_sum_realised_tiled = np.tile(np.cumsum(desc_sorted_observed_counts).reshape(1, -1), (S, 1))

    # take indices of heavy hitters
    predicted_hotspot_indices = np.argsort(counts_predicted_samples, axis=1)[:, ::-1]

    hotspots_predicted = np.zeros((S, N))
    for s in range(S):
        hotspots_predicted[s, :] = counts_observed_tiled[s, predicted_hotspot_indices[s, :]]
    cum_sum_hotspots = np.cumsum(hotspots_predicted, axis=1)

    pai_all = np.divide(cum_sum_hotspots / total_realised, (cum_sum_area / total_area)[np.newaxis, :])
    pei_all = np.divide(cum_sum_hotspots, cum_sum_realised_tiled)
    return (pai_all, pei_all)


def compute_rmses_for_samples(counts_predicted_samples, counts_observed):
    """Computes the RMSEs for all the samples and return as a vector.

    :param counts_predicted_samples: S x N np.ndarray of predicted counts
    :param counts_observed: np.ndarray vector of size N of observed counts
    :returns: np.ndarray of size S with RMSE values for each sample
    :rtype: np.ndarray
    """

    differences = counts_predicted_samples - counts_observed[np.newaxis, :]
    rmse_all = np.sqrt(np.mean(np.square(differences), axis=1))
    return rmse_all


def compute_avg_loglik(counts_predicted_samples, counts_observed):
    S, N = counts_predicted_samples.shape

    log_pmfs = np.zeros(shape=(S, N))
    for i in range(S):
        log_pmfs[i, :] = poisson.logpmf(counts_observed, counts_predicted_samples[i, :])

    return np.mean(log_pmfs)


def compute_waic(counts_predicted_samples, counts_observed):
    """Computes Watanabe-Akaike information criterion. See [1] for the definition.

    [1] Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson,
    Aki Vehtari, and Donald B. Rubin. 2013. Bayesian Data
    Analysis. Chapman and Hall/CRC. https://doi.org/10.1201/b16018.


    :param counts_predicted_samples: S x N np.ndarray of predicted counts
    :param counts_observed: np.ndarray vector of size N of observed counts
    :returns: Watanabe-Akaike information criterion
    :rtype: float
    """

    S, N = counts_predicted_samples.shape

    log_pmfs = np.zeros(shape=(S, N))
    for i in range(S):
        log_pmfs[i, :] = poisson.logpmf(counts_observed, counts_predicted_samples[i, :])

    # log mean of pmfs computed using pmfs in the log domain
    logged_means = - np.log(S) + logsumexp(log_pmfs, axis=0)

    lppd = np.sum(logged_means)

    p_waic = np.sum(np.var(log_pmfs, axis=0, ddof=1))

    waic = -2 * (lppd - p_waic)
    return waic


def compute_mixture_cov_imp_measure(context,
                                    Z_matrix_samples,
                                    beta_matrix_samples):
    """ Compute covariate importance measure for each covariate and for each mixtue component.

    Please  [Povala_et_al_2020]_ for the explanation.

    :param context: model context
    :param Z_matrix_samples: an array of shape (S, N, K) which contains S samples of (N, K) mixture allocation matrix
        where N is the number of cells and K is the number of mixtures.
    :param beta_matrix_samples: an array of shape (S, J, K) which contains S samples of (J, K) matrix of J regression
        coefficients for each of the K mixture components.
    :returns: IMP measure as described in the reference.
    :rtype:

    .. [Povala_et_al_2020] Povala, Jan, Seppo Virtanen, and Mark Girolami. 2020. ‘Burglary in London: Insights from
        Statistical Heterogeneous Spatial Point Processes’. Journal of the Royal Statistical Society. Series C (Applied
        Statistics) forthcoming. https://arxiv.org/abs/1910.05212v1.
    """
    S, N, K = Z_matrix_samples.shape
    _, J, _ = beta_matrix_samples.shape
    observed_counts = context.counts
    covariates_matrix = context.covariates

    contributions_array = np.zeros(shape=(S, K, J))
    cov_imp_sign_array = np.zeros(shape=(S, K, J))

    # Variance decomposition
    for s in range(S):
        for k in range(K):
            beta_k = beta_matrix_samples[s, :, k]
            active_mask = Z_matrix_samples[s, :, k]
            covariates_active = covariates_matrix[active_mask, :]
            for j in range(J):
                restricted_beta_k = np.copy(beta_k)
                restricted_beta_k[j] = 0.0

                saturated_mean = np.exp(np.dot(covariates_active, beta_k))
                restricted_mean = np.exp(np.dot(covariates_active, restricted_beta_k))

                saturated_ss = np.sum(np.square(saturated_mean - observed_counts[active_mask]))
                restricted_ss = np.sum(np.square(restricted_mean - observed_counts[active_mask]))

                contributions_array[s, k, j] = 1 - (saturated_ss / restricted_ss)

            # cov_imp sign
            cov_imp_sign_array[s, k, :] = np.mean(covariates_active, axis=0)

    cov_imp_mean = pd.DataFrame(data=np.mean(contributions_array, axis=0).T,
                                index=context.covariates_names,
                                columns=list(np.arange(K) + 1))
    cov_imp_sd = pd.DataFrame(data=np.std(contributions_array, axis=0).T,
                              index=context.covariates_names,
                              columns=list(np.arange(K) + 1))
    cov_imp_signs = pd.DataFrame(data=np.mean(cov_imp_sign_array, axis=0).T,
                                 index=context.covariates_names,
                                 columns=list(np.arange(K) + 1))

    return (cov_imp_mean, cov_imp_sd, cov_imp_signs)


def compute_lgcp_cov_imp_measure(context, beta_samples, f_samples):
    """ Compute covariate importance measure for each covariate, as well as the non-parametric 'f' component of the
        log-Gaussian Cox process model as described in [Povala_et_al_2020]_.

    :param context: model context
    :param Z_matrix_samples: an array of shape (S, N, K) which contains S samples of (N, K) mixture allocation matrix
        where N is the number of cells and K is the number of mixtures.
    :param beta_matrix_samples: an array of shape (S, J, K) which contains S samples of (J, K) matrix of J regression
        coefficients for each of the K mixture components.
    :returns: IMP measure as described in the reference.
    :rtype:
    """

    S, N = f_samples.shape
    _, J = beta_samples.shape

    covariates = context.covariates

    contributions_array = np.zeros(shape=(S, context.J + 1))
    for s in range(S):
        beta = beta_samples[s, :]

        saturated_mean = np.exp(f_samples[s, :] + np.dot(covariates, beta))
        saturated_ss = np.sum(np.square(saturated_mean - context.counts))

        for j in range(J):
            restricted_beta = np.copy(beta)
            restricted_beta[j] = 0.0

            restricted_mean = np.exp(f_samples[s, :] + np.dot(covariates, restricted_beta))
            restricted_counts_predicted = np.random.poisson(lam=restricted_mean, size=N)
            restricted_ss = np.sum(np.square(restricted_counts_predicted - context.counts))

            contributions_array[s, j] = 1 - (saturated_ss / restricted_ss)

        # compute the measure of usefulness for the field too
        no_gp_mean = np.exp(np.dot(covariates, beta))
        no_gp_ss = np.sum(np.square(no_gp_mean - context.counts))
        contributions_array[s, context.J] = 1 - (saturated_ss / no_gp_ss)

    signs_array = np.zeros(J+1)  # sign of f is not defined
    signs_array[:J] = np.sign(np.mean(beta_samples, axis=0))

    # Variance contributions
    cov_imp_mean = pd.DataFrame(data=np.mean(contributions_array, axis=0).T,
                                index=context.covariates_names + ['field'],
                                columns=['1'])
    cov_imp_std = pd.DataFrame(data=np.std(contributions_array, axis=0).T,
                               index=context.covariates_names + ['field'],
                               columns=['1'])
    cov_imp_signs = pd.DataFrame(data=signs_array,
                                 index=context.covariates_names + ['field'],
                                 columns=['1'])
    return (cov_imp_mean, cov_imp_std, cov_imp_signs)


def compute_mixture_predicted_counts_all_samples(Z_matrix_samples,
                                                 beta_matrix_samples,
                                                 covariates_matrix):
    """Compute the expected counts for the mixture model for the provided samples.

    :param Z_matrix_samples: S x N x K np.ndarray
    :param beta_matrix_samples: S x J x K np.ndarray
    :param covariates_matrix: N x J np.nd.array
    :returns: S x N matrix of predicted counts for all locations given the samples of
              allocation matrix (Z_matrix_samples), samples of beta matrix (beta_matrix_samples),
              and the covariates matrix (covariates_matrix
    :rtype: np.ndarray
    """

    # Aligning the axes is a mind-bending operation
    temp = np.tensordot(covariates_matrix, beta_matrix_samples, (1, 1))
    temp = np.swapaxes(temp, 0, 1)
    return np.exp(np.sum(np.multiply(Z_matrix_samples, temp), axis=2))


def compute_lgcp_predicted_counts_all_samples(beta_matrix_samples, f_samples, covariates_matrix):
    """Compute the expected counts for the LGCP model for the provided samples.

    :param beta_matrix_samples: 
    :param f_samples: 
    :param covariates_matrix: 
    :returns: 
    :rtype:
    """

    return np.exp(np.dot(beta_matrix_samples, covariates_matrix.T) + f_samples)


def process_mixture_based_model(args):
    """ A gather-type method to compute all the summary statistics of interest for a given model"""
    model_class, uid, S = args
    print(f"PROCESSING: {uid}")
    start_time = time.time()
    try:
        context, K = get_model_description(uid)
        model = model_class(uid=uid, grid_context=context, K=K)

        iteration_no = get_latest_snapshot_iteration(uid)
        samples_tuple = model.load_samples_snapshot(iteration_no)
        
        beta_flat_samples = samples_tuple[0][-S:, :]
        Z_flat_samples = samples_tuple[1][-S:, :]
        
        S = min(beta_flat_samples.shape[0], Z_flat_samples.shape[0])
        print(f"Number of samples: {S}, UID: {uid}")
        
        beta_matrix_samples = beta_flat_samples.reshape((S, context.J, model.K), order='F')
        Z_matrix_samples = np.zeros((S, context.N, K), dtype=bool)
        Z_matrix_samples[np.repeat(range(S), context.N),
                         np.tile(range(context.N), S),
                         Z_flat_samples.flatten(order='C')] = 1

        counts_predicted_samples = compute_mixture_predicted_counts_all_samples(Z_matrix_samples,
                                                                                beta_matrix_samples,
                                                                                context.covariates)

        rmse_all_in = compute_rmses_for_samples(counts_predicted_samples, context.counts)
        rmse_all_out = compute_rmses_for_samples(counts_predicted_samples, context.test_counts)

        waic_in = compute_waic(counts_predicted_samples, context.counts)
        waic_out = compute_waic(counts_predicted_samples, context.test_counts)

        kernel_score = compute_joint_kernel_score(counts_predicted_samples, context.test_counts, S, alpha=2, beta=1)

        avg_loglik_in = compute_avg_loglik(counts_predicted_samples, context.counts)
        avg_loglik_out = compute_avg_loglik(counts_predicted_samples, context.test_counts)

        pai_all, pei_all = compute_pai_pei(counts_predicted_samples, context.test_counts, context.resolution ** 2)

        cov_imp_mean, cov_imp_std, cov_imp_signs = compute_mixture_cov_imp_measure(context,
                                                                                   Z_matrix_samples,
                                                                                   beta_matrix_samples)
        finish_time = time.time()
        print(f"[FINISH] Time: {finish_time - start_time}: {uid}")
        return {
            'RMSE_IN': rmse_all_in,
            'RMSE_OUT': rmse_all_out,
            'WAIC_IN': waic_in,
            'WAIC_OUT': waic_out,
            'AVG_LOGLIK_IN': avg_loglik_in,
            'AVG_LOGLIK_OUT': avg_loglik_out,
            'KERNEL_SCORE': kernel_score,
            'PAI': pai_all,
            'PEI': pei_all,
            'COV_IMP_MEAN': cov_imp_mean,
            'COV_IMP_SD': cov_imp_std,
            'COV_IMP_SIGN': cov_imp_signs
        }
    except FileNotFoundError:
        print(f"No result for: {uid}")
        return {}
    except Exception as exc:
        print(f"Processing error occured.\n{exc}\nUID: {uid}\nBut keep going")
        print(traceback.format_exc())
        return {}


def process_lgcp_based_model(args):
    """ A gather-type method to compute all the summary statistics of interest for a given model"""
    model_class, uid, S = args
    
    print(f"PROCESSING: {uid}")
    start_time = time.time()
    
    try:

        context = get_model_description(uid)
        model = model_class(uid=uid, grid_context=context)
        
        iteration_no = get_latest_snapshot_iteration(uid)
        samples = model.load_samples_snapshot(iteration_no)
        S = min(S, samples.shape[0])

        beta_samples = samples[-S:, model.NN:(model.NN + context.J)]
        f_samples = samples[-S:, :model.NN][:, context.mask]

        counts_predicted_samples = compute_lgcp_predicted_counts_all_samples(beta_samples,
                                                                             f_samples,
                                                                             context.covariates)

        rmse_all_in = compute_rmses_for_samples(counts_predicted_samples, context.counts)
        rmse_all_out = compute_rmses_for_samples(counts_predicted_samples, context.test_counts)

        waic_in = compute_waic(counts_predicted_samples, context.counts)
        waic_out = compute_waic(counts_predicted_samples, context.test_counts)

        kernel_score = compute_joint_kernel_score(counts_predicted_samples, context.test_counts, S, alpha=2, beta=1)

        avg_loglik_in = compute_avg_loglik(counts_predicted_samples, context.counts)
        avg_loglik_out = compute_avg_loglik(counts_predicted_samples, context.test_counts)

        pai_all, pei_all = compute_pai_pei(counts_predicted_samples, context.test_counts, context.resolution**2)

        cov_imp_mean, cov_imp_std, cov_imp_signs = compute_lgcp_cov_imp_measure(context, beta_samples, f_samples)
        finish_time = time.time()
        print(f"[FINISH] Time: {finish_time - start_time}: {uid}")
        return {
            'RMSE_IN': rmse_all_in,
            'RMSE_OUT': rmse_all_out,
            'WAIC_IN': waic_in,
            'WAIC_OUT': waic_out,
            'AVG_LOGLIK_IN': avg_loglik_in,
            'AVG_LOGLIK_OUT': avg_loglik_out,
            'KERNEL_SCORE': kernel_score,
            'PAI': pai_all,
            'PEI': pei_all,
            'COV_IMP_MEAN': cov_imp_mean,
            'COV_IMP_SD': cov_imp_std,
            'COV_IMP_SIGN': cov_imp_signs
        }
    except FileNotFoundError:
        print(f"No result for: {uid}")
        return {}
    except Exception as exc:
        print(f"Processing error occured.\n{exc}\nUID: {uid}\nBut keep going")
        print(traceback.format_exc())
        return {}



################################################################################
# LGCP MATERN
################################################################################
def process_lgcp(*, uid_prefix=None,
                 chain_no=None,
                 set_c_types=None,
                 set_resolutions=None,
                 set_model_names=None,
                 set_t_periods=None,
                 output_name='test_name',
                 S=5_000,
                 pool_size=2,
                 max_paipei_n=10_000):
    """ A high-level method that goes through all possible combinations of the contexts given by the parameters and
        computes the relevant summary statistics for the LGCP experiments"""
    ### Results dataframe that can be easily queried and saved to file.
    results_idx = pd.MultiIndex.from_product([set_c_types, set_resolutions, set_model_names, set_t_periods],
                                             names=['crime_type', 'resolution', 'model', 'time_period'])
    results_df = pd.DataFrame(None, index=results_idx, columns=['RMSE_IN', 'RMSE_OUT', 'WAIC_IN', 'WAIC_OUT', 
                                                                'AVG_LOGLIK_IN', 'AVG_LOGLIK_OUT', 'KERNEL_SCORE'])


    ### PAI/PEI is a bit more complicated
    pai_pei_df_index = pd.MultiIndex.from_product([set_c_types, set_resolutions, set_model_names, set_t_periods, np.arange(max_paipei_n)+1],
                                                  names=['crime_type', 'resolution', 'model', 'time_period', 'n'])
    pai_pei_all_df = pd.DataFrame(index=pai_pei_df_index, columns=[u'PAI', u'PEI', u'PAI_SD', u'PEI_SD'])

    # Covariate Importance Measure objects
    covimp_mean_all, covimp_sd_all, covimp_sign_all = {}, {}, {}
    
    list_of_params_func = []
    for (crime_type, resolution, model_name, time_period) in results_idx:
        uid = build_lgcp_uid(prefix=uid_prefix,
                             chain_no=chain_no,
                             c_type=crime_type,
                             t_period=time_period,
                             model_spec=model_name,
                             resolution=resolution)
        list_of_params_func.append((LgcpMatern, uid, S))
        
    with multiprocessing.Pool(processes=pool_size) as pool:
        results_list = pool.map(process_lgcp_based_model, list_of_params_func)
        pool.close()
        
    # go through the results and collect it nicely
    for idx, (crime_type, resolution, model_name, time_period) in enumerate(results_idx):
        results = results_list[idx]
        uid = build_lgcp_uid(prefix=uid_prefix,
                             chain_no=chain_no,
                             c_type=crime_type,
                             t_period=time_period,
                             model_spec=model_name,
                             resolution=resolution)
        
        print(f"Collection {uid}")
        
        if results == {}:
            continue
            
        covimp_mean_all[uid] = results['COV_IMP_MEAN']
        covimp_sd_all[uid] = results['COV_IMP_SD']
        covimp_sign_all[uid] = results['COV_IMP_SIGN']
       
        results_df.loc[crime_type, resolution, model_name, time_period]['RMSE_IN'] = np.mean(results['RMSE_IN'])
        results_df.loc[crime_type, resolution, model_name, time_period]['RMSE_OUT'] = np.mean(results['RMSE_OUT'])
        results_df.loc[crime_type, resolution, model_name, time_period]['WAIC_IN'] = results['WAIC_IN']
        results_df.loc[crime_type, resolution, model_name, time_period]['WAIC_OUT'] = results['WAIC_OUT']
        results_df.loc[crime_type, resolution, model_name, time_period]['AVG_LOGLIK_IN'] = results['AVG_LOGLIK_IN']
        results_df.loc[crime_type, resolution, model_name, time_period]['AVG_LOGLIK_OUT'] = results['AVG_LOGLIK_OUT']
        results_df.loc[crime_type, resolution, model_name, time_period]['KERNEL_SCORE'] = results['KERNEL_SCORE']
        
        pai = np.mean(results['PAI'][:, :max_paipei_n], axis=0)
        pai_sd = np.std(results['PAI'][:, :max_paipei_n], axis=0)
        pei = np.mean(results['PEI'][:, :max_paipei_n], axis=0)
        pei_sd = np.std(results['PEI'][:, :max_paipei_n], axis=0)
        
        n_labels = np.arange(len(pai)) + 1  # pai is a pandas Series
        pai_pei_all_df.loc[crime_type, resolution, model_name, time_period, n_labels] = np.vstack((pai, pei, pai_sd, pei_sd)).T
    
    results_path = Path(os.getcwd()) / 'data' / 'results'    
    results_df.to_csv(results_path / f'lgcp_model_results_{output_name}.csv', header=True)
    pai_pei_all_df.to_csv(results_path / f'lgcp_pai_pei_df_{output_name}.csv', header=True)
    pickle.dump(covimp_mean_all, open(results_path / f"lgcp_cov_imp_mean_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sign_all, open(results_path / f"lgcp_cov_imp_sign_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sd_all, open(results_path / f"lgcp_cov_imp_sd_{output_name}.pickle", "wb"))
    return
    

################################################################################
# BLOCK MIXTURE FLAT
################################################################################
def process_block_mixtures(*, uid_prefix=None,
                           chain_no=None,
                           set_c_types=None,
                           set_resolutions=None,
                           set_model_names=None,
                           set_t_periods=None,
                           set_block_schemes=None,
                           set_Ks=None,
                           output_name='test_name',
                           S=5_000,
                           pool_size=2,
                           max_paipei_n=10_000):
    """ A high-level method that goes through all possible combinations of the contexts given by the parameters and
            computes the relevant summary statistics for the mixture experiments"""
    ### Results dataframe that can be easily queried and saved to file.
    results_idx = pd.MultiIndex.from_product([set_c_types, set_resolutions, set_model_names, set_t_periods, set_block_schemes, set_Ks],
                                             names=['crime_type', 'resolution', 'model', 'time_period', 'block_scheme', 'k'])
    
    results_df = pd.DataFrame(None, index=results_idx,
                              columns=['RMSE_IN', 'RMSE_OUT', 'WAIC_IN', 'WAIC_OUT', 'AVG_LOGLIK_IN', 'AVG_LOGLIK_OUT', 'KERNEL_SCORE'])

    pai_pei_df_index = pd.MultiIndex.from_product([set_c_types, set_resolutions, set_model_names, set_t_periods,
                                                   set_block_schemes, set_Ks, np.arange(max_paipei_n)+1],
                                                  names=['crime_type', 'resolution', 'model', 'time_period', 'block_scheme', 'k', 'n'])
    pai_pei_all_df = pd.DataFrame(index=pai_pei_df_index, columns=[u'PAI', u'PEI', u'PAI_SD', u'PEI_SD'])

    # Covariate Importance Measure objects
    covimp_mean_all, covimp_sd_all, covimp_sign_all = {}, {}, {}

    list_of_func_params = []
    for (c_type, resolution, model_name, t_period, block_scheme, K) in results_idx:
        uid = build_block_mixture_flat_uid(prefix=uid_prefix, chain_no=chain_no, block_scheme=block_scheme,
                                           c_type=c_type, t_period=t_period, model_spec=model_name, resolution=resolution, K=K)
        list_of_func_params.append([FlatRegionMixture, uid, S])
    with multiprocessing.Pool(processes=pool_size) as pool:
        results_list = pool.map(process_mixture_based_model, list_of_func_params)
        pool.terminate()

    # go through the results and collect it nicely
    for idx, (c_type, resolution, model_name, t_period, block_scheme, K) in enumerate(results_idx):
        results = results_list[idx]

        uid = build_block_mixture_flat_uid(prefix=uid_prefix, chain_no=chain_no, block_scheme=block_scheme,
                                           c_type=c_type, t_period=t_period, model_spec=model_name,
                                           resolution=resolution, K=K)
        print(f"Collection {uid}")

        if results == {}:
            continue

        covimp_mean_all[uid] = results['COV_IMP_MEAN']
        covimp_sd_all[uid] = results['COV_IMP_SD']
        covimp_sign_all[uid] = results['COV_IMP_SIGN']
       
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['RMSE_IN'] = np.mean(results['RMSE_IN'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['RMSE_OUT'] = np.mean(results['RMSE_OUT'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['WAIC_IN'] = results['WAIC_IN']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['WAIC_OUT'] = results['WAIC_OUT']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['AVG_LOGLIK_IN'] = results['AVG_LOGLIK_IN']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['AVG_LOGLIK_OUT'] = results['AVG_LOGLIK_OUT']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K]['KERNEL_SCORE'] = results['KERNEL_SCORE']
        
        pai = np.mean(results['PAI'][:, :max_paipei_n], axis=0)
        pai_sd = np.std(results['PAI'][:, :max_paipei_n], axis=0)
        pei = np.mean(results['PEI'][:, :max_paipei_n], axis=0)
        pei_sd = np.std(results['PEI'][:, :max_paipei_n], axis=0)
        
        n_labels = np.arange(len(pai)) + 1  # pai is a pandas Series
        pai_pei_all_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, n_labels] = np.vstack((pai, pei, pai_sd, pei_sd)).T

    results_path = Path(os.getcwd()) / 'data' / 'results'    
    results_df.to_csv(results_path / f'block_mixtures_model_results_{output_name}.csv', header=True)
    pai_pei_all_df.to_csv(results_path / f'block_mixtures_pai_pei_df_{output_name}.csv', header=True)
    pickle.dump(covimp_mean_all, open(results_path / f"block_mixtures_cov_imp_mean_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sign_all, open(results_path / f"block_mixtures_cov_imp_sign_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sd_all, open(results_path / f"block_mixtures_cov_imp_sd_{output_name}.pickle", "wb"))
    return


################################################################################
# BLOCK MIXTURE GP SOFTMAX
################################################################################
def process_block_gps_mixtures(*, uid_prefix=None,
                               chain_no=None,
                               set_c_types=None,
                               set_resolutions=None,
                               set_model_names=None,
                               set_t_periods=None,
                               set_block_schemes=None,
                               set_Ks=None,
                               set_lengthscales=None,
                               output_name='test_name',
                               S=5_000,
                               pool_size=2,
                               max_paipei_n=10_000):
    ### Results dataframe that can be easily queried and saved to file.
    results_idx = pd.MultiIndex.from_product(
        [set_c_types, set_resolutions, set_model_names, set_t_periods, set_block_schemes, set_Ks, set_lengthscales],
        names=['crime_type', 'resolution', 'model', 'time_period', 'block_scheme', 'k', 'lengthscale'])

    results_df = pd.DataFrame(None, index=results_idx,
                              columns=['RMSE_IN', 'RMSE_OUT', 'RMSE_IN_SD', 'RMSE_OUT_SD',
                                       'WAIC_IN', 'WAIC_OUT',
                                       'AVG_LOGLIK_IN', 'AVG_LOGLIK_OUT', 'AVG_LOGLIK_IN_SD', 'AVG_LOGLIK_OUT_SD',
                                       'KERNEL_SCORE'])

    pai_pei_df_index = pd.MultiIndex.from_product([set_c_types, set_resolutions, set_model_names, set_t_periods,
                                                   set_block_schemes, set_Ks, set_lengthscales,
                                                   np.arange(max_paipei_n) + 1],
                                                  names=['crime_type', 'resolution', 'model', 'time_period',
                                                         'block_scheme', 'k', 'lengthscale', 'n'])
    pai_pei_all_df = pd.DataFrame(index=pai_pei_df_index, columns=[u'PAI', u'PEI', u'PAI_SD', u'PEI_SD'])

    # Covariate Importance Measure objects
    covimp_mean_all, covimp_sd_all, covimp_sign_all = {}, {}, {}

    list_of_func_params = []
    for (c_type, resolution, model_name, t_period, block_scheme, K, lengthscale) in results_idx:
        uid = build_block_mixture_gp_softmax_uid(prefix=uid_prefix, chain_no=chain_no, block_scheme=block_scheme,
                                                 c_type=c_type, t_period=t_period, model_spec=model_name,
                                                 resolution=resolution, K=K, lengthscale=lengthscale)
        list_of_func_params.append([BlockMixtureGpSoftmaxAllocation, uid, S])
    with multiprocessing.Pool(processes=pool_size) as pool:
        results_list = pool.map(process_mixture_based_model, list_of_func_params)
        pool.terminate()

    pai_pei_all_df.sort_index(inplace=True)
    results_df.sort_index(inplace=True)

    # go through the results and collect it nicely
    for idx, (c_type, resolution, model_name, t_period, block_scheme, K, lengthscale) in enumerate(results_idx):
        results = results_list[idx]

        uid = build_block_mixture_gp_softmax_uid(prefix=uid_prefix, chain_no=chain_no, block_scheme=block_scheme,
                                                 c_type=c_type, t_period=t_period, model_spec=model_name,
                                                 resolution=resolution, K=K, lengthscale=lengthscale)
        print(f"Collection {uid}")

        if results == {}:
            continue

        covimp_mean_all[uid] = results['COV_IMP_MEAN']
        covimp_sd_all[uid] = results['COV_IMP_SD']
        covimp_sign_all[uid] = results['COV_IMP_SIGN']

        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['RMSE_IN'] = np.mean(
            results['RMSE_IN'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['RMSE_IN_SD'] = np.std(
            results['RMSE_IN'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['RMSE_OUT'] = np.mean(
            results['RMSE_OUT'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['RMSE_OUT_SD'] = np.std(
            results['RMSE_OUT'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['WAIC_IN'] = results[
            'WAIC_IN']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['WAIC_OUT'] = results[
            'WAIC_OUT']
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale][
            'AVG_LOGLIK_IN'] = np.mean(results['AVG_LOGLIK_IN'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale][
            'AVG_LOGLIK_IN_SD'] = np.std(results['AVG_LOGLIK_IN'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale][
            'AVG_LOGLIK_OUT'] = np.mean(results['AVG_LOGLIK_OUT'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale][
            'AVG_LOGLIK_OUT_SD'] = np.std(results['AVG_LOGLIK_OUT'])
        results_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale]['KERNEL_SCORE'] = \
        results['KERNEL_SCORE']

        pai = np.mean(results['PAI'][:, :max_paipei_n], axis=0)
        pai_sd = np.std(results['PAI'][:, :max_paipei_n], axis=0)
        pei = np.mean(results['PEI'][:, :max_paipei_n], axis=0)
        pei_sd = np.std(results['PEI'][:, :max_paipei_n], axis=0)

        for i in range(max_paipei_n):
            pai_pei_all_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale, i + 1]["PAI"] = \
            pai[i]
            pai_pei_all_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale, i + 1][
                "PAI_SD"] = pai_sd[i]
            pai_pei_all_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale, i + 1]["PEI"] = \
            pei[i]
            pai_pei_all_df.loc[c_type, resolution, model_name, t_period, block_scheme, K, lengthscale, i + 1][
                "PEI_SD"] = pei_sd[i]

    results_path = Path(os.getcwd()) / 'data' / 'results'
    results_df.to_csv(results_path / f'block_mixtures_gp_model_results_{output_name}.csv', header=True)
    pai_pei_all_df.to_csv(results_path / f'block_mixtures_gp_pai_pei_df_{output_name}.csv', header=True)
    pickle.dump(covimp_mean_all, open(results_path / f"block_mixtures_gp_cov_imp_mean_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sign_all, open(results_path / f"block_mixtures_gp_cov_imp_sign_{output_name}.pickle", "wb"))
    pickle.dump(covimp_sd_all, open(results_path / f"block_mixtures_gp_cov_imp_sd_{output_name}.pickle", "wb"))
    return


################################################################################
# MAIN FUNCTION FOR TESTING PURPOSES
################################################################################
if __name__ == '__main__':
    freeze_support()

    set_crime_types = ['burglary']
    set_model_names = ['burglary_raw_1']
    set_region_aggs = ["msoa"]
    set_time_periods = ['12015-122015', '12013-122015']
    set_resolutions = ["400"]
    set_urbanisation_cutoffs = ['0_0']

    glm_lgcp_comp_results_tag = 'dummy_test'
    uid_prefix = '2020-02-12-BLOCKMIX'


    process_block_mixtures(uid_prefix=uid_prefix,
                           set_c_types=set_crime_types,
                           set_resolutions=set_resolutions,
                           set_model_names=set_model_names,
                           set_t_periods=set_time_periods,
                           set_block_schemes=set_region_aggs,
                           set_Ks=[2, 3],
                           output_name='test',
                           S=10,
                           pool_size=10)






















































def process_mrf_mixtures(uid_prefix, set_crime_types, set_resolutions, set_model_names, set_time_periods, set_spatial_strengths,
                         set_urbanisation_cutoffs, set_Ks, output_name='test_name', num_samples=100, pool_size=2):
    ### Results dataframe that can be easily queried and saved to file.
    results_idx = pd.MultiIndex.from_product(
        [set_crime_types, set_resolutions, set_model_names, set_time_periods, set_spatial_strengths, set_urbanisation_cutoffs,
         set_Ks],
        names=['crime type', 'resolution', 'model', 'time period', 'spatial strength', 'urbanisation_threshold',
               'k'])

    results_df = pd.DataFrame(None, index=results_idx, columns=['MAE_IN', 'MAE_OUT', 'RMSE_IN', 'RMSE_OUT', 'WAIC_IN',
                                                                'JOINT_KERNEL_SCORE_1_1', 'JOINT_KERNEL_SCORE_2_1',
                                                                'JOINT_KERNEL_SCORE_2_2'])

    # PAI/PEI is a bit more complicated
    pai_pei_df_index = pd.MultiIndex.from_product(
        [set_crime_types, set_resolutions, set_model_names, set_time_periods, set_spatial_strengths, set_urbanisation_cutoffs,
         set_Ks, np.arange(30_000) + 1],
        names=['crime type', 'resolution', 'model', 'time period', 'spatial strength', 'urbanisation_threshold',
               'k', 'n'])
    pai_pei_all_df = pd.DataFrame(index=pai_pei_df_index,
                                  columns=[u'PAI', u'PEI', u'HIT_RATE', u'PAI_SD', u'PEI_SD', u'HIT_RATE_SD'])

    # Variance decomposition object
    variance_contributions_all_models = {}
    variance_contributions_sd_all_models = {}
    covariate_effect_signs_all_models = {}

    list_of_params_func = []
    for (crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold,
         num_mixtures) in results_idx:
        uid = f"{uid_prefix}--{spatial_strength}--{crime_type}--{time_period}--{model_name}--weighted--{resolution}--{num_mixtures}--{urbanisation_threshold}"

        list_of_params_func.append((crime_type, resolution, model_name, time_period, spatial_strength,
                                    urbanisation_threshold, num_mixtures, uid, output_name, num_samples))
    with multiprocessing.Pool(processes=pool_size) as pool:
        results_list = pool.map(None, list_of_params_func)
        pool.terminate()

    # go through the results and collect it nicely
    for idx, (
    crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures, uid,
    output_name, num_samples) in enumerate(list_of_params_func):
        results = results_list[idx]

        print(f"Collection {uid}")

        if results == {}:
            continue

        variance_contributions_all_models[uid] = results['COV_EFFECTS']
        covariate_effect_signs_all_models[uid] = results['COV_EFFECTS_SIGN']
        variance_contributions_sd_all_models[uid] = results['COV_EFFECTS_SD']

        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'MAE_IN'] = results['MAE_IN']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'MAE_OUT'] = results['MAE_OUT']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'RMSE_IN'] = results['RMSE_IN']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'RMSE_OUT'] = results['RMSE_OUT']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'WAIC_IN'] = results['WAIC_IN']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'JOINT_KERNEL_SCORE_1_1'] = results['JOINT_KERNEL_SCORE_1_1']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'JOINT_KERNEL_SCORE_2_1'] = results['JOINT_KERNEL_SCORE_2_1']
        results_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures][
            'JOINT_KERNEL_SCORE_2_2'] = results['JOINT_KERNEL_SCORE_2_2']

        pai = results['PAI']
        pai_sd = results['PAI_SD']
        pei = results['PEI']
        pei_sd = results['PEI_SD']
        hit_rate = results['HIT_RATE']
        hit_rate_sd = results['HIT_RATE_SD']
        n_labels = np.arange(len(pai)) + 1  # pai is a pandas Series
        pai_pei_all_df.loc[
            crime_type, resolution, model_name, time_period, spatial_strength, urbanisation_threshold, num_mixtures, n_labels] = np.vstack(
            (pai, pei, hit_rate, pai_sd, pei_sd, hit_rate_sd)).T

    results_df.to_csv(Path(os.getcwd()) / 'data' / 'results' / f'mrf_mixtures_model_results_{output_name}.csv', header=True)
    pai_pei_all_df.to_csv(Path(os.getcwd()) / 'data' / 'results' / f'mrf_mixtures_pai_pei_df_{output_name}.csv',
                          header=True)
    pickle.dump(variance_contributions_all_models,
                open(Path(os.getcwd()) / 'data' / 'results' / f"mrf_mixtures_variance_contributions_{output_name}.pickle",
                     "wb"))
    pickle.dump(covariate_effect_signs_all_models,
                open(Path(os.getcwd()) / 'data' / 'results' / f"mrf_mixtures_cov_effect_signs_{output_name}.pickle", "wb"))
    pickle.dump(variance_contributions_sd_all_models, open(
        Path(os.getcwd()) / 'data' / 'results' / f"mrf_mixtures_variance_contributions_sd_{output_name}.pickle", "wb"))

    return

    
def process_block_gps_mixtures(Model_class, uid_prefix, set_crime_types, set_resolutions, set_model_names, set_time_periods, set_region_aggs, set_urbanisation_cutoffs, set_Ks, set_lengthscales, output_name='test_name', num_samples=100, pool_size=2):
    ### Results dataframe that can be easily queried and saved to file.
    results_idx = pd.MultiIndex.from_product([set_crime_types, set_resolutions, set_model_names, set_time_periods, set_region_aggs, set_urbanisation_cutoffs, set_Ks, set_lengthscales],
                                     names=['crime type', 'resolution', 'model', 'time period', 'prior weight sharing', 'urbanisation_threshold', 'k', 'lengthscale'])
    
    results_df = pd.DataFrame(None, index=results_idx, columns=['MAE_IN', 'MAE_OUT', 'RMSE_IN', 'RMSE_OUT', 'WAIC_IN',
                                                                'JOINT_KERNEL_SCORE_1_1', 'JOINT_KERNEL_SCORE_2_1', 'JOINT_KERNEL_SCORE_2_2'])

    ### PAI/PEI is a bit more complicated
    pai_pei_df_index = pd.MultiIndex.from_product([set_crime_types, set_resolutions, set_model_names, set_time_periods, set_region_aggs, set_urbanisation_cutoffs, set_Ks, set_lengthscales, np.arange(30_000)+1],
                                                  names=['crime type', 'resolution', 'model', 'time period', 'prior weight sharing', 'urbanisation_threshold', 'k', 'lengthscale', 'n'])
    pai_pei_all_df = pd.DataFrame(index=pai_pei_df_index, columns=[u'PAI', u'PEI', u'HIT_RATE', u'PAI_SD', u'PEI_SD', u'HIT_RATE_SD'])

    # Variance decomposition object
    variance_contributions_all_models = {}
    variance_contributions_sd_all_models = {}
    covariate_effect_signs_all_models = {}
    
    list_of_params_func = []
    for (crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale) in results_idx:
        uid = f"{uid_prefix}--{prior_weight_region}--{crime_type}--{time_period}--{model_name}--weighted--{resolution}--{time_period}--{num_mixtures}--{lengthscale}--{urbanisation_threshold}"
        list_of_params_func.append((Model_class, crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale, uid, output_name, num_samples))
        
    with multiprocessing.Pool(processes=pool_size) as pool:
        results_list = pool.map(None, list_of_params_func)
        pool.terminate()
    
    # go through the results and collect it nicely
    for idx, (_, crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale, uid, output_name, num_samples) in enumerate(list_of_params_func):
        results = results_list[idx]
        
        print(f"Collection {uid}")
        
        if results == {}:
            continue
        
        variance_contributions_all_models[uid] = results['COV_EFFECTS']
        covariate_effect_signs_all_models[uid] = results['COV_EFFECTS_SIGN']
        variance_contributions_sd_all_models[uid] = results['COV_EFFECTS_SD']
        
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['MAE_IN'] = results['MAE_IN']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['MAE_OUT'] = results['MAE_OUT']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['RMSE_IN'] = results['RMSE_IN']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['RMSE_OUT'] = results['RMSE_OUT']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['WAIC_IN'] = results['WAIC_IN']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['JOINT_KERNEL_SCORE_1_1'] = results['JOINT_KERNEL_SCORE_1_1']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['JOINT_KERNEL_SCORE_2_1'] = results['JOINT_KERNEL_SCORE_2_1']
        results_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale]['JOINT_KERNEL_SCORE_2_2'] = results['JOINT_KERNEL_SCORE_2_2']

        pai = results['PAI']
        pai_sd = results['PAI_SD']
        pei = results['PEI']
        pei_sd = results['PEI_SD']
        hit_rate = results['HIT_RATE']
        hit_rate_sd = results['HIT_RATE_SD']
        n_labels = np.arange(len(pai)) + 1 # pai is a pandas Series
        pai_pei_all_df.loc[crime_type, resolution, model_name, time_period, prior_weight_region, urbanisation_threshold, num_mixtures, lengthscale, n_labels] = np.vstack((pai, pei, hit_rate, pai_sd, pei_sd, hit_rate_sd)).T
        
    results_df.to_csv(Path(os.getcwd()) / 'data' / 'results' / f'block_mixtures_gp_model_results_{output_name}.csv', header=True)
    pai_pei_all_df.to_csv(Path(os.getcwd()) / 'data' / 'results' / f'block_mixtures_gp_pai_pei_df_{output_name}.csv', header=True)
    pickle.dump(variance_contributions_all_models, open(Path(os.getcwd()) / 'data' / 'results' / f"block_mixtures_gp_variance_contributions_{output_name}.pickle", "wb"))
    pickle.dump(covariate_effect_signs_all_models, open(Path(os.getcwd()) / 'data' / 'results' / f"block_mixtures_gp_cov_effect_signs_{output_name}.pickle", "wb"))
    pickle.dump(variance_contributions_sd_all_models, open(Path(os.getcwd()) / 'data' / 'results' / f"block_mixtures_gp_variance_contributions_sd_{output_name}.pickle", "wb"))
    
    return
