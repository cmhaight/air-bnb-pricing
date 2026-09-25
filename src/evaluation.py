import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from . import data_processing as dpr
from . import reporting as rep
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

def eval_model_sm(results,*,target,predicted_vals,name="model"):
    adj_r_sqrd = results.rsquared_adj
    mae = mean_absolute_error(target, predicted_vals)
    mape = mean_absolute_percentage_error(target, predicted_vals)
    rootmse = np.sqrt(mean_squared_error(target,predicted_vals))
    results_metric = pd.DataFrame([[name,mae,rootmse,mape,adj_r_sqrd]],
                                  columns=['model_name','mae','rmse','mape','r2_score'])
    return results_metric


def show_vif(df_encoded):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_encoded.columns
    vif_data["vif"] = [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]
    return vif_data


def eval_binned_targets(*,target,predicted_vals,bin_group='predicted',name='model'):

    results_bins = pd.DataFrame({'actual':target,'predicted':predicted_vals})
    price_group = f"price_group"
    results_bins[price_group] = pd.cut(
    results_bins[bin_group],
    bins=[0, 100, 250, 500, 1000, float("inf")],
    labels=["<$100", "$100-$250", "$250-$500", "$500-$1000","$1000+"]
    )

    group_mae = results_bins.groupby(price_group, observed=True).apply(
    lambda group: mean_absolute_error(
        group["actual"],
        group["predicted"]
        )
    )
    group_mape = results_bins.groupby(price_group, observed=True).apply(
    lambda group: mean_absolute_percentage_error(
        group["actual"],
        group["predicted"]
        )
    )
    group_rmse = results_bins.groupby(price_group, observed=True).apply(
    lambda group: root_mean_squared_error(
        group["actual"],
        group["predicted"]
        )
    )
    results_metrics_df = pd.concat([group_mae,group_rmse,group_mape],axis=1,keys = ['mae','rmse','mape']).reset_index()
    results_metrics_df.insert(0,'model_name',name)
    results_metrics_df.insert(1,'bin_type',bin_group)
    return results_metrics_df


def eval_bin_target_raw_err(*,target,predicted_vals,bin_group='predicted',name='model'):
    results_bins = pd.DataFrame({'actual':target,'predicted':predicted_vals})
    price_group = f"price_group"

    results_bins[price_group] = pd.cut(
    results_bins[bin_group],
    bins=[0, 100, 250, 500, 1000, float("inf")],
    labels=["<$100", "$100-$250", "$250-$500", "$500-$1000","$1000+"]
    )
    results_bins["error"] = (
    results_bins["predicted"] - results_bins["actual"]
    )

    results_bin_raw_metrics = results_bins.groupby(price_group)["error"].agg(
    ["mean", "median", "count"]
    ).reset_index()
    results_bin_raw_metrics.insert(0,'model_name',name)
    results_bin_raw_metrics.insert(1,'bin_type',bin_group)
    return results_bin_raw_metrics


def cross_validate_five_split(features_encoded,raw_target,split_group,log_transform=True,name='model'):
    gkf = GroupKFold(n_splits=5)
    mae_scores = []
    rmse_scores = []
    y = raw_target
    X = features_encoded
    groups = split_group

    for train_idx, val_idx in gkf.split(X, y, groups=groups):

        X_tr, X_val = (X.iloc[train_idx], X.iloc[val_idx])
        y_tr, y_val = (y.iloc[train_idx], y.iloc[val_idx])
        if log_transform:
            fold_model = sm.OLS(np.log(y_tr), X_tr).fit()
            log_predictions = fold_model.predict(X_val)
            real_predictions = np.exp(log_predictions)
        else:
            fold_model = sm.OLS(y_tr, X_tr).fit()
            real_predictions = fold_model.predict(X_val)

        fold_mae = mean_absolute_error(y_val, real_predictions)
        fold_rmse = root_mean_squared_error(y_val, real_predictions)

        mae_scores.append(fold_mae)
        rmse_scores.append(fold_rmse)
        cv_df = pd.DataFrame([[name,np.mean(mae_scores),np.std(mae_scores),np.mean(rmse_scores),np.std(rmse_scores)]],
                             columns=['model_name','cv_avg_mae','cv_mae_std_dev','cv_avg_rmse','cv_rmse_std_dev'])
                           
    return cv_df

def cross_val_5_split_sklearn(model,X,y,group,name='model'):
    gkf = GroupKFold(n_splits=5)
    cv_scores_mae = cross_val_score(model,X,y,groups=group, cv=gkf,scoring="neg_mean_absolute_error")
    cv_scores_rmse = cross_val_score(model,X,y,groups=group, cv=gkf,scoring="neg_root_mean_squared_error")
    cv_scores_mape = cross_val_score(model,X,y,groups=group, cv=gkf,scoring="neg_mean_absolute_percentage_error")

    avg_mae = np.mean(cv_scores_mae) * -1
    std_mae = np.std(cv_scores_mae)
    avg_rmse = np.mean(cv_scores_rmse) * -1
    std_rmse = np.std(cv_scores_rmse)
    avg_mape = np.mean(cv_scores_mape) * -1

    cv_df = pd.DataFrame([[name,avg_mae,std_mae,avg_rmse,std_rmse]],
                            columns=['model_name','cv_avg_mae','cv_mae_std_dev','cv_avg_rmse','cv_rmse_std_dev'])
                            
    return cv_df


def pred_interval_statmodel(*,results,X_w_constant):
    smearing_factor = np.mean(np.exp(results.resid))
    prediction_object = results.get_prediction(X_w_constant)
    summary_frame = prediction_object.summary_frame(alpha=0.05)

    pred_df = np.exp(summary_frame)
    pred_df['mean'] = pred_df['mean'] * smearing_factor
    pred_df['ci_difference'] = pred_df['obs_ci_upper'] - pred_df['obs_ci_lower']
    return pred_df


def eval_tree_model(*,target,predicted_vals,name='Model'):
    r2 = r2_score(target, predicted_vals)
    rmse = root_mean_squared_error(target, predicted_vals)
    mae = mean_absolute_error(target, predicted_vals)
    mape= mean_absolute_percentage_error(target,predicted_vals)

    results_metric = pd.DataFrame([[name,mae,rmse,mape,r2]],columns=['model_name','mae','rmse','mape','r2_score'])
                                  
    return results_metric
            

def sm_feature_imp(results):
    t_stats = results.tvalues
    if 'const' in t_stats.index:
        t_stats = t_stats.drop('const')
    abs_t_stats = t_stats.abs()
    top_10_features = abs_t_stats.sort_values(ascending=False).head(10)
    top_10_names = top_10_features.index
    top_10_weights = results.params[top_10_names]
    top_10_feat_df = pd.DataFrame(top_10_weights,columns=['percentage_effect'])
    top_10_feat_df['percentage_effect'] = (np.exp(top_10_feat_df['percentage_effect']) - 1) * 100
    top_10_feat_df = (top_10_feat_df.reindex(top_10_feat_df['percentage_effect'].abs()
                                    .sort_values(ascending=False).index))
    return top_10_feat_df


def hgbr_feat_imp(model,features,target,hyperparams):
    perm_importance = permutation_importance(
    model,features, target,**hyperparams
    )
    raw_importances = perm_importance.importances_mean
    raw_importances_std_devs = perm_importance.importances_std
    top_10_feat_df = (pd.DataFrame({'feature_names':features.columns,'mean_r2_decrease':raw_importances,
                                   'std_r2_decrease':raw_importances_std_devs}))
    top_10_feat_df = top_10_feat_df.sort_values(by='mean_r2_decrease', ascending=False).head(10)
    return top_10_feat_df


def sm_ols_eval_pipe(results,*,features_encoded,split_group,name='model',path):
    logger.info("Evaluating OLS model")
    predicts_orig_scale,target_orig_scale = dpr.log_re_transformation(results)

    cv_scores = cross_validate_five_split(features_encoded,target_orig_scale,split_group,name=name)           
    global_eval = eval_model_sm(results,target=target_orig_scale,predicted_vals=predicts_orig_scale,name=name)           
    bin_eval = eval_binned_targets(target=target_orig_scale,predicted_vals=predicts_orig_scale,name=name)   
    bin_eval_raw = eval_bin_target_raw_err(target=target_orig_scale,predicted_vals=predicts_orig_scale,name=name)
    bin_eval_actual = eval_binned_targets(target=target_orig_scale,predicted_vals=predicts_orig_scale
                                             ,bin_group='actual',name=name)
    bin_eval_raw_actual = eval_bin_target_raw_err(target=target_orig_scale,predicted_vals=predicts_orig_scale
                                                     ,bin_group='actual',name=name)
    top_10_feats = sm_feature_imp(results)
    
    binned_scores_raw = pd.concat([bin_eval_raw,bin_eval_raw_actual],ignore_index=True)
    binned_scores = pd.concat([bin_eval,bin_eval_actual],ignore_index=True)

    rep.save_diag_plots(result=results,path=path,predicts_orig_scale=predicts_orig_scale,
                        target_orig_scale=target_orig_scale,
                        model_name=name,feat_df=top_10_feats)
    
    return global_eval,cv_scores,binned_scores,binned_scores_raw


def histgbr_eval_pipe(*,model,features,target,predicted_vals,split_group,name="model",path,perm_imp_params,
                      logged=False):
    logger.info("Evaluating HGBR model")
    cv_scores = cross_val_5_split_sklearn(model,features,target,split_group,name)
    global_eval = eval_tree_model(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval = eval_binned_targets(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval_raw = eval_bin_target_raw_err(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval_actual = eval_binned_targets(target=target,predicted_vals=predicted_vals,bin_group='actual',name=name)
    bin_eval_raw_actual = eval_bin_target_raw_err(target=target,predicted_vals=predicted_vals,bin_group='actual',
                                                  name=name)
    binned_scores_raw = pd.concat([bin_eval_raw,bin_eval_raw_actual],ignore_index=True)
    binned_scores = pd.concat([bin_eval,bin_eval_actual],ignore_index=True)

    hgbr_top_10_feats = hgbr_feat_imp(model,features,target,perm_imp_params)

    rep.save_diag_plots(path=path,predicts_orig_scale=predicted_vals,target_orig_scale=target,model_name=name,
                        feat_df=hgbr_top_10_feats,logged=logged)
    return global_eval,cv_scores,binned_scores,binned_scores_raw


def test_set_hgbr_eval_pipe(*,model,features,target,predicted_vals,name="model",path,perm_imp_params,logged=False):
    
    logger.info("Evaluating test set with HGBR model")
    global_eval = eval_tree_model(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval = eval_binned_targets(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval_raw = eval_bin_target_raw_err(target=target,predicted_vals=predicted_vals,name=name)
    bin_eval_actual = eval_binned_targets(target=target,predicted_vals=predicted_vals,bin_group='actual',name=name)
    bin_eval_raw_actual = eval_bin_target_raw_err(target=target,predicted_vals=predicted_vals,bin_group='actual',
                                                  name=name)
    binned_scores_raw = pd.concat([bin_eval_raw,bin_eval_raw_actual],ignore_index=True)
    binned_scores = pd.concat([bin_eval,bin_eval_actual],ignore_index=True)

    hgbr_top_10_feats = hgbr_feat_imp(model,features,target,perm_imp_params)

    rep.save_diag_plots(path=path,predicts_orig_scale=predicted_vals,target_orig_scale=target,model_name=name,
                        feat_df=hgbr_top_10_feats,logged=logged)
    return global_eval,binned_scores,binned_scores_raw