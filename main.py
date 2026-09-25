import matplotlib
matplotlib.use('Agg') 
import src.data_processing as dpr
import src.reporting as rep
import src.evaluation as ev
import src.models as mod
import pandas as pd
import yaml
import logging
from pathlib import Path

def main(data_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("AirbnbPipeline")

    with open ("config.yaml","r") as file:
        config = yaml.safe_load(file)
    
    run_root,csv_dir,plot_dir = rep.create_output_folder("config.yaml")
    global_csv_path = csv_dir / config['paths']['output_paths']['csv_paths']['global_path']
    binned_path = csv_dir / config['paths']['output_paths']['csv_paths']['binned_path']
    binned_path_raw = csv_dir / config['paths']['output_paths']['csv_paths']['binned_path_raw']
    cv_path = csv_dir / config['paths']['output_paths']['csv_paths']['cv_path']
    sm_summary_path_m1 = run_root / "statsmodels_ols_summary_m1.txt"
    sm_summary_path_m2 = run_root / "statsmodels_ols_summary_m2.txt"
    perm_hyperparams = config['evals']['permutation_settings']

    raw_zip_path = config['paths']['geo_paths']['raw_zip_path']
    clean_zip_path = Path(raw_zip_path)
    
    logger.info('Starting the data pipeline...')

    df = dpr.csv_to_df(data_path)


    df = dpr.pre_split_col_drop(df,config['col_groups']['col_initial_drop'])
    train_set, test_set = dpr.split_data_with_id_hash(df,0.2,'id')
    
    train_set = dpr.post_split_clean(train_set)

    train_set_clean = (train_set.pipe(dpr.feature_engineering,clean_zip_path)
                .pipe(dpr.drop_cols,config['col_groups']['col_after_ftr_eng_drop']))

    cv_groups = train_set_clean[config['groups']['cv_split_group']]
    
    logger.info("Training and evaluating model 1")
    
    target = config['targets']['original']
    model_1_num_features = config['models']['sm_ols_model_1']['features']['numeric']
    model_1_cat_features = config['models']['sm_ols_model_1']['features']['categorical']
    model_1_features = model_1_num_features + model_1_cat_features

    y_train_m1, X_train_m1 = dpr.prep_data_ols_statsmodels(train_set_clean,features=model_1_features,
                                                           cat_features=model_1_cat_features,
                                                           target=target,log_transform=True)
    
    model_1 = mod.build_stats_models_lin_regr(y=y_train_m1,X=X_train_m1)
    m1_results = model_1.fit()

    m1_global_eval,cv_scores_m1,m1_binned_scores,m1_bin_raw = ev.sm_ols_eval_pipe(m1_results,
                                                                                  features_encoded=X_train_m1,
                                                                                  split_group=cv_groups,
                                                                                  name="ols_model_1",path=plot_dir)

    rep.export_to_csv(m1_global_eval,global_csv_path)
    rep.export_to_csv(m1_binned_scores,binned_path)
    rep.export_to_csv(m1_bin_raw,binned_path_raw)
    rep.export_to_csv(cv_scores_m1,cv_path)
    rep.save_sm_summary(m1_results,sm_summary_path_m1)
    logger.info("\n" + str(m1_results.summary()))
    logger.info("\n" + m1_global_eval.to_string()) 
  


    logger.info("Training and evaluating model 2")    

    model_2_num_features = config['models']['sm_ols_model_2']['features']['numeric']
    model_2_cat_features = config['models']['sm_ols_model_2']['features']['categorical']
    model_2_features = model_2_num_features + model_2_cat_features
 
    y_train_m2,X_train_m2 = dpr.prep_data_ols_statsmodels(train_set_clean,target='price',features=model_2_features,
                                                          cat_features=model_2_cat_features,log_transform=True)
    model_2 = mod.build_stats_models_lin_regr(y=y_train_m2,X=X_train_m2)
    m2_results = model_2.fit()

    m2_global_eval,cv_scores_m2,m2_binned_scores,m2_bin_raw = (ev.sm_ols_eval_pipe(m2_results,
                                                                features_encoded=X_train_m2,split_group=cv_groups,
                                                                name="ols_model_2",path=plot_dir))
    
    rep.export_to_csv(m2_global_eval,global_csv_path)
    rep.export_to_csv(m2_binned_scores,binned_path)
    rep.export_to_csv(m2_bin_raw,binned_path_raw)
    rep.export_to_csv(cv_scores_m2,cv_path)
    rep.save_sm_summary(m2_results,sm_summary_path_m2)
    logger.info("\n" + str(m2_results.summary()))
    logger.info("\n" + m2_global_eval.to_string())
 

    logger.info("Training and evaluating model 3")

    y_train_m3 = train_set_clean['price']
    numeric_cols = config['models']['hist_model_3']['features']['numeric']
    cat_cols = config['models']['hist_model_3']['features']['categorical']
    X_train_m3 = train_set_clean[numeric_cols]
    X_train_m3[cat_cols] = train_set_clean[cat_cols].astype('category')
    hyperparams_m3 = config['models']['hist_model_3']['hyperparameters']

    model_3 = mod.build_sklearn_histgbr(hyperparams_m3)
    model_3 = model_3.fit(X_train_m3,y_train_m3.squeeze())
    fitted_val_m3 = model_3.predict(X_train_m3)

    m3_global_eval,m3_cv_scores,m3_binned_scores,m3_bin_raw = ev.histgbr_eval_pipe(model=model_3,features=X_train_m3,
                                                                        target=y_train_m3,
                                                                        predicted_vals=fitted_val_m3,
                                                                        split_group=cv_groups,name="model_3",
                                                                        path=plot_dir,perm_imp_params=perm_hyperparams)
    
    rep.export_to_csv(m3_global_eval,global_csv_path)
    rep.export_to_csv(m3_binned_scores,binned_path)
    rep.export_to_csv(m3_bin_raw,binned_path_raw)
    rep.export_to_csv(m3_cv_scores,cv_path)
    logger.info("\n" + m3_global_eval.to_string())
    logger.info("\n" + m3_cv_scores.to_string())

    logger.info("Training and evaluating model 4")

    y_train_m4 = train_set_clean['price']
    numeric_cols = config['models']['hist_model_4']['features']['numeric']
    cat_cols = config['models']['hist_model_4']['features']['categorical']
    X_train_m4 = train_set_clean[numeric_cols]
    X_train_m4[cat_cols] = train_set_clean[cat_cols].astype('category')
    hyperparams_m4 = config['models']['hist_model_4']['hyperparameters']

    model_4 = mod.build_sklearn_histgbr(hyperparams_m4)
    model_4 = model_4.fit(X_train_m4,y_train_m4)
    fitted_val_m4 = model_4.predict(X_train_m4)
   
    m4_global_eval,m4_cv_scores,m4_binned_scores,m4_bin_raw = ev.histgbr_eval_pipe(model=model_4,features=X_train_m4,
                                                                        target=y_train_m4,
                                                                        predicted_vals=fitted_val_m4,
                                                                        split_group=cv_groups,name="model_4",
                                                                        path=plot_dir,perm_imp_params=perm_hyperparams)
                                                                        

    rep.export_to_csv(m4_global_eval,global_csv_path)
    rep.export_to_csv(m4_binned_scores,binned_path)
    rep.export_to_csv(m4_bin_raw,binned_path_raw)
    rep.export_to_csv(m4_cv_scores,cv_path)
    logger.info("\n" + m4_global_eval.to_string())
    logger.info("\n" + m4_cv_scores.to_string())

    logger.info("Comparing Models")

    global_results_all_mods = pd.concat([m1_global_eval,m2_global_eval,m3_global_eval,m4_global_eval])

    logger.info(global_results_all_mods)

    logger.info("Evaluating test set with Model 4 - HGBR")

    test_set = dpr.post_split_clean(test_set)
    
    test_set_clean = (test_set.pipe(dpr.feature_engineering,clean_zip_path)
                    .pipe(dpr.drop_cols,config['col_groups']['col_after_ftr_eng_drop']))

    y_test = test_set_clean['price']
    numeric_cols = config['models']['hist_model_4']['features']['numeric']
    cat_cols = config['models']['hist_model_4']['features']['categorical']
    X_test = test_set_clean[numeric_cols]
    X_test[cat_cols] = test_set_clean[cat_cols].astype('category')

    test_predictions = model_4.predict(X_test)

    test_global_eval,test_binned,test_bin_raw = ev.test_set_hgbr_eval_pipe(model=model_4,features=X_test,
                                                                        target=y_test,
                                                                        predicted_vals=test_predictions,
                                                                        name="test_set_m4_hgrb",
                                                                        path=plot_dir,perm_imp_params=perm_hyperparams)
                                                                        

    rep.export_to_csv(test_global_eval,global_csv_path)
    rep.export_to_csv(test_binned,binned_path)
    rep.export_to_csv(test_bin_raw,binned_path_raw)
    logger.info(test_global_eval)


    rep.build_html_dashboard(csv_dir,plot_dir,run_root)

if __name__ == "__main__":
   
    DATA_PATH = "data/*listings.csv"
    main(DATA_PATH)