import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from . import visualizations as viz
import shutil
import logging

logger = logging.getLogger(__name__)

def build_html_dashboard(csv_dir: Path, plot_dir: Path, run_root: Path):
    """
    Extracts graphs and data from saved CSVs and uses them to create an HTML summary of the model performance. 
    """
    with open ("config.yaml","r") as file:
        config = yaml.safe_load(file)
   
    global_csv_path = csv_dir / config['paths']['output_paths']['csv_paths']['global_path']
    binned_path = csv_dir / config['paths']['output_paths']['csv_paths']['binned_path']
    binned_path_raw = csv_dir / config['paths']['output_paths']['csv_paths']['binned_path_raw']
    cv_path = csv_dir / config['paths']['output_paths']['csv_paths']['cv_path']
      
    df_global = pd.read_csv(global_csv_path)
    df_accuracy = pd.read_csv(binned_path)
    df_bias = pd.read_csv(binned_path_raw)
    df_cv = pd.read_csv(cv_path)

    display_names = {
    "model_name": "Model Name",
    "r2_score": "R² Score",
    "mape": "MAPE",
    "mae": "MAE",
    "rmse": "RMSE",
    "cv_avg_mae": "Average MAE",
    "cv_mae_std_dev": "MAE Standard Deviation",
    "cv_avg_rmse": "Average RMSE",
    "cv_rmse_std_dev": "RMSE Standard Deviation",
    "bin_type": "Bin Type",
    "price_group":"Price Group",
    "mean": "Mean",
    "median": "Median",
    "count": "Count"
    }

    model_names = {
    "ols_model_1": "Model 1 - OLS",
    "ols_model_2": "Model 2 - OLS",
    "model_3": "Model 3 - HGBR",
    "model_4": "Model 4 - HGBR",
    "test_set_m4_hgrb" : "Test Set Evaulated with Model 4 - HGBR"
    }

    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    tables_styles_indiv = f""" "classes="w-full text-left text-base text-slate-600 border-collapse" """
    
    html_global_scores = (
            df_global.iloc[:-1,:]
            .assign(
             model_name=df_global["model_name"].map(model_names).fillna(df_global["model_name"]))
            .rename(columns=display_names)
            .to_html(classes="w-full text-left text-base text-slate-600 border-collapse", 
                    index=False, justify="left", border=0, float_format=lambda x: f"{x:.2f}"))
   
    unique_models = df_global["model_name"].unique()
    unique_models_train_sets = unique_models[:-1]
    test_set_model = unique_models[-1:]
    individual_sections_html = ""
    
    for model in unique_models_train_sets:
        model_title = model_names.get(model,model)
        safe_id = model.lower().replace(" ", "_")
        model_accuracy = df_accuracy[df_accuracy["model_name"] == model].iloc[5:,2:]
        model_bias = df_bias[df_bias["model_name"] == model].iloc[5:,2:]
        model_cv = df_cv[df_cv['model_name'] == model].iloc[:,1:]
        model_global = df_global[df_global['model_name'] == model].iloc[:,1:]
        
        html_mod_accuracy = (model_accuracy.rename(columns=display_names).to_html
                            (classes = tables_styles_indiv,index=False, justify="left", border=0,
                             float_format=lambda x: f"{x:.2f}"))
        html_mod_bias = (model_bias.rename(columns=display_names)
                         .to_html(classes=tables_styles_indiv,index=False, justify="left", border=0, 
                                  float_format=lambda x: f"{x:.2f}"))
        html_mod_cv = (model_cv.rename(columns=display_names)
                         .to_html(classes="w-full text-left text-base text-slate-600 border-collapse",
                                  index=False, justify="left", border=0, 
                                  float_format=lambda x: f"{x:.2f}"))
        html_mod_global = (model_global.rename(columns=display_names)
                         .to_html(classes="tight-table w-auto text-left text-base text-slate-600 border-collapse",
                                  index=False, justify="left", border=0, 
                                  float_format=lambda x: f"{x:.2f}"))
           
        individual_sections_html += f"""

        <div id ={safe_id} class="bg-white p-6 rounded-xl shadow-md mb-12 border border-gray-100">
            <h3 class="text-2xl font-bold text-gray-800 border-b pb-2 mb-6 text-center"> 
            {model_title} Individual Analysis</h3>

        <div class="mb-6 max-w-5xl mx-auto">
            <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 text-center">
                Overall Model Scores
            </p>
            <div class="overflow-x-auto mb-6 max-w-4xl mx-auto shadow-sm rounded-xl border border-slate-100 bg-white p-4">
                {html_mod_global}
            </div>
            <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 text-center">
                5-Fold Cross-Validation Metrics 
            </p>
            <div class="overflow-x-auto shadow-sm rounded-xl border border-slate-100 bg-white p-4">
                {html_mod_cv}
            </div>
        </div>            
            <!-- Graphs Side-by-Side -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm font-semibold text-gray-500 mb-2">Residuals vs Predicted </p>
                    <a href="#modal-{model}-resid">
                        <img src="plots/{model}_resid_vs_pred.png" class="zoomable-graph rounded-lg shadow-sm" 
                        alt="{model} Residuals" 
                        class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                    <div id="modal-{model}-resid" class="graph-modal-overlay">
                        <a href="#_" class="modal-close-backdrop"></a>
                        <div class="modal-content-container">
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                                <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                            </div>   
                    <img src="plots/{model}_resid_vs_pred.png" class="max-w-full max-h-[80vh] rounded-lg">
                        </div>
                    </div>
                </div>
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm font-semibold text-gray-500 mb-2">Error Percentage Distribution 
                    (Real-World Spread)</p>
                    <a href="#modal-{model}-hist">
                        <img src="plots/{model}_error_hist.png" class="zoomable-graph rounded-lg shadow-sm"
                        alt="{model} Error Distribution" 
                        class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                    <div id="modal-{model}-hist" class="graph-modal-overlay">
                        <a href="#_" class="modal-close-backdrop"></a>
                        <div class="modal-content-container">
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                                <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                            </div>
                    <img src="plots/{model}_error_hist.png" class="max-w-full max-h-[80vh] rounded-lg">
                        </div>
                    </div>
                </div>
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm font-semibold text-gray-500 mb-2">Error Distribution 
                    (Predicted vs. Actual Prices)</p>
                    <a href="#modal-{model}-pred_orig">
                    <img src="plots/{model}_pred_vs_orig_95_ci.png" class="zoomable-graph rounded-lg shadow-sm" 
                    alt="{model} Error Distribution" 
                    class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                    <div id="modal-{model}-pred_orig" class="graph-modal-overlay">
                        <a href="#_" class="modal-close-backdrop"></a>
                        <div class="modal-content-container">
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                                <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                            </div>
                    <img src="plots/{model}_pred_vs_orig_95_ci.png" class="max-w-full max-h-[80vh] rounded-lg">
                        </div>
                    </div>
                </div>
                 <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm font-semibold text-gray-500 mb-2">Feature Importance </p>
                    <a href="#modal-{model}-feat_bar_chart">
                    <img src="plots/{model}_feat_bar_chart.png" class="zoomable-graph rounded-lg shadow-sm" 
                    alt="{model} Residuals" 
                    class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                    <div id="modal-{model}-feat_bar_chart" class="graph-modal-overlay">
                        <a href="#_" class="modal-close-backdrop"></a>
                        <div class="modal-content-container">
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                                <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                            </div>   
                    <img src="plots/{model}_feat_bar_chart.png" class="max-w-full max-h-[80vh] rounded-lg">
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Binned Tables Side-by-Side for this Model -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div>
                    <h4 class="text-md font-bold text-gray-700 mb-2 text-center">
                    Target Price Tier Error Magnitudes</h4>
                    <div class="overflow-x-auto">{html_mod_accuracy}</div>
                </div>
                <div>
                    <h4 class="text-md font-bold text-gray-700 mb-2 text-center">
                    Directional Over/Under Prediction Bias</h4>
                    <div class="overflow-x-auto">{html_mod_bias}</div>
                </div>
            </div>
        </div>
        """

    for model in test_set_model:
            model_title = model_names.get(model,model)
            safe_id = model.lower().replace(" ", "_")
            model_accuracy = df_accuracy[df_accuracy["model_name"] == model].iloc[5:,2:]
            model_bias = df_bias[df_bias["model_name"] == model].iloc[5:,2:]
            model_global = df_global[df_global['model_name'] == model].iloc[:,1:]
            
            html_mod_accuracy = (model_accuracy.rename(columns=display_names).to_html
                                (classes = tables_styles_indiv,index=False, justify="left", border=0,
                                 float_format=lambda x: f"{x:.2f}"))
            html_mod_bias = (model_bias.rename(columns=display_names)
                             .to_html(classes=tables_styles_indiv,index=False, justify="left", border=0, 
                                      float_format=lambda x: f"{x:.2f}"))
      
            html_mod_global = (model_global.rename(columns=display_names)
                             .to_html(classes="tight-table w-auto text-left text-base text-slate-600 border-collapse",
                                      index=False, justify="left", border=0, 
                                      float_format=lambda x: f"{x:.2f}"))

    test_section = f"""
    <div id ={safe_id} class="bg-white p-6 rounded-xl shadow-md mb-12 border border-gray-100">
        <h3 class="text-2xl font-bold text-gray-800 border-b pb-2 mb-6 text-center"> 
            {model_title} Individual Analysis</h3>
        <div class="mb-6 max-w-5xl mx-auto">
        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 text-center">
                Overall Model Scores</p>
            <div class="overflow-x-auto mb-6 max-w-4xl mx-auto shadow-sm rounded-xl border border-slate-100 bg-white p-4">
                        {html_mod_global}
            </div>
        </div>            
        <!-- Graphs Side-by-Side -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div class="text-center p-4 bg-gray-50 rounded-lg">
                <p class="text-sm font-semibold text-gray-500 mb-2">Error Distribution (Predicted vs. Actual Prices)</p>
                    <a href="#modal-{model}-pred_orig">
                    <img src="plots/{model}_pred_vs_orig_95_ci.png" class="zoomable-graph rounded-lg shadow-sm" 
                        alt="{model} Error Distribution" 
                        class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                <div id="modal-{model}-pred_orig" class="graph-modal-overlay">
                    <a href="#_" class="modal-close-backdrop"></a>
                    <div class="modal-content-container">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                    <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                        </div>
                        <img src="plots/{model}_pred_vs_orig_95_ci.png" class="max-w-full max-h-[80vh] rounded-lg">
                    </div>
                </div>
            </div>
            <div class="text-center p-4 bg-gray-50 rounded-lg">
                <p class="text-sm font-semibold text-gray-500 mb-2">Error Percentage Distribution (Real-World Spread)</p>
                    <a href="#modal-{model}-hist">
                    <img src="plots/{model}_error_hist.png" class="zoomable-graph rounded-lg shadow-sm"
                        alt="{model} Error Distribution" 
                        class="mx-auto rounded border shadow-sm max-h-[300px] object-contain">
                    </a>
                <div id="modal-{model}-hist" class="graph-modal-overlay">
                    <a href="#_" class="modal-close-backdrop"></a>
                    <div class="modal-content-container">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-sm font-semibold text-slate-700">Detailed Chart View</span>
                    <a href="#_" class="text-slate-400 hover:text-slate-600 text-lg font-bold">✕ Close</a>
                        </div>
                        <img src="plots/{model}_error_hist.png" class="max-w-full max-h-[80vh] rounded-lg">
                    </div>
                </div>
            </div>
        </div>                                         
                    <!-- Binned Tables Side-by-Side for this Model -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
                        <div>
                            <h4 class="text-md font-bold text-gray-700 mb-2 text-center">
                            Target Price Tier Error Magnitudes</h4>
                            <div class="overflow-x-auto">{html_mod_accuracy}</div>
                        </div>
                        <div>
                            <h4 class="text-md font-bold text-gray-700 mb-2 text-center">
                            Directional Over/Under Prediction Bias</h4>
                            <div class="overflow-x-auto">{html_mod_bias}</div>
                        </div>
                    </div>
        </div>
        """
        
    html_template = f"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Dashboard</title>
    <!-- Tailwind CSS for modern styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts (Inter) -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}

        .leaderboard-container {{
        margin: 0 auto;            
        max-width: 100%;        
        }}

        .leaderboard-container table {{
        width: 100% !important;
        border-collapse: collapse;
        }}

        .leaderboard-container td {{
        font-size: 16px !important;
        text-align: left !important;
        padding: 12px 16px;
        border-bottom: 1px solid #edf2f7;
         }}

        .grid table {{
        width: 100% !important;
        border-collapse: collapse;
        table-layout: fixed; 
        }}

        .grid th {{
        font-size: 14px !important;
        text-align: left !important;
        background-color: #f8fafc;
        color: #475569;
        font-weight: 700;
        padding: 12px 16px !important; /* 12px top/bottom, 16px left/right padding */
        border-bottom: 2px solid #e2e8f0;
        }}

        .grid td {{
        font-size: 13px !important;
        text-align: left !important;
        padding: 12px 16px !important; 
        border-bottom: 1px solid #edf2f7;
        color: #334155;
        text-overflow: ellipsis;
        overflow: hidden;
        }}

        .grid tr:hover {{
        background-color: #f8fafc;
        }}


        .zoomable-graph {{
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out !important;
        cursor: zoom-in;
        }}

        .zoomable-graph:hover {{
        transform: scale(1.08) !important; 
        position: relative;
        z-index: 40; 
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1) !important;
        }}
 
        .graph-modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(4px);        
        z-index: 9999;                     
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease-in-out;
        }}

        .graph-modal-overlay:target {{
        opacity: 1;
        pointer-events: auto;
        }}


        .modal-close-backdrop {{
        position: absolute;
        width: 100%;
        height: 100%;
        cursor: zoom-out;
        }}

        .modal-content-container {{
        position: relative;
        background: white;
        padding: 20px;
        border-radius: 16px;
        max-width: 90vw;
        max-height: 90vh;
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        z-index: 10000;
        }}

        table.tight-table {{
        margin: 0 auto; 
        }}
    
        table.tight-table th, 
        table.tight-table td {{
        padding-top: 4px;
        padding-bottom: 4px;
        padding-left: 26px;
        padding-right: 26px  
        }}

    </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased">

    <!-- Sticky Navigation Bar (Acts as a multi-page menu via jump links) -->
    <header class="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-end relative">
          
            
            <!-- Jump Links Menu -->
            <nav class="absolute left-1/2 -translate-x-1/2 flex space-x-6 text-sm font-medium text-slate-600">
                <a href="#summary" class="text-sm font-medium text-slate-600 hover:text-indigo-600 transition">
                Summary</a>
                <a href="#ols_model_1" class="text-sm font-medium text-slate-600 hover:text-indigo-600 transition">
                Model 1 - OLS</a>
                <a href="#ols_model_2" class="text-sm font-medium text-slate-600 hover:text-indigo-600 transition">
                Model 2 - OLS </a>
                <a href="#model_3" class="text-sm font-medium text-slate-600 hover:text-indigo-600 
                transition">Model 3 - HGBR</a>
                <a href="#model_4" class="text-sm font-medium text-slate-600 
                hover:text-indigo-600 transition">Model 4 - HGBR</a>
                <a href="#test_set_m4_hgrb" class="text-sm font-medium text-slate-600 
                hover:text-indigo-600 transition">Test Set with Model 4 - HGBR</a>
            </nav>

            <div class="text-xs font-semibold text-slate-400 z-10 bg-slate-100 px-3 py-1.5 rounded-full">
                Generated: {generation_time}
            </div> 
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
        
        <!-- Summary Section -->
     <section id="summary" class="space-y-12 pt-8 scroll-mt-24">
    <!-- Centered Header Block -->
    <div class="text-center max-w-3xl mx-auto border-b border-slate-200/80 pb-10">
        <div class="text-indigo-600 text-sm font-bold uppercase tracking-widest mb-3">
            Global Summary
        </div>
        
        <h1 class="text-4xl md:text-5xl font-black tracking-tight text-slate-900 mb-4">
            Model Performance Suite
        </h1>
        
    </div>

    <div class="max-w-6xl mx-auto px-4">
        <h4 class="text-xl font-extrabold text-slate-800 mb-6 text-center tracking-wide">
            Model Comparison 
        </h4>
            <div class="leaderboard-container overflow-x-auto shadow-md rounded-xl border border-slate-100 
            bg-white p-4">
                {html_global_scores}
            </div>  
        </div>
    </div>
</section>


 
        <div class="space-y-12">     
            {individual_sections_html}
        </div>
        <div class="space-y-12">     
            {test_section}
        </div>

    </main>

    <!--  Footer -->
    <footer class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center text-xs text-slate-400 border-t 
    border-slate-200/80 mt-20">
        <p class="font-medium text-slate-500"></p>
        <p class="mt-1"></p>
    </footer>

</body>
</html>
"""

    output_html_file = run_root / "model_performance_report.html"
    with open(output_html_file, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"Executive summary dashboard compiled at: {output_html_file}")


def save_sm_summary(results,filepath):

    path = Path(filepath)
    path.write_text(results.summary().as_text())
    
    logger.info(f"Model summary successfully saved to {path}")


def create_output_folder(config_path: Path):
    """
    Creates timestamped output folder for CSVs, graphs, and statsmodels results. 
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(f"outputs/run_{timestamp}")
    run_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_folder / "config_snapshot.yaml")
    csv_dir = run_folder / "data"
    plot_dir = run_folder / "plots"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return run_folder, csv_dir, plot_dir


def export_to_csv(df: pd.DataFrame,path :str):
    file_exists = os.path.exists(path)
    df.to_csv(path,mode='a',index=False,header=not file_exists)


def save_diag_plots(result=None,*,path,predicts_orig_scale,target_orig_scale,model_name='model',logged=False,
                    feat_df=None):
    """
    Creates and saves diagnostic plots that display model performance. 
    """
    resid_orig_scale = target_orig_scale - predicts_orig_scale
    resids_log_scale = np.log(target_orig_scale) - np.log(predicts_orig_scale)
    lower_bound, upper_bound = np.percentile(resid_orig_scale, [1, 99])
    if result is not None:

        fig_resid_pred = viz.plot_resids_predicted(result)
        fig_resid_pred.savefig(f"{path}/{model_name}_resid_vs_pred.png",dpi=200, bbox_inches='tight')
        plt.close(fig_resid_pred)

        fig_feat_imp = viz.feat_imp_bar_chart_ols(feat_df)
        fig_feat_imp.savefig(f"{path}/{model_name}_feat_bar_chart.png",dpi=200, bbox_inches='tight')       
        plt.close(fig_feat_imp)

    if result is None:
        if logged:
            fig_resid_pred = viz.plot_resids_sklearn(pred_vals=np.log(predicts_orig_scale),resids=resids_log_scale)
            fig_resid_pred.savefig(f"{path}/{model_name}_resid_vs_pred.png",dpi=200, bbox_inches='tight')
            plt.close(fig_resid_pred)
            
        else:
            fig_resid_pred =viz.plot_resids_sklearn(pred_vals=predicts_orig_scale,resids=resid_orig_scale)
            fig_resid_pred.savefig(f"{path}/{model_name}_resid_vs_pred.png",dpi=200, bbox_inches='tight')
            plt.close(fig_resid_pred)

        fig_feat_imp = viz.feat_imp_bar_chart_hgbr(feat_df)
        fig_feat_imp.savefig(f"{path}/{model_name}_feat_bar_chart.png",dpi=200, bbox_inches='tight')       
        plt.close(fig_feat_imp)

    plt.figure(figsize=(10,6))
    sns.histplot(x=resid_orig_scale,kde=True,bins=50,stat='percent')
    plt.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
    plt.xlim(lower_bound,upper_bound)
    plt.xlabel("Error Amount",fontsize=11, fontweight='normal', color='#334155')
    plt.ylabel("Percentage of Total Listings",fontsize=11, fontweight='normal', color='#334155')
    fig_2 = plt.gcf()

    fig_2.savefig(f"{path}/{model_name}_error_hist",dpi=200, bbox_inches='tight')

    fig_3 = viz.generate_residuals_plot(predicts_orig_scale=predicts_orig_scale,target_orig_scale=target_orig_scale,
                                        name=model_name)
    fig_3.savefig(f"{path}/{model_name}_pred_vs_orig_95_ci.png",dpi=200, bbox_inches='tight')
    
    plt.close(fig_2)
    plt.close(fig_3)

 

