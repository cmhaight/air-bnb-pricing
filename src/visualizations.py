import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np

def create_bar_plot(df: pd.DataFrame,value,category,title: str = "Bar Chart",x_label: str | None = None,
                    y_label: str| None = None) -> sns.barplot:

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df,
        x=value,
        y=category,
        color="steelblue")
    plt.xlabel(x_label if x_label else value)
    plt.ylabel(y_label if y_label else None)
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()  
    ax = plt.gca() 
    return fig, ax


def create_cluster_plot(df: pd.DataFrame,*,geo_column: str,clusters: str,title: str = "Geo Plot"):

    gdf = gpd.GeoDataFrame(df, geometry=geo_column)

    fig, ax = plt.subplots(figsize=(10,20))

    gdf.plot(
    column=clusters,  
    cmap='Set1',            
    markersize=50,          
    legend=True,            
    ax=ax                 
    )   

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig, ax
    

def create_histogram(df: pd.DataFrame,*,fig_cols=3,fig_rows=2,data_cols: list,logged=False):

    fig, ax = plt.subplots(figsize=(20,10),ncols=fig_cols,nrows=fig_rows)

    ax[0,0].hist(df[data_cols[0]],bins=20,color='salmon',edgecolor='white')     
    ax[0,0].set(ylabel="freq",xlabel=data_cols[0])
    ax[0,0].set_ylim(0,700)

    ax[0,1].hist(df[data_cols[1]],bins=40,color="salmon", edgecolor="white")
    ax[0,1].set(ylabel='freq',xlabel=data_cols[1])
    ax[0,1].set_ylim(0,1000)

    ax[0,2].hist(df[data_cols[2]],bins=10,color="salmon", edgecolor="white")
    ax[0,2].set(ylabel='freq',xlabel=data_cols[2])

    ax[1,0].hist(df[data_cols[3]],bins=20,color='salmon',edgecolor='white')
    ax[1,0].set(ylabel='freq',xlabel=data_cols[3])
    ax[1,0].set_ylim(0,100)

    ax[1,1].hist(df[data_cols[4]],bins=50,color='salmon',edgecolor='white')
    ax[1,1].set(ylabel='freq',xlabel=data_cols[4])
    ax[1,1].set_ylim(0,100)

    ax[1,2].hist(df[data_cols[5]],bins=20,color='salmon',edgecolor='white')
    ax[1,2].set(ylabel='freq',xlabel=data_cols[5])

    return fig, ax

def plot_histograms(df: pd.DataFrame,*,cols: list,y_limits: dict[int,tuple[float,float]] | None = None,logged=False):

    bins_list = [20,40,10,20,50,20]
    fig, axes = plt.subplots(figsize=(20,10),ncols=3,nrows=2)
    axes_flat = axes.flatten()

    if logged == True:
        values = np.log1p(df[cols])
    else:
        values = df[cols]

    for i, col in enumerate(cols):

        axes_flat[i].hist(values[col],bins=bins_list[i],color='salmon',edgecolor='white')
        axes_flat[i].set(ylabel='freq',xlabel=col)

        if y_limits and i in y_limits:
            axes_flat[i].set_ylim(y_limits[i])

    plt.tight_layout()
    return fig, axes



def plot_boxplots(df: pd.DataFrame,cols: list,y_labels: list | None = None):

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8),layout="constrained")
    axes_flat = axes.flatten()
    values = df[cols]

    for i, col in enumerate(cols):

        sns.boxplot(
        y=values[col], 
        ax=axes_flat[i],
        color="#4c72b0",  
        fliersize=2,
        width=0.4
        )
        axes_flat[i].set_title(col)
        axes_flat[i].set_ylabel(y_labels[i] if y_labels else col)

    return fig, axes

def plot_boxplot_one_fig(df: pd.DataFrame,*,col: str,bins: str,y_label: str | None = None,x_label: str | None = None,
                         title : str = None, x_axis_order = None):


    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(26, 12),layout="constrained")

    sns.boxplot(
        y=df[col],
        x=df[bins],
        color="#4c72b0",  
        fliersize=5,
        order=x_axis_order)
    ax.tick_params(axis="x", labelsize=16)   
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel(x_label if x_label else bins,fontsize=24)
    ax.set_ylabel(y_label if y_label else col,fontsize=24)
    ax.set_title(title,fontsize=24)
    return fig, ax


def make_scatter_plots(df: pd.DataFrame,*, x_vars: list, y_vars: str | list):

    fig, axes = plt.subplots(figsize=(20,10),ncols=3,nrows=2)
    axes_flat = axes.flatten()

    for i, col in enumerate(x_vars):
        axes_flat[i].scatter(x=df[col],y=df[y_vars[i]],color='salmon',edgecolor='white')
        axes_flat[i].set(ylabel=y_vars[0],xlabel= col)
    return fig, axes


def generate_residuals_plot(*,predicts_orig_scale,target_orig_scale,name='model'):
    fig, ax = plt.subplots(figsize=(10, 6),dpi=200,layout='constrained')
    
    ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")   
    sns.regplot(
        x=target_orig_scale,
        y=predicts_orig_scale,
        ax=ax,
        ci=95,                              
        scatter_kws={'alpha': 0.4, 'color': '#64748b', 's': 25}, 
        line_kws={'color': '#4f46e5', 'linewidth': 2} 
    )
    
    all_data = np.concatenate([target_orig_scale, predicts_orig_scale])
    min_val, max_val = all_data.min(), all_data.max()
    ax.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle=':', alpha=0.8, label='Perfect Fit')
    
    ax.set_title("Actual vs. Predicted", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Actual Values", fontsize=11, fontweight='normal', color='#334155')
    ax.set_ylabel("Predicted Values", fontsize=11, fontweight='normal', color='#334155')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
    return fig


def plot_resids_predicted(results):
    predicted_vals = results.fittedvalues
    residuals = results.resid
    fig, ax =  plt.subplots(figsize=(8, 5),layout="constrained")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
    ax.scatter(predicted_vals, residuals, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5) 
    ax.set_title('Residual Plot',fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Values',fontsize=11, fontweight='normal', color='#334155')
    ax.set_ylabel('Residuals (Actual - Predicted)',fontsize=11, fontweight='normal', color='#334155')
    return fig


def plot_resids_sklearn(*,pred_vals,resids):
    fig, ax =  plt.subplots(figsize=(8, 5),layout="constrained")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
    ax.scatter(pred_vals, resids, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5) 
    ax.set_title("Residual Plot", fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Residuals (Actual - Predicted)',fontsize=11, fontweight='normal', color='#334155')
    ax.set_xlabel("Predicted Values", fontsize=11, fontweight='normal', color='#334155') 
    return fig


def plot_resids_features(results,features):
    all_features = results.model.exog_names
    exog_data = results.model.exog
    residuals = results.resid
    all_features_clean = [f.strip().lower() for f in all_features]

    fig, axes = plt.subplots(figsize=(22,12),ncols=3,nrows=2)
    axes_flat = axes.flatten()
    plot_index = 0
    for feat in features:
        clean_feature = feat.strip().lower()

        if clean_feature in all_features_clean:
            if plot_index >= 5:
                break
            col_index = all_features_clean.index(clean_feature)
            orig_name = all_features_clean[col_index]
  
            feature_values = exog_data[:,col_index]
            axes_flat[plot_index].scatter(x=feature_values,y=residuals,color='blue',edgecolor='white',alpha=0.5,s=80)
            axes_flat[plot_index].set_ylabel('Residuals (Actual - Predicted)',fontsize=18)
            axes_flat[plot_index].set_xlabel(orig_name, fontsize=22)
            axes_flat[plot_index].axhline(y=0.0, color='r', linestyle='--', linewidth=2)
            plot_index += 1
    plt.tight_layout()
    return fig


def plot_resid_boxplt(results,df,feature):

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(24, 10))

    sns.boxplot(y=results.resid,x=df[feature],color="#4c72b0",fliersize=5)
    plt.axhline(0, color='red', linestyle='--')
    ax.set_xlabel(feature,fontsize=22)
    ax.set_ylabel("Residuals (Actual - Predicted)",fontsize=18)
    ax.set_title(f'Error Distribution by {feature}',fontsize=18)
    ax.tick_params(axis="x", labelsize=16)   
    ax.tick_params(axis="y", labelsize=16)
    plt.tight_layout()
    return fig


def feat_imp_bar_chart_ols(df):
   
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.grid(True, axis='x', linestyle="--", alpha=0.6, color="#cbd5e1")
    ax.set_axisbelow(True)
    
    features = df.index.tolist()
    percentages = df.values.flatten().tolist()
    colors = ['#4f46e5' if val >= 0 else '#64748b' for val in percentages]
    bars = ax.barh(features, percentages, color=colors, alpha=0.9, edgecolor='none', height=0.6)

    ax.axvline(0, color='#334155', linestyle='-', linewidth=1.2, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        ha_alignment = 'left' if width >= 0 else 'right'
        offset = 1 if width >= 0 else -1   
        plt.text(
            width + offset,
            bar.get_y() + bar.get_height()/2,
            f"{'+' if width >= 0 else ''}{width:.1f}%",
            va='center',
            ha=ha_alignment,
            fontsize=9,
            fontweight='normal',
            color='#334155'
        )    
    ax.set_title("Top 10 Drivers by Estimated % Impact (OLS)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Real-World Price Change Effect (%)", fontsize=11, fontweight='normal', color='#334155') 
    max_abs = np.abs(df.values).max()
    max_pos = df.values.max()
    ax.set_xlim(-max_abs * 1.2, max_pos * 1.6)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def feat_imp_bar_chart_hgbr(df):
   
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.grid(True, axis='x', linestyle="--", alpha=0.6, color="#cbd5e1")
    ax.set_axisbelow(True)
    
    features = df['feature_names'].tolist()
    mean_decrease = df['mean_r2_decrease']
    colors = ['#4f46e5' if val >= 0 else '#64748b' for val in mean_decrease]
    bars = ax.barh(features, mean_decrease,xerr=df['std_r2_decrease'], color=colors,ecolor='#555555',capsize=4,
                   error_kw={'elinewidth': 3, 'capthick': 1.5}, alpha=0.9, edgecolor='none', height=0.6)

    ax.axvline(0, color='#334155', linestyle='-', linewidth=1.2, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        ha_alignment = 'left' if width >= 0 else 'right'
        plt.text(
            width + 0.015,
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            va='center',
            ha=ha_alignment,
            fontsize=9,
            fontweight='normal',
            color='#334155'
        )    
    ax.set_title("Top 10 Feature Importances (HistGradientBoosting)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Decrease in R^2 (Permutation Importance)", fontsize=11, fontweight='normal', color='#334155') 
    max_val = df['mean_r2_decrease'].max()
    ax.set_xlim(0,max_val * 1.15)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig