import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import glob
import ast
import statsmodels.api as sm
from zlib import crc32
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

def csv_to_df(path): 
    csv_files = glob.glob(path)
    df_combined_raw = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    df = df_combined_raw.copy()
    logger.info("Raw data loaded")
    return df


def geo_zip_encode(df,file_path):
    df_geo = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df["longitude"],df["latitude"]),crs="EPSG:4326")

    zips = gpd.read_file(
        file_path,
        engine="pyogrio")
    zips = zips.to_crs(df_geo.crs)
    
    df = gpd.sjoin(
        df_geo,
        zips[["ZCTA5CE20", "geometry"]],
        how="left",
        predicate="within")
    
    df = df.rename(
        columns={"ZCTA5CE20": "zipcode"})
    df = df.drop(columns=['index_right'])
    return df


def pre_split_col_drop(df: pd.DataFrame,cols: list):
    logger.info("Cleaning and splitting data...")
    cols_keep_clean = df.columns.intersection(cols)
    df = df[cols_keep_clean].copy()
    return df


def clean_format(df: pd.DataFrame):
    df['price'] = df['price'].str.replace(r"[$,]", "", regex=True)
    df['price'] = pd.to_numeric(df['price'],errors='coerce')
    df["month"] = pd.to_datetime(df["last_scraped"]).dt.month_name()
    return df


def drop_cols(df: pd.DataFrame,cols: list):
    df = df.drop(columns=cols)
    return df


def filter_pricing(df):
    df = df[df['price'] < 4000]

    accomodate_anomaly_mask = (df['accommodates'] <= 4) & (df['price'] > 3000)
    df = df[~accomodate_anomaly_mask]
    
    min_night_anomaly_mask = (
        ((df['minimum_nights'] >= 60) & (df['price'] > 300)) | 
        ((df['minimum_nights'] >= 30) & (df['minimum_nights'] < 60) & (df['price'] > 500)))
    
    df = df[~min_night_anomaly_mask].copy()
    return df


def drop_na_othr_columns(df):
    df = df.dropna(subset=['minimum_nights','bathrooms_text','price'])
    return df


def fill_na_columns(df):
    review_cols_nums = ['review_scores_rating']
    train_medians = df[review_cols_nums].median()
    text_to_digit = {'half-bath': '0.5'}

    df['bathrooms_text'] = df['bathrooms_text'].astype(str).str.lower()
    df['bathrooms_text'] = df['bathrooms_text'].replace(text_to_digit,regex=True)
    df['bathrooms_text'] = df['bathrooms_text'].str.extract(r'(\d*\.?\d+)')
    df['bathrooms_text'] = pd.to_numeric(df['bathrooms_text'],errors='coerce')
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms_text'])

    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')

    df[review_cols_nums] = df[review_cols_nums].fillna(train_medians)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        
    bed_median_map = df.groupby('accommodates')['beds'].median()
    train_bed_medians = df['accommodates'].map(bed_median_map)
    df['beds'] = df['beds'].fillna(train_bed_medians)
        
    bedroom_median_map = df.groupby('beds')['bedrooms'].median()
    train_bedroom_medians = df['beds'].map(bedroom_median_map)
    df['bedrooms'] = df['bedrooms'].fillna(train_bedroom_medians)
    df = df.drop(columns=['bathrooms_text'])
    return df


def per_accom_feature(df):
    df['bed_per_accom'] = df['beds'].div(df['accommodates'].replace(0,np.nan))
    df['bath_per_accom'] = df['bathrooms'].div(df['accommodates'].replace(0,np.nan))
    return df


def time_format_change_month(df: pd.DataFrame):
    df["month"] = pd.to_datetime(df["last_scraped"]).dt.month_name()
    df = df.drop(columns=['last_scraped'])
    return df


def cluster_zipcodes(df):
    scaler = StandardScaler()
    all_scaled_coords = scaler.fit_transform(df[['longitude','latitude']])
    hc_all = AgglomerativeClustering(n_clusters=20, linkage='ward')
    cluster_numbers = hc_all.fit_predict(all_scaled_coords)
    df['pure_spatial_cluster'] = [f"Macro_Zone_{x}" for x in cluster_numbers]
    df['pure_spatial_cluster'].value_counts(dropna=False)
    return df


def extract_amenities(df):
    amen_list =['outdoor dining area','patio or balcony','long term stays allowed','life size games','bbq grill',
                'fire pit','private patio or balcony','fireplace',r'\bpool(?!.*table)','hot tub','free parking',
                'air conditioning']
    target_list = '|'.join(amen_list)

    df['amenities'] = df['amenities'].apply(ast.literal_eval)
    exploded_df = df.explode('amenities')['amenities']
    filtered_df_exploded = exploded_df[exploded_df.str.contains(target_list, case=False, na=False)]
    df['has_pool'] = (filtered_df_exploded.str.contains('pool', case=False, na=False)
                      .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['hot_tub'] = (filtered_df_exploded.str.contains('hot tub', case=False, na=False)
                     .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['free_parking'] = (filtered_df_exploded.str.contains('free parking', case=False, na=False)
                          .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['fireplace'] = (filtered_df_exploded.str.contains('fireplace', case=False, na=False)
                       .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['ac'] = (filtered_df_exploded.str.contains('conditioning', case=False, na=False)
                .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['patio_balcony'] = (filtered_df_exploded.str.contains('patio|balcony', case=False, na=False)
                           .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    df['life_size_games'] = (filtered_df_exploded.str.contains('life size games', case=False, na=False)
                             .groupby(level=0).any().reindex(df.index, fill_value=False).astype(int))
    return df


def bin_min_nights(df):
    conditons = ([(df['minimum_nights'] <=2),((df['minimum_nights'] > 2) & (df['minimum_nights'] < 5)),
                 ((df['minimum_nights'] >= 5) & (df['minimum_nights'] < 30)),
                (df['minimum_nights'] == 30),((df['minimum_nights'] > 30) & (df['minimum_nights'] < 60)),
                (df['minimum_nights'] >= 60)])
    
    buckets = (['min_nights_one_two','min_nights_three_four','min_night_five_to_29','min_night_30','min_night_30_60',
               'min_over_60'])
    
    df['minimum_nights_cat'] = np.select(conditons,buckets,default='Unknown')
    df['minimum_nights_cat'].value_counts()
    return df


def bin_accommodates(df):
    conditons = ([df['accommodates'] == 1,df['accommodates'] == 2,df['accommodates'] == 3,df['accommodates'] == 4,
                 df['accommodates'] == 5,df['accommodates'] == 6,df['accommodates'] == 7,df['accommodates'] == 8, 
                 ((df['accommodates'] == 10) | (df['accommodates'] == 9)),
                 ((df['accommodates'] == 11) | (df['accommodates'] == 12)),(df['accommodates'] >= 13)])
    
    buckets = (['one_guest','two_guests','three_guests','four_guests','five_guests','six_guests','seven_guests',
               'eight_guests','nine_ten_guests','eleven_twelve_guests','thirteen_plus_guests'])
    
    df['accommodates_cat'] = np.select(conditons,buckets,default='Unknown')
    df['accommodates_cat'].value_counts()
    return df


def bin_bathrooms(df):
    conditons = ([(df['bathrooms'] < 1.0), (df['bathrooms'] == 1.0),(df['bathrooms'] > 1.0) & (df['bathrooms'] <= 2.0),
                  (df['bathrooms'] > 2.0) & (df['bathrooms'] <= 3.5),(df['bathrooms'] > 3.5) & (df['bathrooms'] < 6.0),
                  (df['bathrooms'] >= 6.0 ) & (df['bathrooms'] <= 8),(df['bathrooms'] > 8.0)])
    
    buckets = ['bath_less_1','bath_1','bath_1.5_2','bath_2.5_3.5','bath_4_6','bath_6_8','bath_8_plus']
    
    df['bathrooms_cat'] = np.select(conditons,buckets,default="Unknown")
    df["bathrooms_cat"].value_counts()
    return df


def condense_prop_type(df):
    top_prop_types = df['property_type'].value_counts().nlargest(5).index
    df['property_type_clean'] = df['property_type'].where(df['property_type'].isin(top_prop_types), 'Other')
    return df


def is_id_in_test_set(identifier,test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def log_re_transformation(results,use_smearing=True):
    target_values = pd.Series(results.model.endog,index=results.model.data.row_labels,name=results.model.endog_names)
    predicted_values = results.fittedvalues
    residuals = results.resid
    if use_smearing:
        smearing_factor = np.mean(np.exp(residuals))
        predictions_orig_scale = np.exp(predicted_values) * smearing_factor
    else:
        predictions_orig_scale = np.exp(predicted_values)
    target_values_orig_scale = np.exp(target_values)
    return predictions_orig_scale,target_values_orig_scale


def prep_data_ols_statsmodels(df: pd.DataFrame,*,features: list,target: str,cat_features = None,log_transform=False): 
    
    X = df[features].copy()
    X = pd.get_dummies(X,columns=cat_features,drop_first=True, dtype=int)
    X = sm.add_constant(X)

    if log_transform:
        y = np.log(df[target]).copy()
    else:
        y = df[target].copy()
    return y,X


def post_split_clean(df: pd.DataFrame):
    df = df.copy()
    processed_df = (
        df
        .pipe(drop_na_othr_columns)
        .pipe(fill_na_columns)
        .pipe(clean_format)
        .pipe(filter_pricing))
    logger.info("Data cleaned and split")
    return processed_df


def feature_engineering(df: pd.DataFrame,file_path):
    logger.info("Engineering features...")
    df = df.copy()
    processed_df = (
        df
        .pipe(per_accom_feature)
        .pipe(geo_zip_encode,file_path)
        .pipe(cluster_zipcodes)
        .pipe(extract_amenities)
        .pipe(bin_min_nights)
        .pipe(bin_accommodates)
        .pipe(bin_bathrooms)
        .pipe(condense_prop_type)

    )
    logger.info("Feature engineering complete")
    return processed_df




