import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

def build_stats_models_lin_regr(*,y,X): 
    return sm.OLS(y,X)


def build_sklearn_histgbr(hyperparams):
    return HistGradientBoostingRegressor(**hyperparams)

