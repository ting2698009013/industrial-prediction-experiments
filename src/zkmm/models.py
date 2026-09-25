def make_regressor(model_name='lgbm', fast=False, random_state=42):
    model_name=model_name.lower()
    if model_name=='lgbm':
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=300 if fast else 900,learning_rate=0.06,max_depth=8,num_leaves=96,min_child_samples=80,subsample=0.85,colsample_bytree=0.85,reg_alpha=0.1,reg_lambda=1.0,random_state=random_state,n_jobs=-1,verbose=-1)
        except Exception: model_name='histgb'
    if model_name=='xgb':
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300 if fast else 800,max_depth=8,learning_rate=0.06,min_child_weight=30,subsample=0.85,colsample_bytree=0.85,reg_lambda=1.0,objective='reg:squarederror',tree_method='hist',n_jobs=-1,random_state=random_state,verbosity=0)
        except Exception: model_name='histgb'
    if model_name=='cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300 if fast else 800,depth=8,learning_rate=0.06,l2_leaf_reg=10,random_seed=random_state,verbose=0,loss_function='RMSE')
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=250 if fast else 600,learning_rate=0.06,max_leaf_nodes=63,l2_regularization=0.1,random_state=random_state)
