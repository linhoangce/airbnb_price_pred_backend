import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from preprocessing import prepare_features_from_user_input

# Load all model resources
with open("models/linear_reg_model_resources.joblib", "rb") as f:
    resources = joblib.load(f)

MODEL_PARAMS = resources['model_params']
SCALER_X = resources['scaler_X']
ENCODER = resources['encoder']
PROPERTY_MAPPING = resources['property_mapping']
MEDIAN_PRICE_PER_NEIGH = resources['median_price_per_neigh']
SELECTED_FEATURES = resources['selected_features']
CAT_COLS = resources['cat_cols']
NUM_COLS = resources['num_cols']
CITY_MEDIANS = resources['city_medians']
ALL_FEATURE_NAMES = resources['all_feature_names']

# Load XGBoost model
XGB_MODEL = xgb.XGBRegressor()
XGB_MODEL.load_model("models/xgboost_model.json")


def predict_price(input_data: dict, model_type: str = "ols") -> dict:
    """
    Main prediction function - supports both OLS and XGBoost

    Args:
        input_data: User input dictionary
        model_type: "ols" or "xgboost"
    """

    # Prepare features
    features_df = prepare_features_from_user_input(
        input_data=input_data,
        encoder=ENCODER,
        property_mapping=PROPERTY_MAPPING,
        median_price_per_neigh=MEDIAN_PRICE_PER_NEIGH,
        city_medians=CITY_MEDIANS,
        all_feature_names=ALL_FEATURE_NAMES,
        categorical_cols=CAT_COLS
    )

    # Select features
    X = features_df[SELECTED_FEATURES].values

    if model_type.lower() == "xgboost":
        # XGBoost prediction (no scaling needed)
        X_input = features_df[SELECTED_FEATURES]
        prediction = float(XGB_MODEL.predict(X_input)[0])
        model_name = "XGBoost"

    else:  # OLS
        # Scale for OLS
        X_scaled = SCALER_X.transform(X.reshape(1, -1))
        X_with_const = np.concatenate([np.ones((1, 1)), X_scaled], axis=1)

        # Get params
        if hasattr(MODEL_PARAMS, 'values'):
            params = MODEL_PARAMS.values
        else:
            params = np.array(MODEL_PARAMS)

        prediction = float(np.dot(X_with_const, params)[0])
        model_name = "OLS Regression"

    # Confidence interval
    std_dev = prediction * 0.15
    confidence_interval = {
        "lower": float(max(0, prediction - 1.96 * std_dev)),
        "upper": float(prediction + 1.96 * std_dev)
    }

    # City comparison
    city_median = float(CITY_MEDIANS.get(input_data['city'], np.median(list(CITY_MEDIANS.values()))))
    city_comparison = {
        "city_average": city_median,
        "difference_pct": float(((prediction - city_median) / city_median) * 100) if city_median > 0 else 0.0
    }

    # Feature importance
    if model_type.lower() == "xgboost":
        feature_importances = get_xgb_feature_importance(features_df)
    else:
        feature_importances = get_ols_feature_importance(features_df)

    return {
        "price": prediction,
        "model": model_name,
        "confidence_interval": confidence_interval,
        "top_features": feature_importances[:5],
        "city_comparison": city_comparison
    }


def get_ols_feature_importance(features_df: pd.DataFrame) -> list:
    """Get OLS feature importance based on coefficient magnitudes"""

    if hasattr(MODEL_PARAMS, 'values'):
        params = MODEL_PARAMS.values
    else:
        params = np.array(MODEL_PARAMS)

    coef_abs = np.abs(params[1:])

    feature_contributions = []
    for i, feat in enumerate(SELECTED_FEATURES):
        feature_contributions.append({
            "feature": feat,
            "value": float(features_df[feat].iloc[0]),
            "importance": float(coef_abs[i]),
            "coefficient": float(params[i + 1])
        })

    feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
    return feature_contributions


def get_xgb_feature_importance(features_df: pd.DataFrame) -> list:
    """Get XGBoost feature importance"""

    # Get feature importance from XGBoost model
    importance_dict = XGB_MODEL.get_booster().get_score(importance_type='gain')

    # Map to feature names (XGBoost uses f0, f1, f2... internally)
    feature_contributions = []
    for i, feat in enumerate(SELECTED_FEATURES):
        importance = importance_dict.get(f'f{i}', 0)
        feature_contributions.append({
            "feature": feat,
            "value": float(features_df[feat].iloc[0]),
            "importance": float(importance),
            "coefficient": float(importance)  # For consistency with OLS
        })

    feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
    return feature_contributions