import joblib
import numpy as np
import pandas as pd
from preprocessing import prepare_features_from_user_input

# Load all model resources
with open("models/model_resources.joblib", "rb") as f:
    resources = joblib.load(f)

MODEL_PARAMS = resources['model_params']
SCALER_X = resources['scaler_X']
CAT_IMPUTER = resources['cat_imputer']
NUM_IMPUTER = resources['num_imputer']
ENCODER = resources['encoder']
PROPERTY_MAPPING = resources['property_mapping']
MEDIAN_PRICE_PER_NEIGH = resources['median_price_per_neigh']
SELECTED_FEATURES = resources['selected_features']
CAT_COLS = resources['cat_cols']
NUM_COLS = resources['num_cols']
CITY_MEDIANS = resources['city_medians']
ALL_FEATURE_NAMES = resources['all_feature_names']


def predict_price(input_data: dict) -> dict:
    """Main prediction function"""

    # Step 1: Prepare features
    features_df = prepare_features_from_user_input(
        input_data=input_data,
        encoder=ENCODER,
        property_mapping=PROPERTY_MAPPING,
        median_price_per_neigh=MEDIAN_PRICE_PER_NEIGH,
        city_medians=CITY_MEDIANS,
        all_feature_names=ALL_FEATURE_NAMES,
        categorical_cols=CAT_COLS
    )

    # Step 2: Select features
    X = features_df[SELECTED_FEATURES].values  # Convert to numpy array immediately

    # Step 3: Scale
    X_scaled = SCALER_X.transform(X.reshape(1, -1))  # Ensure 2D array

    # Step 4: Add constant
    X_with_const = np.concatenate([np.ones((1, 1)), X_scaled], axis=1)

    # Step 5: Predict
    if hasattr(MODEL_PARAMS, 'values'):
        params = MODEL_PARAMS.values
    else:
        params = np.array(MODEL_PARAMS)

    prediction = float(np.dot(X_with_const, params)[0])

    # Step 6: Confidence interval
    std_dev = prediction * 0.15
    confidence_interval = {
        "lower": float(max(0, prediction - 1.96 * std_dev)),
        "upper": float(prediction + 1.96 * std_dev)
    }

    # Step 7: City comparison
    city_median = float(CITY_MEDIANS.get(input_data['city'], np.median(list(CITY_MEDIANS.values()))))
    city_comparison = {
        "city_average": city_median,
        "difference_pct": float(((prediction - city_median) / city_median) * 100) if city_median > 0 else 0.0
    }

    # Step 8: Feature importance
    feature_importances = get_feature_importance(features_df)

    return {
        "price": prediction,
        "confidence_interval": confidence_interval,
        "top_features": feature_importances[:5],
        "city_comparison": city_comparison
    }


def get_feature_importance(features_df: pd.DataFrame) -> list:
    """Get feature contributions based on coefficient magnitudes"""

    if hasattr(MODEL_PARAMS, 'values'):
        params = MODEL_PARAMS.values
    else:
        params = np.array(MODEL_PARAMS)

    # Skip const (first param)
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