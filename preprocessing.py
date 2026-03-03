import pandas as pd
import numpy as np


def prepare_features_from_user_input(input_data: dict, encoder, property_mapping,
                                     median_price_per_neigh, city_medians, all_feature_names,
                                     categorical_cols):
    """
    Convert user input to processed features matching training pipeline EXACTLY
    """
    # Create dataframe with user inputs
    df = pd.DataFrame([{
        'accommodates': input_data['accommodates'],
        'bedrooms': input_data['bedrooms'],
        'bathrooms': input_data['bathrooms'],
        'beds': input_data['beds'],
        'cleaning_fee': 1 if input_data['cleaning_fee'] > 0 else 0,
        'review_scores_rating': input_data.get('review_scores_rating', 90),
        'number_of_reviews': input_data.get('number_of_reviews', 10),
        'host_response_rate': input_data.get('host_response_rate', 95),
        'host_identity_verified': 1 if input_data.get('host_identity_verified', True) else 0,
        'host_has_profile_pic': 1,
        'instant_bookable': 0,
        'property_type_grouped': property_mapping.get(input_data['property_type'], 'Other'),
        'room_type': input_data['room_type'],
        'bed_type': 'Real Bed',
        'cancellation_policy': input_data['cancellation_policy'],
        'city': input_data['city'],
        'host_exp': 'mid'
    }])

    # Get neighborhood median price using COMPUTED city median
    df['neighbourhood_median_price'] = city_medians.get(
        input_data['city'],
        np.median(list(city_medians.values()))
    )

    # One-hot encode categorical columns
    # categorical_cols = ['property_type_grouped', 'room_type', 'bed_type',
    #                     'cancellation_policy', 'city', 'host_exp']
    numeric_cols = ['accommodates', 'bedrooms', 'bathrooms', 'beds',
                    'cleaning_fee', 'review_scores_rating', 'number_of_reviews',
                    'host_response_rate', 'host_identity_verified',
                    'host_has_profile_pic', 'instant_bookable',
                    'neighbourhood_median_price']

    # Encode using the fitted encoder
    encoded = encoder.transform(df[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=cat_feature_names, index=df.index)

    # Combine numeric + encoded
    df_processed = pd.concat([df[numeric_cols], encoded_df], axis=1)

    # Create interaction features
    df_processed = create_feature_interactions(df_processed)

    # Ensure ALL features from training exist (fill missing with 0)
    for feat in all_feature_names:
        if feat not in df_processed.columns:
            df_processed[feat] = 0

    # Return only features that existed in training, in same order
    return df_processed[all_feature_names]


def create_feature_interactions(df):
    """
    Create polynomial and interaction features - EXACTLY as in training
    """
    # Polynomial features
    df['accommodates_squared'] = df['accommodates'] ** 2
    df['bedrooms_squared'] = df['bedrooms'] ** 2
    df['bathrooms_squared'] = df['bathrooms'] ** 2
    df['beds_squared'] = df['beds'] ** 2

    # Size interactions
    df['accommodates_bedrooms'] = df['accommodates'] * df['bedrooms']
    df['accommodates_bathrooms'] = df['accommodates'] * df['bathrooms']
    df['bedrooms_bathrooms'] = df['bedrooms'] * df['bathrooms']
    df['accommodates_beds'] = df['accommodates'] * df['beds']

    # City-specific pricing
    for city in ['LA', 'SF', 'NYC', 'Chicago']:
        if f'city_{city}' in df.columns:
            df[f'{city}_neighbourhood_price'] = df[f'city_{city}'] * df['neighbourhood_median_price']
            df[f'{city}_accommodates'] = df[f'city_{city}'] * df['accommodates']

    # Room type interactions
    if 'room_type_Private room' in df.columns:
        df['private_room_bedrooms'] = df['room_type_Private room'] * df['bedrooms']
        df['private_room_accommodates'] = df['room_type_Private room'] * df['accommodates']

    if 'room_type_Shared room' in df.columns:
        df['shared_room_accommodates'] = df['room_type_Shared room'] * df['accommodates']

    # Property type interactions
    if 'property_type_grouped_House' in df.columns:
        df['house_bedrooms'] = df['property_type_grouped_House'] * df['bedrooms']
        df['house_accommodates'] = df['property_type_grouped_House'] * df['accommodates']

    # Reviews interactions
    df['reviews_rating'] = df['number_of_reviews'] * df['review_scores_rating']
    df['reviews_per_accommodates'] = df['number_of_reviews'] / (df['accommodates'] + 1)
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)

    # Cancellation policy
    if 'cancellation_policy_strict' in df.columns and 'cancellation_policy_super_strict_60' in df.columns:
        df['strict_policy_accommodates'] = (
                                                   df['cancellation_policy_strict'] + df[
                                               'cancellation_policy_super_strict_60']
                                           ) * df['accommodates']

    # Other interactions
    df['cleaning_per_accommodates'] = df['cleaning_fee'] / (df['accommodates'] + 1)
    df['verified_response_rate'] = df['host_identity_verified'] * df['host_response_rate']
    df['luxury_indicator'] = (
            (df['bedrooms'] >= 3).astype(int) *
            (df['bathrooms'] >= 2).astype(int) *
                df['neighbourhood_median_price']
    )

    return df