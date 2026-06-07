from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def process_companies(companies: pd.DataFrame) -> pd.DataFrame:
    companies = companies.copy()
    companies["iata_approved"] = companies["iata_approved"] == "t"
    companies["company_rating"] = (
        companies["company_rating"].str.replace("%", "", regex=False).astype(float)
    )
    return companies


def process_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    shuttles = shuttles.copy()
    shuttles["d_check_complete"] = shuttles["d_check_complete"] == "t"
    shuttles["moon_clearance_complete"] = shuttles["moon_clearance_complete"] == "t"
    shuttles["price"] = (
        shuttles["price"].str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    return shuttles


def create_model_input_table(
    processed_companies: pd.DataFrame,
    reviews: pd.DataFrame,
    processed_shuttles: pd.DataFrame,
) -> pd.DataFrame:
    rated_shuttles = processed_shuttles.merge(
        reviews, left_on="id", right_on="shuttle_id"
    )
    model_input_table = rated_shuttles.merge(
        processed_companies, left_on="company_id", right_on="id"
    )
    return model_input_table.dropna()


def prepare_features(model_input_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = model_input_table[
        [
            "engines",
            "passenger_capacity",
            "crew",
            "d_check_complete",
            "moon_clearance_complete",
            "iata_approved",
            "company_rating",
            "review_scores_rating",
        ]
    ]
    y = model_input_table["price"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    model_train: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=model_train["test_size"],
        random_state=model_train["random_state"],
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_model(model: LinearRegression, X_test: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X_test), index=X_test.index)
