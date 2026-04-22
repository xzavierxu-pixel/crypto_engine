from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.data.dataset_builder import infer_feature_columns
from src.features.builder import build_feature_frame


class FeatureService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_feature_frame(
        self,
        df: pd.DataFrame,
        horizon_name: str | None = None,
        select_grid_only: bool | None = None,
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return build_feature_frame(
            df,
            self.settings,
            horizon_name=horizon_name,
            select_grid_only=select_grid_only,
            derivatives_frame=derivatives_frame,
        )

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        horizon_name: str | None = None,
        derivatives_frame: pd.DataFrame | None = None,
    ) -> list[str]:
        frame = self.build_feature_frame(df, horizon_name=horizon_name, derivatives_frame=derivatives_frame)
        return infer_feature_columns(frame)

    def build_latest_feature_row(
        self,
        df: pd.DataFrame,
        horizon_name: str | None = None,
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.Series:
        feature_frame = self.build_feature_frame(
            df,
            horizon_name=horizon_name,
            select_grid_only=False,
            derivatives_frame=derivatives_frame,
        )
        feature_frame = feature_frame.loc[feature_frame["is_grid_t0"]].reset_index(drop=True)
        if feature_frame.empty:
            raise ValueError("No feature rows available for the requested horizon.")
        return feature_frame.iloc[-1]

    def build_freqai_feature_dataframe(
        self,
        df: pd.DataFrame,
        horizon_name: str | None = None,
        timestamp_column: str = "date",
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if timestamp_column not in df.columns:
            raise ValueError(f"Expected timestamp column '{timestamp_column}' in dataframe.")

        shared_input = df.rename(columns={timestamp_column: DEFAULT_TIMESTAMP_COLUMN})
        feature_frame = self.build_feature_frame(
            shared_input,
            horizon_name=horizon_name,
            select_grid_only=False,
            derivatives_frame=derivatives_frame,
        ).rename(columns={DEFAULT_TIMESTAMP_COLUMN: timestamp_column})

        feature_columns = [column for column in infer_feature_columns(feature_frame) if column != timestamp_column]
        rename_map = {column: f"%-{column}" for column in feature_columns}
        return feature_frame.rename(columns=rename_map)
