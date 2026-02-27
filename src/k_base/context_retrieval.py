import json, ast
import argparse, sys, os
from constants import CITIES
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Navigate up to collab-rec-2026/
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/collab-rec-2026/input-data/kb/")

INTEREST_TYPE_MAP = {"see": "see",
                     "do": "do",
                     "drink": "drink at",
                     "eat": "eat at",
                     "go": "go to"}


class ContextRetrieval:
    """
    A class for managing and querying the structured knowledge base.
    """

    def __init__(self):
        """
        Initializes the KnowledgeGraphRetrieval class.
        """

        # Load source database
        self.source_df = pd.read_csv(DATA_DIR + "eu-cities-database_2.csv")

    def get_cities_from_filters(self, filters: dict) -> list:
        df = self.source_df.copy()
        for key, value in filters.items():
            if value is None:
                continue

            if key not in df.columns:
                continue

            df = df[
                df[key]
                .astype(str)
                .str.lower()
                .eq(value.strip().lower())
            ]

        return df.city.unique().tolist()

    def match_city_with_filters(self, city, filters):
        city_df = self.source_df[self.source_df['city'] == city]

        filter_matches = {}
        for key, val in filters.items():
            if "month" in key:
                filter_matches[key] = val

            elif "seasonality" in key:
                low_season_months = city_df['low_season'].notna()
                if len(low_season_months):
                    filter_matches[key] = val

            else:
                try:
                    if val.casefold() in (item.casefold() for item in city_df[key].values.tolist()):
                        filter_matches[key] = val
                except Exception as e:
                    # print(city_df[key].notna().values.tolist())
                    pass

        return filter_matches


def test():
    # Sample config for sub-graph retrieval
    sample_config = {
        "config_id": "c_p_0_pop_low_easy",
        "kg_filters": {
            "popularity": "High",
            "budget": "Low",
            "interests": "Outdoors & Recreation"
        },
    }
    sample_config = {'config_id': 'c_p_1_pop_medium_sustainable', 'p_id': 'p_1',
                     'persona': 'A former DJ at WSUM who is now working as a music journalist',
                     'filters': {'popularity': 'medium', 'interests': 'Food', 'budget': 'low', 'walkability': 'great'}}
    sample_config = {"config_id": "c_p_3_pop_high_sustainable",
                     "filters": {
                         "popularity": "high",
                         "interests": "Shops & Services",
                         "budget": "high",
                         "seasonality": "low"
                     }}
    sample_config = {"config_id": "c_p_143_pop_high_hard",
                     'filters':
                         {'popularity': 'high',
                          'month': 'February', 'budget': 'high',
                          'interests': 'Arts & Entertainment'},
                     "query": "Recommend European cities for a high-budget trip in March with a focus on arts, "
                              "culture, and nightlife, avoiding cities with low seasons.",
                     "cities": ['Zurich', 'Warsaw', 'Vienna', 'Valencia', 'Toulouse', 'Stuttgart', 'Strasbourg',
                                'Stockholm', 'Salzburg', 'Rome', 'Prague', 'Porto', 'Paris', 'Naples', 'Munich',
                                'Milan', 'Madrid', 'Lyon', 'London', 'Ljubljana', 'Lille', 'Helsinki', 'Hamburg',
                                'Geneva', 'Dublin', 'Dresden', 'Cork', 'Copenhagen', 'Cagliari', 'Brussels', 'Bordeaux',
                                'Bologna', 'Berlin', 'Bergen', 'Barcelona', 'Amsterdam']}

    retrieval = ContextRetrieval()

    results = retrieval.match_city_with_filters(city="Zurich", filters=sample_config["filters"])
    results = retrieval.get_matching_cities(sample_config)
    print(results)


if __name__ == "__main__":
    test()
