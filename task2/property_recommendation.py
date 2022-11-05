from encodings import search_function
import numpy as np
import pandas as pd
import math
import re
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

FEATURES = ['num_beds', 'num_baths', 
            'lat', 'lng', # easy to find properties with the same property name
            'property_type_apartment', 'property_type_bungalow', 'property_type_cluster_house', 'property_type_condo',	'property_type_conservation_house', 'property_type_corner_terrace', 'property_type_executive_condo', 'property_type_hdb',	'property_type_land_only', 'property_type_landed', 'property_type_semi_detached_house', 'property_type_shophouse',	'property_type_terraced_house', 'property_type_townhouse', 'property_type_walk_up', 
            'tenure_nan', 'tenure_freehold',	'tenure_99-110_year', 'tenure_999+year',
            'size_sqft', 'dist_to_nearest_important_mrt_rounded', 'price', 'built_year']
FEATURES_WITH_LISTING_ID = ['listing_id'] + FEATURES
FEATURES_FOR_DISPLAY = ['listing_id', 'property_type', 'num_beds', 'num_baths',
            'lat', 'lng', 'size_sqft', 'dist_to_nearest_important_mrt_rounded', 
            'price', 'built_year', 'tenure']
MAX_RATING = 10
SEARCH_CRITERIA = ['num_beds', 'num_baths', 'price', 'property_type', 'planning_area', 'subzone', 'name_of_nearest_mrt',
'tenure', 'built_year', 'size_sqft', 'per_price']

class SearchCriteriaParser:

    @staticmethod
    def filter_by_criteria(items, user_profile):
        search_criteria = SearchCriteriaParser._parse_search_criteria(user_profile)
        for k, v in search_criteria.items():
            if v:
                if k in ['price', 'size_sqft', 'per_price']: # range
                    items = items[items[k].between(*v)]
                else: # equality
                    if 'nan' in v:
                        items = items[items[k].isin(v) or items[k].isna()]
        return items
    
    @staticmethod
    def _parse_search_criteria(user_profile):
        criteria = user_profile.head() # suffices to take the first row
        criteria_dict = {}
        for k in SEARCH_CRITERIA:
            if isinstance(criteria[k], str):
                if k in ['num_beds', 'num_baths', 'price', 'size_sqft', 'per_price']:
                    criteria_dict[k] = [int(float(x)) for x in criteria[k].split(',')]
                else:
                    criteria_dict[k] = criteria[k].split(',')
            else: # nan is float
                criteria_dict[k] = None
        return criteria_dict

class RecEngine:
    def __init__(self, items):
        self.all_items = items
        self.items = items
        self.scaler = preprocessing.MinMaxScaler()
        self._update_items()
    
    def _update_items(self):
        self.item_features = self.items[FEATURES_WITH_LISTING_ID]
        # scale feature values
        self.item_features_scaled = self.item_features.copy(deep=True)
        self.item_features_scaled[FEATURES] = pd.DataFrame(
            self.scaler.fit_transform(self.item_features_scaled[FEATURES].values),
            columns=FEATURES,
            index=self.item_features_scaled.index
        )
        self.item_features_scaled_numpy = self.item_features_scaled.to_numpy()

    def _get_most_similar_items_listing_id(self, view_history, k, verbose=False):
        pass

    def get_top_recommendations(self, view_history, k, verbose=False):
        listing_ids = self._get_most_similar_items_listing_id(view_history, k, verbose)
        return listing_ids
    
    def get_top_recommendations_based_on_view_history(self, view_history, k, verbose=False):
        self.items = SearchCriteriaParser.filter_by_criteria(self.all_items, view_history)
        self.items.reset_index(inplace=True, drop=True)
        self._update_items()
        listing_ids = self._get_most_similar_items_listing_id(view_history, k, verbose)
        return listing_ids

class PairwiseItemRecEngine(RecEngine):
    def __init__(self, items):
        super().__init__(items)

    def _get_most_similar_items_listing_id(self, view_history, k, verbose=False):
        reference_item_listing_ids = view_history['listing_id'].tolist()
        if verbose:
            print("Getting {} most similar items for {} reference items".format(k, len(reference_item_listing_ids)))
        reference_item_indices = self.item_features_scaled.index[self.item_features_scaled['listing_id'].isin(reference_item_listing_ids)].tolist()
        reference_items = self.item_features_scaled_numpy[reference_item_indices]

        similarities = cosine_similarity(np.delete(self.item_features_scaled_numpy[:,1:], reference_item_indices, axis=0), reference_items[:,1:]) # exclude listing_id and items already in view_history
        
        if verbose:
            print("Similarities: {}".format(similarities))
        similarities_agg = np.mean(similarities, axis=1) # average similarities across reference items for each item, can also use view_time as weight?
        if verbose:
            print("Average similarities: {}".format(similarities_agg))
        most_similar_indices = np.argpartition(similarities_agg, -k)[-k:]
        most_similar_indices = most_similar_indices[np.argsort(similarities_agg[most_similar_indices])]
        if verbose:
            print("Highest averge similarities: {}".format(similarities_agg[most_similar_indices]))
        return self.items.loc[most_similar_indices]['listing_id'].to_list()

class UserItemRecEngine(RecEngine):
    def __init__(self, items):
        super().__init__(items)

    def _get_user_profile_from_view_history(self, view_history):
        # later, more frequent views -> higher importance
        viewed_item_listing_ids = view_history['listing_id'].tolist()
        last_view_time = view_history['view_time'].max()
        viewed_item_indices = self.item_features_scaled.index[self.item_features_scaled['listing_id'].isin(viewed_item_listing_ids)].tolist()
        viewed_items = self.item_features_scaled_numpy[viewed_item_indices][:,1:] # exclude listing_id and items already in view_history
        view_history['view_score'] = view_history['view_time']/last_view_time
        # 1 item is viewed many times, sum or take the latest view time?
        view_scores = view_history.groupby(['listing_id'])['view_score'].sum() # sum
        # view_scores = view_history.groupby(['listing_id'])['view_score'].max() # take latest view time
        n = len(view_scores)
        weighted_sum = np.apply_along_axis(lambda c: np.dot(c, view_scores)/n, 0, viewed_items).reshape(1, -1)
        return weighted_sum

    def _get_most_similar_items_listing_id(self, view_history, k, verbose=False):
        viewed_item_listing_ids = view_history['listing_id']
        viewed_item_indices = self.item_features_scaled.index[self.item_features_scaled['listing_id'].isin(viewed_item_listing_ids)].tolist()
        user_profile = self._get_user_profile_from_view_history(view_history)
        if verbose:
            print("user profile: {}".format(user_profile))
        similarities = cosine_similarity(np.delete(self.item_features_scaled_numpy[:,1:], viewed_item_indices, axis=0), user_profile) # exclude listing_id
        if verbose:
            print("Similarities: {}".format(similarities))
        similarities_agg = similarities.flatten()
        most_similar_indices = np.argpartition(similarities_agg, -k)[-k:]
        most_similar_indices = most_similar_indices[np.argsort(similarities_agg[most_similar_indices])]
        if verbose:
            print("Highest similarities: {}".format(similarities_agg[most_similar_indices]))
        return self.items.loc[most_similar_indices]['listing_id'].to_list()