"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
RECIPES = {'source_full_match': {'scope': 'full', 'anchor': 0.001, 'loss': 'huber', 'matched': True}, 'source_last_match': {'scope': 'last', 'anchor': 0.001, 'loss': 'huber', 'matched': True}, 'source_full_strong': {'scope': 'full', 'anchor': 0.1, 'loss': 'huber', 'matched': True}, 'source_last_strong': {'scope': 'last', 'anchor': 0.1, 'loss': 'huber', 'matched': True}, 'source_full_mismatch': {'scope': 'full', 'anchor': 0.001, 'loss': 'huber', 'matched': False}, 'source_full_mse': {'scope': 'full', 'anchor': 0.001, 'loss': 'huber_mse', 'matched': True}, 'source_supervised': {'scope': 'full', 'anchor': 0.001, 'loss': 'supervised', 'matched': True}, 'random_matched': {'scope': 'full', 'anchor': 0.001, 'loss': 'huber', 'matched': True}}

def check_features(features):
    if {'soh', 'capacity'} & set(features):
        raise ValueError('Query labels prohibited')
