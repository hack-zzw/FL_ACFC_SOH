"""Load dataset-specific adapted clients; never accept query labels."""
import pandas as pd
import torch
from configs.schema import 特征列
from models.external.source import PROJECT, digest, load_scaler, read_config
from models.external.initialization import initialized_model
from models.external.adaptation import METHODS, SupportTransform, infer
from models.external.predict_base import predict_saved_client

def resolve_source_artifacts(value):
    return PROJECT / 'checkpoints/source'

def predict_dataset_client(checkpoint, features, device):
    if 'soh' in features or 'capacity' in features:
        raise ValueError('Only query features are accepted, not SOH or capacity')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    artifacts = resolve_source_artifacts(saved['source_artifacts'])
    seed = saved['metadata']['seed']
    if digest(artifacts / f'source_seed_{seed}.pt') != saved['source_checkpoint_sha256']:
        raise ValueError('Source checkpoint checksum mismatch')
    if digest(artifacts / 'source_scaler.json') != saved['source_scaler_sha256']:
        raise ValueError('Scaler checksum mismatch')
    method = saved['metadata']['method']
    if method == 'nasa_residual':
        from models.external.oxford import nasa_ridge_predict
        config = read_config(PROJECT / 'configs/best_no_pinn_experiment_config.json')
        model = initialized_model(config, saved['state'], device, seed)
        scaler = load_scaler(artifacts / 'source_scaler.json')
        state = saved['adaptation']
        x = SupportTransform(scaler, pd.DataFrame(), state['candidate']['transform']).apply(features)
        prior, _ = infer(model, x, device)
        return state['candidate']['prior_weight'] * prior + nasa_ridge_predict(features, x, state['fit'])
    if method in METHODS:
        return predict_saved_client(checkpoint, features, artifacts, device)
    if method not in {'standalone_feature_ridge', 'standalone_embedding_ridge', 'zero_shot'}:
        raise ValueError('This entry supports ACfC and fitted ridge models; anchor/trend are diagnostic controls')
    scaler = load_scaler(artifacts / 'source_scaler.json')
    state = saved['adaptation']
    transform = SupportTransform(scaler, pd.DataFrame(), state['candidate']['transform'])
    transform.mask = [特征列.index(name) for name in state.get('masked_features', [])]
    x = transform.apply(features)
    if method in {'standalone_embedding_ridge', 'zero_shot'}:
        config = read_config(PROJECT / 'configs/best_no_pinn_experiment_config.json')
        model = initialized_model(config, saved['state'], device, seed)
        prediction, x = infer(model, x, device)
        if method == 'zero_shot':
            return prediction
    from models.external.oxford import nasa_ridge_predict
    return nasa_ridge_predict(features, x, state['fit'])
