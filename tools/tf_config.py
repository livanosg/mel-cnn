import json
import os


def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cfg = {'cluster': resolver.cluster_spec().as_dict(),
           'task': {'type': resolver.get_task_info()[0], 'index': resolver.get_task_info()[1]},
           'rpc_layer': resolver.rpc_layer}
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)
