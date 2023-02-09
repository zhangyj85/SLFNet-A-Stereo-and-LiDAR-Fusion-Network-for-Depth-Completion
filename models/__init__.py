from .SLFNet import SLFNet

MODELS = {
    "SLFNet": SLFNet
}

def get_model(name: str):
    """Get backbone given the name"""
    if name not in MODELS.keys():
        raise ValueError(
            f"Model {name} not in model list. Valid models are {MODELS.keys()}"
        )
    return MODELS[name]