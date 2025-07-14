import torch
from functools import partial


def global_cosine_hm_percent(encoder_features, decoder_features, p=0.9, factor=0.0):
    """Global cosine hard mining with percentage.

    Args:
        encoder_features: Source feature maps
        decoder_features: Target feature maps
        p: Percentage for hard mining
        factor: Factor for gradient modification

    Returns:
        Computed loss
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(encoder_features)):
        en_ = encoder_features[item].detach()
        de_ = decoder_features[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(en_.reshape(en_.shape[0], -1), de_.reshape(de_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        de_.register_hook(partial_func)

    loss = loss / len(encoder_features)
    return loss


def modify_grad(x, inds, factor=0.0):
    """Modify gradients based on indices and factor.

    Args:
        x: Input tensor
        inds: Boolean indices indicating which elements to modify
        factor: Factor to multiply the selected gradients by

    Returns:
        Modified tensor
    """
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x
