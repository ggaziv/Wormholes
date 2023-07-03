import torch as ch
import numpy as np
from torch import nn
import dill
import os
from .tools import helpers, constants, invert_dict
from .attacker import AttackerModel


class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

class DummyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=False, pytorch_pretrained=False, add_custom_forward=False):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the 
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel 
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill)
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = ch.nn.DataParallel(model)
    model = model.cuda()

    return model, checkpoint


def get_restricted_imagenet_mapped_model(arch, restricted_imagenet_ds, pytorch_pretrained, resume_path):
    from wormholes.datasets import DATASETS
    from wormholes.tools.folder import ImageFolder
    from collections import OrderedDict
    
    imagenet_ds = DATASETS['imagenet']('')
    folder_ds = ImageFolder(root=f"{restricted_imagenet_ds.data_path}/val", label_mapping=restricted_imagenet_ds.label_mapping)
    class_to_idx_imagenet = ImageFolder(root=f"{restricted_imagenet_ds.data_path}/val", label_mapping=imagenet_ds.label_mapping).class_to_idx
    class_mapper = invert_dict({class_to_idx_imagenet[k]: v for k, v in folder_ds.class_to_idx.items()})
    class_mapper = {k: np.array(v) for k, v in class_mapper.items()}
    class_mapper = OrderedDict(sorted(class_mapper.items()))

    def forward_wrapper(forward):
        def forwarded(*args, **kwargs):
            x = forward(*args, **kwargs)
            if 'with_latent' in kwargs and kwargs['with_latent']:
                return x
            x = ch.stack([x[:, v].max(1).values for v in class_mapper.values()], 1)
            return x
        return forwarded

    net = imagenet_ds.get_model(arch, pytorch_pretrained)
    net.forward = forward_wrapper(net.forward)
    return make_and_restore_model(arch=net, dataset=imagenet_ds, 
                                  pytorch_pretrained=pytorch_pretrained, resume_path=resume_path)
