
from wormholes import PROJECT_ROOT
from wormholes.tools import *
from wormholes.attacker import interp
import torch
from torch.nn.functional import cross_entropy
from torchvision import transforms
from wormholes.model_utils import make_and_restore_model, get_restricted_imagenet_mapped_model
from wormholes.attacker import AttackerModel, Attacker
from wormholes.datasets import CIFAR, ImageNet, RestrictedImageNet
from wormholes.tools.folder import ImageFolder
from wormholes.tools.label_maps import CLASS_DICT

 
im_res = 224


def attack_target_centroid(self, 
                           images_source: np.ndarray, 
                           images_target: list, 
                           images_distract: list=None,
                           embedder=None, criterion=None,
                           constraint='2', eps=30, 
                           step_size=0.5, n_iter=200, 
                           interp_alpha=None,
                           do_tqdm=False, miniters=int(223265/100)):
    """Attack towards class centroid
    """
    images_source = pil2tor(images_source)
    images_target = [pil2tor(x) for x in images_target]
    if len(images_target) == 1:
        images_target *= len(images_source)
    
    @torch.no_grad()
    def h_embedder(h, images): # Dynamic embedding for centroid/single image case
        if isinstance(images, list):
            out_list = []
            for x in images:
                out = h(x)
                if isinstance(out, torch.Tensor):
                    out_list.append(out.detach().mean(0))
                else: # namedtuple
                    out_list.append([out_.detach().mean(0) for out_ in out])
            if isinstance(out, torch.Tensor):
                return torch.stack(out_list)
            else:
                return [torch.stack(out) for out in zip(*out_list)]
        else: 
            return h(images)
    if embedder:    
        def custom_loss(_, inp, target_embed): 
            loss_res = criterion(embedder(inp), target_embed)
            return loss_res, None
        # Computer centroids
        target_embed = h_embedder(lambda x: embedder(self.normalize(x)), images_target)
        should_normalize=True
        im_adv = self.forward(images_source, target_embed, 
                                should_normalize=should_normalize,
                                custom_loss=custom_loss,
                                targeted=True, do_tqdm=do_tqdm, miniters=miniters, 
                                constraint=constraint, eps=eps, step_size=step_size, 
                                iterations=n_iter).cpu()
    elif interp_alpha is not None:
        im_adv = interp(images_source, images_target, interp_alpha, constraint, eps).cpu()
    else:
        raise NotImplementedError
    budget_usage = get_norm(im_adv, images_source.cpu()) / eps
    res_list = [im_adv, budget_usage]
    return res_list


def attack_target_class(self,
                        images_source: np.ndarray, 
                        target_class_indices: np.ndarray, 
                        constraint='2', eps=30, 
                        step_size=0.5, n_iter=200, 
                        targeted=True,
                        do_tqdm=False, miniters=int(223265/100)):
    """Attack towards (or against) a specific class
    """
    assert len(images_source) == len(target_class_indices)
    assert target_class_indices.ndim == 1
    images_source = pil2tor(images_source)    
    target_class_indices = torch.from_numpy(target_class_indices)
    im_adv = self.attack(images_source, target_class_indices, 
                         targeted=targeted, do_tqdm=do_tqdm, miniters=miniters, 
                         constraint=constraint, eps=eps, step_size=step_size, 
                         iterations=n_iter)
    budget_usage = get_norm(im_adv, images_source.cpu(), p=float(constraint)) / eps
    return [im_adv, budget_usage]


def attack_composite_target(self,
                            images_source: np.ndarray, 
                            target_logits: np.ndarray, 
                            custom_loss,
                            constraint='2', eps=30, 
                            step_size=0.5, n_iter=200, 
                            do_tqdm=False, miniters=int(223265/100)):
    """Attack towards a specific class
    """
    assert len(images_source) == len(target_logits)
    images_source = pil2tor(images_source)    
    target_logits = torch.from_numpy(target_logits)
    im_adv = self.attack(images_source, target_logits, 
                         custom_loss=custom_loss,
                         targeted=True, do_tqdm=do_tqdm, miniters=miniters, 
                         constraint=constraint, eps=eps, step_size=step_size, 
                         iterations=n_iter)
    budget_usage = get_norm(im_adv, images_source.cpu(), p=float(constraint)) / eps
    return [im_adv, budget_usage]


def contrast_blend(images_source, images_target, interp_alpha, constraint='2', eps=30):
    images_source, images_target = [pil2tor(images) for images in [images_source, images_target]]
    im_adv = interp(images_source, images_target, interp_alpha, constraint, eps).cpu()
    budget_usage = get_norm(im_adv, images_source.cpu(), p=float(constraint)) / eps
    return [im_adv, budget_usage]


def read_image(image_path):
    return array(Image.open(image_path).convert('RGB'))
   

def get_norm(im1, im2, p=2):
    diff = im1 - im2
    return diff.reshape(len(diff), -1).norm(p=p, dim=1)

    
def mse_batchwise(x, y): 
    return ((x - y)**2).mean(1)


class AttackMultiTargetUniform:  
    @staticmethod
    def make_comb_targets(class_dict, k_comb=1, n_sample_combs=None, seed=0):
        """Create composite logit targets for attack
        """
        combs = list(itertools.combinations(class_dict.items(), k_comb))
        targets_dict = {}
        if n_sample_combs is not None:
            combs = [combs[i] for i in np.random.RandomState(seed).permutation(len(combs))[:n_sample_combs]]
        for c in combs:
            compos_class_indices, compos_class_names = [list(x) for x in zip(*c)]
            compos_class_name = '-'.join(compos_class_names)
            target_logits = -np.ones(len(class_dict))
            target_logits[compos_class_indices] = 1
            targets_dict[compos_class_name] = {
                'compos_class_indices': compos_class_indices,
                'compos_class_names': compos_class_names,
                'target_logits': target_logits
                }
        return targets_dict
    
    @staticmethod
    def make_all_target(class_dict):
        class_indices, class_names = [list(x) for x in zip(*class_dict.items())]
        target_logits = -np.ones(len(class_names))
        target_logits[class_indices] = 1
        return {'all': {
            'compos_class_indices': class_indices,
            'compos_class_names': class_names,
            'target_logits': target_logits
            }}
    
    @staticmethod
    def make_none_target(class_dict):
        class_indices, class_names = [list(x) for x in zip(*class_dict.items())]
        target_logits = -np.ones(len(class_names))
        target_logits[class_indices] = -1
        return {'none': {
            'compos_class_indices': class_indices,
            'compos_class_names': class_names,
            'target_logits': target_logits
            }}


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_images(images_paths, n_threads=20):
    """Return the list of frames given by list of absolute paths.
    """
    reader_fn = lambda image_path: np.array(Image.open(image_path).convert('RGB'))
    with Pool(n_threads) as pool:
        res_list = pool.map(reader_fn, images_paths)
    return res_list
    

def pil2tor(pil_img, transform_norm=identity, device=None):
    tor_img = torch.tensor(pil_img, device=device, requires_grad=False)
    if pil_img.ndim == 4:
        tor_img = tor_img.permute(0, 3, 1, 2)
    elif pil_img.ndim == 3:
        tor_img = tor_img.permute(2, 0, 1).unsqueeze(0)
    else:
        raise NotImplementedError
    tor_img = transform_norm(tor_img.type(torch.cuda.FloatTensor) / 255.)
    return tor_img


def tor2pil(tor_img, inv_transform_norm=identity):
    tor_img = torch.round(255. * inv_transform_norm(tor_img).squeeze_(0))
    if tor_img.ndim == 4:
        tor_img = tor_img.permute(0, 2, 3, 1)
    elif tor_img.ndim == 3:
        tor_img = tor_img.permute(1, 2, 0)
    else:
        raise NotImplementedError
    pil_img = tor_img.detach().cpu().numpy().astype(np.uint8)
    return pil_img


def extract_from_string(s, ref_str, regextype='\d+'):
    """Supports recursion
    E.g., 
        >>> extract_from_string('meta_job27.nc', 'job')
        27
    """
    if len(ref_str) == 0:
        return []
    if isinstance(ref_str, list):
        return [extract_from_string(s, ref_str[0], regextype)] + extract_from_string(s, ref_str[1:], regextype)
    else:
        return int(re.findall(r'{}{}'.format(ref_str,regextype) , s)[0].replace(ref_str, ''))


def chained(l):
    return list(itertools.chain(*l))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    pass