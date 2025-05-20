import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from llava.utils import rank0_print
def gather_features(
        query_features,
        target_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_query_features = hvd.allgather(query_features)
            all_target_features = hvd.allgather(target_features)
        else:
            with torch.no_grad():
                all_query_features = hvd.allgather(query_features)
                all_target_features = hvd.allgather(target_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_query_features = list(all_query_features.chunk(world_size, dim=0))
                gathered_target_features = list(all_target_features.chunk(world_size, dim=0))
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
                all_query_features = torch.cat(gathered_query_features, dim=0)
                all_target_features = torch.cat(gathered_target_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_query_features = torch.cat(torch.distributed.nn.all_gather(query_features), dim=0)
            all_target_features = torch.cat(torch.distributed.nn.all_gather(target_features), dim=0)
        else:
            gathered_query_features = [torch.zeros_like(query_features) for _ in range(world_size)]
            gathered_target_features = [torch.zeros_like(target_features) for _ in range(world_size)]
            dist.all_gather(gathered_query_features, query_features)
            dist.all_gather(gathered_target_features, target_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
            all_query_features = torch.cat(gathered_query_features, dim=0)
            all_target_features = torch.cat(gathered_target_features, dim=0)

    return all_query_features, all_target_features

        
class WeightedClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, query_features, target_features, logit_scale):
        if self.world_size > 1:
            all_query_features, all_target_features = gather_features(
                query_features, target_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_query = logit_scale * query_features @ all_target_features.T
                logits_per_target = logit_scale * target_features @ all_query_features.T
            else:
                logits_per_query = logit_scale * all_query_features @ all_target_features.T
                logits_per_target = logits_per_query.T
        else:
            logits_per_query = logit_scale * query_features @ target_features.T
            logits_per_target = logit_scale * target_features @ query_features.T
        
        return logits_per_query, logits_per_target

    def forward(self, query_features, target_features, logit_scale, alpha=0.5, output_dict=False):
        device = query_features.device
        logits_per_query, logits_per_target = self.get_logits(query_features, target_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_query.shape[0])

        # reward model same as the policy model
        logits_per_query_weight = torch.log(torch.exp(alpha * logits_per_query.detach() / logit_scale))
        logits_per_target_weight = torch.log(torch.exp(alpha * logits_per_target.detach() / logit_scale))

        logits_per_query_weight.scatter_(1, labels.unsqueeze(1), 0)
        logits_per_target_weight.scatter_(1, labels.unsqueeze(1), 0)

        new_logits_per_query = logits_per_query + logits_per_query_weight
        new_logits_per_target = logits_per_target + logits_per_target_weight

        total_loss = (
            F.cross_entropy(new_logits_per_query, labels) +
            F.cross_entropy(new_logits_per_target, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss