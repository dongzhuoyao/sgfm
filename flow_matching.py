import copy
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor


# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint

import torch.distributed as dist
import torch.nn.functional as F


_RTOL = 1e-5
_ATOL = 1e-5


@torch.no_grad()
def distributed_sinkhorn(
    out, epsilon, sinkhorn_iterations, world_size
):  # https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py#L353C1-L376C17
    Q = torch.exp(
        out / epsilon
    ).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class Flow_Matching(nn.Module):
    def __init__(self, net, is_rectifiedflow=False, rf_kwargs=None):
        super().__init__()
        self.net = net

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        **kwargs,
    ) -> Tensor:
        if t.numel() == 1:
            t = t.expand(x.size(0))
        res = self.net(x, t, y, **kwargs)
        if isinstance(res, tuple) or isinstance(res, list):
            _pred = res[0]
        else:
            _pred = res
        return _pred

    @torch.no_grad()
    def get_cluster_info(self, x_t, t):
        res = self.net(
            x_t,
            t,
            y=None,  # we don't need y here for self-guidance, y is only used for NMI calculation
            is_stage2=False,
        )

        _, _embed_detached, _softlabel, null_bool_bs, dm_emb = res
        cluster_ids = _softlabel.argmax(1)
        try:
            y_emb = self.net.module.prototypes.weight.data[cluster_ids, :]
        except:
            y_emb = self.net.prototypes.weight.data[cluster_ids, :]

        return y_emb, dm_emb, cluster_ids

    @torch.no_grad()
    def sep_get_mid_embed(self, x_t, t, is_stage2):
        res = self.net(
            x_t,
            t,
            y=None,  # we don't need y here for self-guidance, y is only used for NMI calculation
            is_stage2=is_stage2,
            return_dict=True,
        )
        return res["mid_embed"]

    @torch.no_grad()
    def sep_get_cls_token_mid(self, x_t, t, is_stage2):
        res = self.net(
            x_t,
            t,
            y=None,  # we don't need y here for self-guidance, y is only used for NMI calculation
            is_stage2=is_stage2,
            return_dict=True,
        )
        return res["cls_token_mid"]

    def get_loss_fm(self, x_t, u, t, y, config, **kwargs):
        self.normalize_prototypes()
        pred_vf, _embed_detached, _softlabel, null_bool_bs, dm_emb = self.net(
            x_t,
            t,
            y=y,
            config=config,
            **kwargs,
        )
        vf_loss = (pred_vf - u).square().flatten(1).mean(-1)  # [BS]
        return vf_loss, _softlabel

    def get_loss_sk(
        self,
        _softlabel,
        accelerator,
        temperature=0.1,
        epsilon=0.05,
        sinkhorn_iterations=3,
    ):
        world_size = accelerator.num_processes
        q = distributed_sinkhorn(
            _softlabel.detach().clone(),
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            world_size=world_size,
        )
        # accelerator.wait_for_everyone()
        x = _softlabel / temperature
        _loss = -torch.sum(q * F.log_softmax(x, dim=1), dim=1)
        cluster_ids = q.argmax(dim=1)
        return _loss, cluster_ids

    @torch.no_grad()
    def uvit_get_mid_mean_embed(self, x_t, t):
        mid_embed = self.net(
            x_t,
            t,
            y=None,
        )
        assert len(mid_embed.shape) == 2
        return mid_embed

    def normalize_prototypes(
        self,
    ):
        with torch.no_grad():
            try:
                w = self.net.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.net.module.prototypes.weight.copy_(w)
            except:  # make it work on single-gpu for debugging
                w = self.net.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.net.prototypes.weight.copy_(w)

    def decode(
        self,
        z: Tensor,
        y: Tensor,
        **kwargs,
    ) -> Tensor:
        func = lambda t, x: self(t, x, y=y, **kwargs)
        ode_kwargs = dict(
            method="dopri5",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
        )
        res = odeint(
            func,
            z,
            torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
            **ode_kwargs,
        )
        return res[-1]

    def encode(
        self,
        z: Tensor,
        y: Tensor,
        **kwargs,
    ) -> Tensor:
        func = lambda t, x: self(t, x, y=y, **kwargs)
        ode_kwargs = dict(
            method="dopri5",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
        )
        return odeint(
            func,
            z,
            torch.tensor([1.0, 0.0], device=z.device, dtype=z.dtype),
            **ode_kwargs,
        )[-1]

    def decode_fm_chain(
        self,
        z: Tensor,
        step_size,
        **kwargs,
    ) -> Tensor:
        _, x_ests = self.sample_euler_raw(
            z=z,
            step_num=int(1.0 / step_size),
            return_x_est=True,
            return_x_est_num=8,
            **kwargs,
        )
        return x_ests

    @torch.no_grad()
    def sample_euler_raw(
        self, z, step_num, return_x_est=False, return_x_est_num=None, **kwargs
    ):
        dt = 1.0 / step_num
        traj = []  # to store the trajectory

        z = z.detach().clone()
        bs = len(z)

        est = []

        if return_x_est:
            est_ids = [
                int(i * step_num / return_x_est_num) for i in range(return_x_est_num)
            ]

        traj.append(z.detach().clone())
        for i in range(0, step_num, 1):
            t = torch.ones(bs, device=z.device) * i / step_num
            pred = self.forward(t, z, **kwargs)

            _est_now = z + (1 - i * 1.0 / step_num) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> b t w h c")
            return traj[-1], est
        else:
            return traj[-1]

    def decode_euler(
        self,
        z: Tensor,
        y: Tensor,
        step_size,
        **kwargs,
    ) -> Tensor:
        func = lambda t, x: self(t, x, y=y, **kwargs)
        ode_kwargs = dict(
            method="euler",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
            options=dict(step_size=step_size),
        )
        return odeint(
            func,
            z,
            torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
            **ode_kwargs,
        )[-1]
