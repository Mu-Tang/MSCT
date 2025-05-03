# A wrapper to add max_norm weight correction capability to PyTorch SGD optimizer.

import torch


class MNSGD(torch.optim.SGD):
    def step(self):
        super().step()
        with torch.no_grad():
            # apply the max_norm weight correction per group of parameters
            for group in self.param_groups:
                # rescale iff group has specified max_norm
                if group.get('max_norm'):
                    for tensor in group['params']:
                        if tensor.dim() > 1:
                            torch.renorm(tensor, p=2, dim=0, maxnorm=group['max_norm'], out=tensor)


def check_norms(model, max_norm, tol=1e-4):
    with torch.no_grad():
        norm_status = 'Norms OK'
        max_vector_norm = 0
        for i, tensor in enumerate(model.hidden.parameters()):
            if tensor.dim() > 1:
                vector_norm = torch.max(
                    torch.linalg.vector_norm(tensor, ord=2,
                                             dim=1, keepdim=True)
                )
                if vector_norm - tol > max_norm:
                    norm_status = 'Norms NOK'
                if vector_norm > max_vector_norm:
                    max_vector_norm = vector_norm
        print(norm_status + f", max vector norm is {max_vector_norm}\n")
