import torch
from torch.optim import Optimizer


# -------------------------------------------------------------------------------------------------
# Move_Optimizer_To_Device
# -------------------------------------------------------------------------------------------------

def Move_Optimizer_To_Device(optim : torch.optim.Optimizer, device : str) -> None:
    """
    This function moves an optimizer object to a specific device. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    optim : Optimizer
        The optimizer whose device we want to change.

    device : str
        The device we want to move optim onto. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing.
    """

    # Cycle through the optimizer's parameters.
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device);
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device);
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device);
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device);
