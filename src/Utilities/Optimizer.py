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



# -------------------------------------------------------------------------------------------------
# reset optimizer
# -------------------------------------------------------------------------------------------------


def Reset_Optimizer(optimizer  : torch.optim.Optimizer) -> None:
    """
    Set the optimizer's m_t and v_t attributes (first and second moments) to zero. After each 
    training round, the momentum from the previous epoch may point us in the wrong direction. 
    Resetting the momentum eliminates this problem.
    """

    # Cycle through the optimizer's parameter groups.
    for group in optimizer.param_groups:

        # Cycle through the parameters in the group.
        for p in group['params']:
            state : dict = optimizer.state[p];

            # If the state is empty, skip this parameter.
            if not state:
                continue;
            
            # zero the biased first moment estimate
            state['exp_avg'].zero_();

            # zero the biased second moment estimate
            state['exp_avg_sq'].zero_();
            
            # if you're using amsgrad:
            if 'max_exp_avg_sq' in state:
                state['max_exp_avg_sq'].zero_();