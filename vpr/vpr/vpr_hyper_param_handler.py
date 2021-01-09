import os
from vpr.common.hyper_param_handler import get_params as sgp


def get_params(case_study, run_id, base_directory=os.curdir):
    loss_name = 'VPRLoss'
    lr = 1e-3
    optimizer_name = 'adam'
    model_name = 'VPR-' + run_id

    hp = sgp(case_study, optimizer_name, lr, loss_name, model_name, base_directory=base_directory)

    hp['batch_size'] = 128
    #hp['evaluation_iterations'] = 100000
    hp['epochs'] = 50

    hp['embed_size'] = 128
    hp['hidden_size_enc'] = 64
    hp['hidden_size_dec'] = 32
    hp['tr_n_negatives'] = 10
    
    # number of positive items for each row in a batch
    hp['row_size'] = 100

    return hp
