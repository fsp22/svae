import os


def get_params(case_study, optimizer_name, lr, loss_name, model_name, base_directory=os.curdir):
    
    base_directory = os.path.abspath(base_directory)
    print('BASEDIR:', base_directory)

    buildpath = lambda x: os.path.join(base_directory, x)

    hp = dict()
    hp['epochs'] = 20
    hp['batch_size'] = 256
    hp['dataset_with_one_id'] = True  # True if dataset is encoded with 1-based id

    hp['batch_log_interval'] = 1000
    hp['evaluation_iterations'] = 0  # 0 = evaluation after each epoch, >0 = evaluation every each #iterations
    hp['optimizer_name'] = optimizer_name
    hp['lr'] = lr
    hp['loss_name'] = loss_name
    hp['anneal_cap'] = 0.2
    hp['total_anneal_steps'] = 2000000
    hp['input_dir'] = buildpath('datasets/' + case_study + '/pro_sg/')

    tmp_dir = buildpath('datasets/' + case_study + '/tmp/')
    hp['tmp_dir'] = tmp_dir

    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    output_dir = buildpath('results/' + case_study + '/')
    hp['output_dir'] = output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_dir = buildpath('vpr_saved_models/' + case_study + '/')
    hp['model_dir'] = model_dir

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    hp['model_name'] = 'model_' + model_name + '.m'
    hp['model_file'] = os.path.join(model_dir, hp['model_name'])
    hp['param_name'] = 'param_' + model_name + '.json'
    hp['param_file'] = os.path.join(model_dir, hp['param_name'])

    log_dir = buildpath('vpr_saved_logs/' + case_study + '/')
    hp['log_dir'] = log_dir

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    hp['log_name'] = 'log_' + model_name + '.txt'
    hp['log_file'] = os.path.join(log_dir, hp['log_name'])

    return hp
