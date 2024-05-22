import yaml
import os

COMP = '/home/user'

# List of lambda values
lambda_values = [0.001, 0.0001, 1000]
#alpha_low_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
batches = [32, 64, 128, 256, 512]
#epochs = [10,20]
mid_num = [100,200,400]

for i in [100,200,400]:
    with open(COMP + '/configs/config_default.yaml', 'r') as file:
        setup = yaml.safe_load(file)

    setup['architecture']['num_mid'] = i
    setup['architecture']['seed'] = 0
    setup['training']['num_epochs'] = 5
    setup['training']['num_tasks'] = 5
    setup['training']['weight_decay'] = 0
    setup['training']['lambda'] = 0.01
    setup['training']['batch_size'] = 128
    setup['training']['alpha_low'] = 0
    setup['data']['dataset'] = 'MNIST_permuted'
    setup['paths']['dir_exp'] = 'permuted_MNIST/tuning/newnet/'+str(i)


    # Specify the filename for the YAML file based on the lambda value
    filename = f'/config_{i}.yaml'

    WHERE = '/configs/permuted_MNIST/newnet/'
    if not os.path.exists(COMP+WHERE):
        os.makedirs(COMP+WHERE)

    # Write the dictionary to a YAML file
    with open(COMP+WHERE+filename, 'w') as file:
        yaml.dump(setup, file, default_flow_style=False)

    print(f"Config file created: {filename}")