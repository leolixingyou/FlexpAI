import yaml
import numpy as np

def load_mpc_parameters(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    
    # Convert parameters to variables
    for key in params.keys():
        exec(f"{key} = params['{key}']")
    
    # Special handling for matrix parameters
    for matrix in ['R', 'Rd', 'Q', 'Qf']:
        if matrix in params:
            exec(f"{matrix} = np.diag(params['{matrix}'])")
    
    # Convert degrees to radians for MAX_STEER and MAX_DSTEER
    if 'MAX_STEER' in params:
        MAX_STEER = np.deg2rad(params['MAX_STEER'])
    if 'MAX_DSTEER' in params:
        MAX_DSTEER = np.deg2rad(params['MAX_DSTEER'])
    
    return locals()

# Usage
if __name__ == "__main__":
    file_path = '/workspace/src/control/src/controller/config/carla_mpc_parameters.yaml'  # Assume the YAML file is named mpc_parameters.yaml
    variables = load_mpc_parameters(file_path)
    
    # Print out all variables to verify
    for name, value in variables.items():
        if not name.startswith('__') and name != 'params':
            print(f"{name} = {value}")

    # Example of using the variables
    print("\nExamples of using variables:")
    print(f"NX = {variables['NX']}")
    print(f"R = \n{variables['R']}")
    print(f"MAX_STEER = {variables['MAX_STEER']}")
