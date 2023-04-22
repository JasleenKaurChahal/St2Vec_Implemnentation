from Trainer import STsim_Trainer
import torch
import os

if __name__ == '__main__':
    print(torch.__version__)          # Print the PyTorch version being used
    print(torch.cuda.device_count())  # Print the number of available GPUs
    print(torch.cuda.is_available())  # Print whether a GPU is available

    # Create an instance of the STsim_Trainer class
    STsim = STsim_Trainer()

    load_model_name = None            # Set the name of the model checkpoint to be loaded to None
    load_optimizer_name = None        # Set the name of the optimizer checkpoint to be loaded to None

    # Call the ST_train method of the STsim_Trainer instance to train the model
    # If load_model_name and load_optimizer_name are not None, the corresponding checkpoints will be loaded
    STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)

    # Call the ST_eval method of the STsim_Trainer instance to evaluate the trained model
    # If load_model_name is not None, the corresponding checkpoint will be loaded
    # This line is currently commented out, so the evaluation will not be performed
    # STsim.ST_eval(load_model=load_model_name)
