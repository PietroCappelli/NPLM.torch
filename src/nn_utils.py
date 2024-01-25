import random
import torch 
import h5py
import os
import numpy as np
import time

def loss_function(true, pred):
    # Compute the loss function
    f = pred[:, 0]
    y = true[:, 0]
    w = true[:, 1]

    loss = torch.sum((1 - y) * w * (torch.exp(f) - 1) - y * w * f)
    return loss


def parametric_loss(true, pred):
    y = true[:, 0]
    w = true[:, 1]
    nu = true[:, 2].unsqueeze(1)

    f = torch.sum(pred * (nu ** torch.arange(1, pred.shape[1] + 1, device=pred.device)), dim=1)
    c = 1. / (1 + torch.exp(f))
    return 100 * torch.mean(y * w * c**2 + (1 - y) * w * (1 - c)**2)


def delta_nu_poly(true, pred):
    nu = true[:, 2].unsqueeze(1)
    f = torch.sum(pred * (nu ** torch.arange(1, pred.shape[1] + 1, device=pred.device)), dim=1)
    return f



class NPLMnetwork(torch.nn.Module):
    """New Physics Learning Machine network.

    The network's architecture, activation function, and weight clipping value can be set during initialization.
    The network can be trained or not by setting the `trainable` parameter during initialization.
    The network's weights can be clipped using the `clip_weights` method.
    The network's gradients can be reset using the `reset_grads` method.
    The network can be saved and loaded using the `save` and `load` methods.
    The network's weights, biases, gradients, and other information can be printed using various methods.
    The network's weights, biases, and gradients can be accessed using the `get_weights`, `get_biases`, and `get_grads` methods.
    """

    def __init__(self, architecture, activation_func, weight_clip_value, trainable=True, device="cpu"):
        """
        Initializes a neural network with the given architecture, activation function, weight clipping value, and trainability.

        Args:
            architecture (list[int]): A list of integers representing the number of neurons in each layer of the network.
            activation_func (callable): The activation function to use in the network.
            weight_clip_value (float): The maximum absolute value of the weights in the network.
            trainable (bool, optional): Whether the network's weights should be updated during training. Defaults to True.
            device (str, optional): The device to use for computation. Defaults to "cpu".
        """

        super(NPLMnetwork, self).__init__()
        
        # Store the weight clipping value and activation function
        self._weight_clip_value = weight_clip_value
        self._activation_func   = activation_func
        
        # Initialize training logs
        self.losses = []
        self.t_list = []
        
        # Set the device for the model
        self.device = torch.device(device)
        
        # Create the layers based on the architecture list
        layers = []
        for i in range(len(architecture) - 1):
            layer = torch.nn.Linear(architecture[i], architecture[i + 1], bias=True)
            
            # Set the layer's weight and bias initialization
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

            # Append the layer to the layers list
            layers.append(layer)
            
            # Set the layer to be trainable or not
            for param in layer.parameters():
                param.requires_grad = trainable
        
        # Store the layers as a module list
        self.layers = torch.nn.ModuleList(layers)
        
        
    def forward(self, x):
        # Forward pass through the network
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation after the last layer
                x = self._activation_func(x)
        
        return x
    
    def clip_weights(self):
        # Apply weight clipping to all layers
        for layer in self.layers:
            with torch.no_grad():
                layer.weight.data.clamp_(-self._weight_clip_value, self._weight_clip_value)
                
                
    def train_model(self, feature, target, loss_function, optimizer, n_epochs, patience=1000):
        """
        Trains the model and updates training logs.

        Args:
            feature (torch.Tensor): Input features.
            target (torch.Tensor): Target outputs.
            loss_function (torch.nn.modules.loss): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            N_EPOCHS (int): Number of training epochs.
        """
        self.train()  # Set the model to training mode

        for epoch in range(1, n_epochs + 1):
            feature, target = feature.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            pred = self(feature)

            # Calculate loss
            loss = loss_function(target, pred)
            self.losses.append(loss.item())

            # Calculate test statistic
            t = -2 * loss.item()
            self.t_list.append(t)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Clip weights
            self.clip_weights()

            # Print progress
            if epoch % patience == 0:
                print(f"Epoch {epoch}/{n_epochs} - Loss: {loss:.4f}")
                
                
    def save_model(self, output_path='model_output'):
        """
        Saves the current model state to a file.

        Args:
            output_path (str, optional): The directory path where the model will be saved. Defaults to 'model_output'.
        """
        # Create the output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the model state
        model_file_name = f'{output_path}/nplm_{self.architecture}_model.pth'
        torch.save(self.state_dict(), model_file_name)
        print(f"Model saved to {model_file_name}")


    def save_history(self, output_path='history_output'):
        """
        Saves the training history to a file.

        Args:
            output_path (str, optional): The directory path where the history will be saved. Defaults to 'history_output'.
        """
        # Create the history output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the training history in HDF5 format
        history_file_name = f'{output_path}/npml_{self.architecture}_history.h5'
        with h5py.File(history_file_name, 'w') as f:
            f.create_dataset('losses', data=np.array(self.losses), compression='gzip')
            f.create_dataset('t_list', data=np.array(self.t_list), compression='gzip')
            print(f"History saved to {history_file_name}")


    def get_losses(self):
        """Returns the stored list of losses."""
        return self.losses

    def get_t_list(self):
        """Returns the stored list of test statistics."""
        return self.t_list
    
                
    def reset_grads(self):
        # Reset the gradients for all layers
        for layer in self.layers:
            layer.weight.grad = None
            layer.bias.grad   = None
        
    def __repr__(self):
        # Print the network's architecture
        return "NPLMnetwork(architecture={}, activation_func={}, weight_clip_value={})".format(
            self.architecture, self._activation_func, self._weight_clip_value
        )

    def __str__(self):
        # Print the network's architecture
        return "NPLMnetwork(architecture={}, activation_func={}, weight_clip_value={})".format(
            self.architecture, self._activation_func, self._weight_clip_value
        )
        
    @property
    def architecture(self):
        # Return the network's architecture
        return [layer.in_features for layer in self.layers] + [self.layers[-1].out_features]
    
    @property
    def trainable(self):
        # Return whether the network is trainable or not
        return self.layers[0].weight.requires_grad
    
    @property
    def weight_clip_value(self):
        # Return the weight clipping value
        return self._weight_clip_value
    
    @weight_clip_value.setter
    def weight_clip_value(self, value):
        # Set the weight clipping value
        self._weight_clip_value = value
        
    @property
    def activation_func(self):
        # Return the activation function
        return self._activation_func
    
    @activation_func.setter
    def activation_func(self, value):
        # Set the activation function
        self._activation_func = value
    
    def save(self, path):
        # Save the network to a file
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        # Load the network from a file
        self.load_state_dict(torch.load(path))
        
    def print_weights(self):
        # Print the network's weights
        for layer in self.layers:
            print("Weights",      layer.weight)
            
    def print_biases(self):
        # Print the network's biases
        for layer in self.layers:
            print("Biases",       layer.bias)
            
    def print_grads(self):
        # Print the network's gradients
        for layer in self.layers:
            print("Weights grad", layer.weight.grad)
            print("Biases grad",  layer.bias.grad)
            
    def print_params(self):
        # Print the network's parameters
        for layer in self.layers:
            print("Weights",      layer.weight)
            print("Biases",       layer.bias)
            print("Weights grad", layer.weight.grad)
            print("Biases grad",  layer.bias.grad)
            
    def print_info(self):
        # Print the network's information
        print("Architecture: {}".format(self.architecture))
        print("Weight clipping value: {}".format(self._weight_clip_value))
        print("Activation function: {}".format(self._activation_func))
        print("Trainable: {}".format(self.trainable))
        
    def get_weights(self):
        # Return the network's weights
        return [layer.weight for layer in self.layers]
    
    def get_biases(self):
        # Return the network's biases
        return [layer.bias for layer in self.layers]
    
    def get_grads(self):
        # Return the network's gradients
        return [layer.weight.grad for layer in self.layers] + [layer.bias.grad for layer in self.layers]
        

        
        

class ParametricNet(torch.nn.Module):
    """
    Parametric Neural Network.

    Attributes:
        device (str): The device on which the model operates (e.g., 'cpu', 'cuda').
        architecture (list[int]): A list specifying the architecture of each NPLMnetwork.
        poly_degree (int): The degree of the polynomial used in the parametric model.
        training_history (dict): Stores the training history including loss values.
        model_state (dict): Contains the state of the model.
        train_coeffs (list[bool]): Indicates whether each polynomial term is trainable.
        coeffs (torch.nn.ModuleList): A list of NPLMnetwork instances, each representing a term in the polynomial model.
    """
    def __init__(
        self, 
        architecture  = [1, 10, 1], 
        activation    = torch.nn.Sigmoid(), 
        poly_degree   = 1,
        initial_model = None, 
        train_coeffs  = True,
        device        = "cpu"
        ):
        """
        Initializes the ParametricNet with a specified architecture, activation function, polynomial degree, 
        initial model state, training coefficients, and device.

        Args:
            architecture (list[int], optional): A list specifying the architecture of each NPLMnetwork. Defaults to [1, 10, 1].
            activation (callable, optional): The activation function to be used in each NPLMnetwork. Defaults to torch.nn.Sigmoid().
            poly_degree (int, optional): The degree of the polynomial model. Defaults to 1.
            initial_model (str, optional): Path to a pre-trained model to initialize the ParametricNet. Defaults to None.
            train_coeffs (bool or list[bool], optional): Indicates if the coefficients of the polynomial are trainable. Defaults to True.
            device (str, optional): The device to use for computation. Defaults to "cpu".
        """
        super(ParametricNet, self).__init__()
        
        self.device = device
        # self.to(self.device)  # Move the model to the specified device
        
        self.architecture = architecture  # Architecture of each NPLMnetwork
        
        self.poly_degree = poly_degree  # Polynomial degree of the model
        self.training_history = None  # Placeholder for training history
        self.model_state = None  # Placeholder for model state

        # Handling the train_coeffs argument to ensure it's a list
        if not isinstance(train_coeffs, list):
            self.train_coeffs = [train_coeffs for _ in range(self.poly_degree)]
        else:
            self.train_coeffs = train_coeffs

        # Creating the polynomial coefficients as instances of NPLMnetwork
        self.coeffs = torch.nn.ModuleList([
            NPLMnetwork(architecture, activation_func=activation, trainable=self.train_coeffs[i], weight_clip_value=None)
            for i in range(self.poly_degree)
        ])

        # Load initial model if provided
        if initial_model is not None:
            self.load_state_dict(torch.load(initial_model))
            
            
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        out = []
        for coeff in self.coeffs:
            out.append(coeff(x))
        
        # Handling output based on the polynomial degree
        if self.poly_degree == 1:
            return out[0]
        else:
            return torch.cat(out, dim=1)


    def clip_weights(self, wc):
        """
        Applies weight clipping to all layers in the polynomial network.

        Args:
            wc (float): The weight clipping value.
        """
        for module in self.coeffs:
            for layer in module.children():
                if hasattr(layer, 'weight'):
                    with torch.no_grad():
                        layer.weight.data.clamp_(-wc, wc)


    def train_model(self, feature_train, target_train, feature_val, target_val, total_epochs, optimizer, wc, patience, gather_after=1, batch_fraction=0.3):
        """
        Trains the model using the provided training and validation data.

        Args:
            feature_train (torch.Tensor): The features of the training data.
            target_train (torch.Tensor): The targets of the training data.
            feature_val (torch.Tensor): The features of the validation data.
            target_val (torch.Tensor): The targets of the validation data.
            total_epochs (int): The total number of epochs for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            wc (float): Weight clipping value to be used during training.
            patience (int): Number of epochs to wait before stopping if no improvement in validation loss.
            gather_after (int, optional): Number of iterations after which to apply the optimizer step. Defaults to 1.
            batch_fraction (float, optional): Fraction of the training data to use in each batch. Defaults to 0.3.
        """
        self.train()  # Set the model to training mode

        pars_total = []  # List to store parameters of each epoch
        loss_total = []  # List to store training loss of each epoch
        loss_val_total = []  # List to store validation loss of each epoch

        for epoch in range(int(total_epochs / patience)):
            running_loss = 0.0
            for p in range(patience):
                # Batch selection for mini-batch training
                if batch_fraction < 1:
                    indices = random.sample(range(len(feature_train)), int(batch_fraction * len(feature_train)))
                    feature_tmp, target_tmp = feature_train[indices], target_train[indices]
                else:
                    feature_tmp, target_tmp = feature_train, target_train
                
                feature_tmp, target_tmp = feature_tmp.to(self.device), target_tmp.to(self.device)

                # Forward pass
                optimizer.zero_grad()  # Reset gradients
                output = self(feature_tmp)  # Compute model output
                loss = parametric_loss(target_tmp, output)  # Compute loss

                # Backward pass and optimize
                loss.backward()  # Backpropagation
                if (p + 1) % gather_after == 0:
                    optimizer.step()  # Update weights
                    optimizer.zero_grad()  # Reset gradients after updating

                running_loss += loss.item()
                
                # Apply weight clipping
                self.clip_weights(wc)

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                pred_val = self(feature_val.to(self.device))
                loss_val = parametric_loss(target_val.to(self.device), pred_val)

            # Logging for monitoring
            pars_total.append([param.clone() for param in self.parameters()])
            loss_total.append(running_loss / patience)
            loss_val_total.append(loss_val.item())

            print(f'Epoch: {(epoch + 1) * patience}, Loss: {loss_total[-1]:.4f}, Val Loss: {loss_val_total[-1]:.4f}')

        # Store training history and update model state
        self.training_history = {
            "loss": loss_total,
            "loss_val": loss_val_total
        }
        self.model_state = self.state_dict()
            
        
    def save_model(self, output_path='model_output'):
        """
        Saves the current model state to a file.

        Args:
            architecture (str): A string identifier for the architecture, used in naming the saved file.
            output_path (str, optional): The directory path where the model will be saved. Defaults to 'model_output'.
        """
        # Create the output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the model state
        torch.save(self.model_state, f'{output_path}/parametric_deg{self.poly_degree}_{self.architecture}_final.pth')


    def save_history(self, output_path='history_output'):
        """
        Saves the training history to a file.

        Args:
            architecture (str): A string identifier for the architecture, used in naming the saved file.
            output_path (str, optional): The directory path where the history will be saved. Defaults to 'history_output'.
        """
        # Create the history output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the training history in HDF5 format
        with h5py.File(f'{output_path}/parametric_deg{self.poly_degree}_{self.architecture}_history.h5', 'w') as f:
            f.create_dataset('pars', data=np.array(self.training_history["pars"]), compression='gzip')
            f.create_dataset('loss', data=np.array(self.training_history["loss"]), compression='gzip')
            f.create_dataset('loss_val', data=np.array(self.training_history["loss_val"]), compression='gzip')
