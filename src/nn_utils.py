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

    This class defines a neural network for language modeling using the NPLM architecture.
    It consists of multiple linear layers with an activation function applied after each layer.
    The network's architecture, activation function, and weight clipping value can be set during initialization.
    The network can be trained or not by setting the `trainable` parameter during initialization.
    The network's weights can be clipped using the `clip_weights` method.
    The network's gradients can be reset using the `reset_grads` method.
    The network can be saved and loaded using the `save` and `load` methods.
    The network's weights, biases, gradients, and other information can be printed using various methods.
    The network's weights, biases, and gradients can be accessed using the `get_weights`, `get_biases`, and `get_grads` methods.
    """

    def __init__(self, architecture, activation_func, weight_clip_value, trainable=True):
        """
        Initializes a neural network with the given architecture, activation function, weight clipping value, and trainability.

        Args:
            architecture (list[int]): A list of integers representing the number of neurons in each layer of the network.
            activation_func (callable): The activation function to use in the network.
            weight_clip_value (float): The maximum absolute value of the weights in the network.
            trainable (bool, optional): Whether the network's weights should be updated during training. Defaults to True.
        """

        super(NPLMnetwork, self).__init__()
        
        # Store the weight clipping value and activation function
        self._weight_clip_value = weight_clip_value
        self._activation_func   = activation_func
        
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
        if self._weight_clip_value is None:
            return
        # Apply weight clipping to all layers
        for layer in self.layers:
            with torch.no_grad():
                layer.weight.data.clamp_(-self._weight_clip_value, self._weight_clip_value)
                
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
    def __init__(
        self, 
        architecture  = [1, 10, 1], 
        activation    = torch.nn.Sigmoid(), 
        poly_degree   = 1,
        initial_model = None, 
        train_coeffs  = True,
        device        = "cpu"
        ):
        super(ParametricNet, self).__init__()
        
        # Store the device on which the model is
        self.device = device
        self.to(self.device)
        
        self.poly_degree = poly_degree
        self.training_history = None
        self.model_state = None
            

        if not isinstance(train_coeffs, list):
            self.train_coeffs = [train_coeffs for _ in range(self.poly_degree)]
        else:
            self.train_coeffs = train_coeffs
        
        self.coeffs = torch.nn.ModuleList([
            NPLMnetwork(architecture, activation_func=activation, trainable=self.train_coeffs[i], weight_clip_value=None)
            for i in range(self.poly_degree)
        ])

        if initial_model is not None:
            # Load the initial model weights
            self.load_state_dict(torch.load(initial_model))

    def forward(self, x):
        out = []
        for coeff in self.coeffs:
            out.append(coeff(x))
        
        if self.poly_degree == 1:
            return out[0]
        else:
            return torch.cat(out, dim=1)

    def clip_weights(self, wc):
        # Apply weight clipping to all layers
        for module in self.coeffs:
            for layer in module.children():
                if hasattr(layer, 'weight'):
                    with torch.no_grad():
                        layer.weight.data.clamp_(-wc, wc)


    def train_model(self, feature_train, target_train, feature_val, target_val, total_epochs, optimizer, wc, patience, gather_after=1, batch_fraction=0.3):
        self.train()

        pars_total = []
        loss_total = []
        loss_val_total = []

        for epoch in range(int(total_epochs/patience)):
            running_loss = 0.0
            for p in range(patience):
                # Batch selection
                if batch_fraction < 1:
                    indices = random.sample(range(len(feature_train)), int(batch_fraction * len(feature_train)))
                    feature_tmp, target_tmp = feature_train[indices], target_train[indices]
                else:
                    feature_tmp, target_tmp = feature_train, target_train
                
                feature_tmp, target_tmp = feature_tmp.to(self.device), target_tmp.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self(feature_tmp)
                loss = parametric_loss(target_tmp, output)

                # Backward pass and optimize
                loss.backward()
                if (p + 1) % gather_after == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()
                
                # Apply weight clipping
                self.clip_weights(wc)

            # Validation
            self.eval()
            with torch.no_grad():
                pred_val = self(feature_val.to(self.device))
                loss_val = parametric_loss(target_val.to(self.device), pred_val)

            # Logging
            pars_total.append([param.clone() for param in self.parameters()])
            loss_total.append(running_loss / patience)
            loss_val_total.append(loss_val.item())

            print(f'epoch: {(epoch + 1)*patience}, loss: {loss_total[-1]}, val_loss: {loss_val_total[-1]}')

        # Store the training history and update model state here
        self.training_history = {
            "loss": loss_total,
            "loss_val": loss_val_total
        }
        self.model_state = self.state_dict()
            
            
    # def train_model(self, feature_train, target_train, feature_val, target_val, total_epochs, optimizer, wc, patience, gather_after=1, batch_fraction=0.3):
    #     self.train()  # Set the model to training mode

    #     training_start = time.time()

    #     # Tracking loss in tensors to avoid CPU-GPU transfer
    #     loss_total = torch.tensor([], device=self.device)
    #     loss_val_total = torch.tensor([], device=self.device)

    #     for epoch in range(total_epochs // patience):
    #         epoch_start = time.time()

    #         # Training Phase
    #         running_loss = 0.0
    #         optimizer.zero_grad()  # Clear gradients at the start of each epoch

    #         for step in range(patience):
    #             # Custom batch sampling
    #             indices = torch.randperm(len(feature_train))[:int(len(feature_train) * batch_fraction)]
    #             feature_batch = feature_train[indices].to(self.device)
    #             target_batch = target_train[indices].to(self.device)

    #             output = self(feature_batch)
    #             loss = parametric_loss(target_batch, output)
    #             loss.backward()

    #             # Gradient accumulation
    #             if (step + 1) % gather_after == 0 or step == patience - 1:
    #                 optimizer.step()
    #                 optimizer.zero_grad()  # Clear gradients after optimization step
    #                 self.clip_weights(wc)

    #             running_loss += loss.item()

    #         # Average training loss for the epoch
    #         epoch_loss = running_loss / patience
    #         loss_total = torch.cat((loss_total, torch.tensor([epoch_loss], device=self.device)))

    #         # Validation Phase
    #         val_loss_sum = 0.0
    #         val_samples = 0
    #         self.eval()
    #         with torch.no_grad():
    #             for idx in range(0, len(feature_val), batch_fraction * len(feature_val)):
    #                 val_feature_batch = feature_val[idx:idx + int(batch_fraction * len(feature_val))].to(self.device)
    #                 val_target_batch = target_val[idx:idx + int(batch_fraction * len(feature_val))].to(self.device)

    #                 val_output = self(val_feature_batch)
    #                 val_loss = parametric_loss(val_target_batch, val_output)

    #                 val_loss_sum += val_loss.item() * val_feature_batch.size(0)
    #                 val_samples += val_feature_batch.size(0)

    #         avg_val_loss = val_loss_sum / val_samples
    #         loss_val_total = torch.cat((loss_val_total, torch.tensor([avg_val_loss], device=self.device)))
            
    #         epoch_end = time.time()

    #         if (epoch + 1) % patience == 0:
    #             print(f'Epoch: {epoch + 1}/{total_epochs} ({epoch_end - epoch_start:.4f}s) | Training loss: {epoch_loss:.4f} | Validation loss: {avg_val_loss:.4f}')

    #     training_end = time.time()
    #     print(f"Training time: {training_end - training_start:.2f}s")

    #     # Store the training history and update model state here
    #     self.training_history = {
    #         "loss": loss_total.cpu().numpy(),
    #         "loss_val": loss_val_total.cpu().numpy()
    #     }
    #     self.model_state = self.state_dict()

        
    def save_model(self, architecture, output_path='model_output'):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.model_state, f'{output_path}/parametric_{architecture}_final.pth')


    def save_history(self, architecture, output_path='history_output'):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with h5py.File(f'{output_path}/parametric_{architecture}_history.h5', 'w') as f:
            f.create_dataset('pars', data=np.array(self.training_history["pars"]), compression='gzip')
            f.create_dataset('loss', data=np.array(self.training_history["loss"]), compression='gzip')
            f.create_dataset('loss_val', data=np.array(self.training_history["loss_val"]), compression='gzip')