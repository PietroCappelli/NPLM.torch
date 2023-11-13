import torch 


def loss_function(true, pred):
    # Compute the loss function
    f = pred[:, 0]
    y = true[:, 0]
    w = true[:, 1]

    loss = torch.sum((1 - y) * w * (torch.exp(f) - 1) - y * w * f)
    return loss



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
        
