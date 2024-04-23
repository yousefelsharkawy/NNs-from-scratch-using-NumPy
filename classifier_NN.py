# Description: This file contains the implementation of a binary classifier using a neural network classifier

# import libraries
import numpy as np

# create a binary classifier using a neural network classifier
class ClassifierNN:


    ## initialization function of the model 
    def __init__(self, layer_dims,activations,initialization_method = "he"):
        # layer_dims is a list containing the number of neurons in each layer starting from the input layer
        self.layer_dims = layer_dims
        # activations is a list containing the activation functions of each layer starting from the first hidden layer, that is why the length of the list is equal to the number of hidden layers
        assert (len(layer_dims) - 1) == len(activations), "The number of hidden layers and activations must be the same"
        self.activations = activations
        self.num_layers = len(layer_dims)
        self.parameters = self.initialize_parameters(initialization_method)
        self.costs = []
        

    ## initialize the parameters of the model
    def initialize_parameters(self, initialization_method):
        np.random.seed(3)
        #assert self.layer_dims[-1] == 1, "The output layer must have 1 neuron"
        parameters = {}
        for l in range(1, self.num_layers): # the index of the last layer is num_layers - 1 that's why num_layers is not included
            #print(l)
            if initialization_method == "random":
                # in random initialization we sample from a normal distribution with mean 0 and variance 1 and then multiply by 0.01 to make the variance 0.01 -small weights-
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            elif initialization_method == "he":
                # in he initialization we sample from a normal distribution with mean 0 and variance 2/n[l-1] where n[l-1] is the input to the layer 
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        return parameters
    
    
    ## define the forward propagation
    def forward_propagation(self, X, keep_prob = None, parameters = None):
        # we do this to allow the user to pass parameters to the forward propagation function anywhere outside, if he didn't pass any parameters, we will use the parameters of the model
        if parameters is None:
            parameters = self.parameters
        
        cashes = {'A0': X} # this is not efficient with larger datasets
        A = X # first activation is the input layer 
        
        # dropout on the input layer
        if keep_prob is not None:
            # D will be of the same size as A, and will contain numbers between 0 and 1
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob[0]).astype(int) # convert the matrix to 0s and 1s, with 0s having a probability of keep_prob[0] and 1s having a probability of 1 - keep_prob[0]
            # shut down some neurons of A 
            A = np.multiply(A, D)
            # divide A by keep_prob[0] to keep the expected value of A the same as before dropping out some of the neurons
            A /= keep_prob[0]
            # cache the D matrix to use it in the backward propagation by multiplying them with the upstream derivatives dA
            cashes['D0'] = D

        
        for l in range(1, self.num_layers): # will count from 1 to num_layers - 1 
            
            ## Z calculation for the current layer Z = W.A + b
            Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]

            ## apply the activation function on Z to get A
            # we check the activation function of the current layer - we use l-1 because we start activations with index 0 and layers with index 1 - as we have an index mismatch between the two lists
            if self.activations[l-1] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[l-1] == 'relu':
                A = self.relu(Z)
            elif self.activations[l-1] == 'tanh':
                A = self.tanh(Z)
            elif self.activations[l-1] == 'softmax':
                A = self.softmax(Z)


            ## apply dropout on A if keep_prob is not None
            if keep_prob != None:
                print("from inside dropout")
                if l != self.num_layers - 1: # We don't apply dropout on the output layer
                    # print("l = ", l)
                    # print("A shape = ", A.shape)
                    D = np.random.rand(A.shape[0], A.shape[1]) # Create a matrix of random numbers with the same shape as A with numbers between 0 and 1
                    D = (D < keep_prob[l]).astype(int) # Convert the matrix to 0s and 1s
                    A = np.multiply(A, D) # Multiply by the activations to shut down those that correspond to 0
                    A /= keep_prob[l] # Divide by the keep probability to keep the expected value of the activations the same as before dropping out some of them
                    cashes['D' + str(l)] = D # save the D matrix to use it in the backward propagation by multiplying them with the upstream derivatives dA
                    # print("D shape = ", cashes['D' + str(l)].shape)
            
            ## cache the Z and A of the current layer to use them in the backward propagation
            cashes['Z' + str(l)] = Z
            cashes['A' + str(l)] = A
        return A, cashes
    
    
    ## forward propagation helper functions

    # define the activation functions
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z)) # 1 / (1 + e^-Z)

    def relu(self, Z):
        return np.maximum(0, Z) # max(0, Z)
    
    def tanh(self, Z):
        return np.tanh(Z) # tanh(Z)
    
    def softmax(self, Z):
        # we sum over the rows - we sum the different classes outputs for the same example then we apply e^z and divide each element by the sum, we do that for each and every example
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
    
    # define the cost function
    def compute_cost(self, A, Y, lambd = 0, loss = 'BinaryCrossEntropy'):
        m = Y.shape[1]
        ## Handle the case when A is 0 or 1, as log(0) is undefined
        A[A == 0] = 1e-10
        A[A == 1] = 1 - 1e-10

        ## calculate the cost function which will be the average of the losses of all the examples in the batch 
        if loss == 'BinaryCrossEntropy':
            cost = (-1/ m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))) # Here Y is a vector of shape (1, M) and values of 0 or 1, and A is a vector of shape (1, M) and values between 0 and 1
        elif loss == 'CategoricalCrossEntropy':
            cost = (-1/ m) * (np.sum(Y*np.log(A))) # Here Y is a matrix of shape (C, M) and values of one-hot vectors, and A is a matrix of shape (C, M) and values between 0 and 1
        elif loss == "SpareCategoricalCrossEntropy":
            # Selected_probs is a vector of shape (1, M) containing the probabilities of the selected classes for each example
            selected_probs = A[Y, np.arange(m)] # We use the value of Y as an index to select the probabilities of the selected classes for each example and np.arange(m) is used to index all the examples in order 
            cost = (-1/ m) * (np.sum(np.log(selected_probs)))
    
        ## add the L2 regularization cost
        if lambd != 0:
            L2_regularization_cost = 0
            for l in range(1, self.num_layers):
                L2_regularization_cost += np.sum(np.square(self.parameters['W' + str(l)]))
            L2_regularization_cost *= (lambd/(2*m))
            cost += L2_regularization_cost
        return cost
    
    
    
    
    ##  backward propagation
    def backward_propagation(self, A, Y, cashes, lambd = 0, keep_prob = None, loss = 'BinaryCrossEntropy'):
        grads = {}
        m = Y.shape[1]
        
        ## calculation of the first upstream derivative dA - which is the derivative of the cost function with respect to Y_hat -
        if loss == 'BinaryCrossEntropy':
            dA = - (1 / m) * (np.divide(Y, A) - np.divide(1 - Y, 1 - A)) # this is the derivative of the cost function with respect to A
        elif loss == 'CategoricalCrossEntropy': 
            # Y is of shape (C, M) , written as 1-hot matrix, A is of shape (C, M) 
            # the cost function is -1/M * sum(Y * log(A)) , so the derivative of the cost function with respect to A is -1/M * Y/A
            dA = - (1 / m) * (np.divide(Y, A)) 
        elif loss == "SpareCategoricalCrossEntropy":
            Y = np.eye(A.shape[0])[Y.reshape(-1)].T # Y is a vector of shape (1, M) and we want to convert it to a matrix of shape (C, M) where C is the number of classes
            dA = - (1 / m) * (np.divide(Y, A))

        for l in reversed(range(1, self.num_layers)):
            #print("l = ", l)
            ## dZ calculation from dA (by multiplying dA by dA_dZ -which is the derivative of the activation function with respect to Z-) 
            if self.activations[l-1] == 'sigmoid':
                dZ = self.sigmoid_backward(dA, cashes['A' + str(l)])
            elif self.activations[l-1] == 'relu':
                dZ = self.relu_backward(dA, cashes['Z' + str(l)])
            elif self.activations[l-1] == 'tanh':
                dZ = self.tanh_backward(dA, cashes['A' + str(l)])
            elif self.activations[l-1] == 'softmax':
                dZ = self.softmax_backward(dA, cashes['A' + str(l)])
            # since Z and A are of the same shape, dZ and dA will be of the same shape as well
            assert dZ.shape == dA.shape
            
            ## dW and db calculation from dZ (it will be different depending on whether we are using L2 regularization or not)
            if lambd != 0:
                # if we are applying L2 regularization, the equation becomes dW = (1/m) * (dZ * A_prev.T) + (lambd/m) * W
                dW = (1/m) * np.dot(dZ, cashes['A' + str(l-1)].T) + (lambd/m) * self.parameters['W' + str(l)]
            else:
                # if we are not applying L2 regularization, the equation is dW = (1/m) * (dZ * A_prev.T)
                dW = (1/m) * np.dot(dZ, cashes['A' + str(l-1)].T)
            # db = (1/m) * (sum(dZ))
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            ## The next upstream calculation dA = W.T * dZ
            dA = np.dot(self.parameters['W' + str(l)].T, dZ) 
            
            ## Apply dropout
            if keep_prob is not None:
                # notice that we subtract 1 because the dA that we work with now is the dA of the layer l-1 (the next layer in the backward propagation)
                # print("l = ", l - 1)
                # print("dA shape = ", dA.shape)
                # print("D shape = ", cashes['D' + str(l - 1)].shape)
                # print("keep_prob = ", keep_prob[l - 1])
                dA = np.multiply(dA, cashes['D' + str(l - 1)]) # we will use the same D that we created in the forward propagation to shut down the upstream of the neurons that were shut down in the forward propagation
                dA /= keep_prob[l - 1] # divide by the keep probability to keep the expected value of the derivatives the same as before dropping out some of them
            grads['db' + str(l)] = db
            grads['dW' + str(l)] = dW
        return grads
    
    
    ## backward propagation helper functions

    # define the backward activations they take dA and return dZ by multiplying dA by the derivative of the activation function element wise
    def sigmoid_backward(self, dA, A):
        g_dash = A * (1 - A)
        return dA * g_dash
    
    def relu_backward(self, dA, Z):
        # we wil multiply dA by 1 if Z > 0 and 0 if Z <= 0, so instead we pass dA if Z > 0 and 0 if Z <= 0
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0 # keep dA and reset the rest -where dZ <= 0-  to 0
        return dZ
    
    def tanh_backward(self, dA, A):
        g_dash = (1 - np.power(A, 2))
        return dA * g_dash
    
    def softmax_backward(self, dA, A):
        g_dash = A * (1 - A)
        return dA * g_dash
    
    
    # define the update parameters
    def update_parameters(self, grads, learning_rate, adam_counter, optimizer = "gd", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        if optimizer == "adam":
            v_corrected = {} # temporary variables to store the corrected values of v and s and use them in the update equations
            s_corrected = {}


        for l in range(1, self.num_layers): # num_layers is not included, but since it contains the input layers , things add up
            if optimizer == "gd":
                # for normal gradient descent, we use the normal update equation on dW and db
                # w = w - learning_rate * dw
                self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
                # b = b - learning_rate * db
                self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
            elif optimizer == "momentum":
                ## For the momentum optimizer, we use the exponentially weighted averages of the gradients to update the parameters
                # Calculate the exponentially weighted averages of the gradients
                self.v["dW" + str(l)] = beta1 * self.v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
                self.v["db" + str(l)] = beta1 * self.v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]
                # Update the parameters using the exponentially weighted averages of the gradients
                # W = W - learning_rate * v
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * self.v["dW" + str(l)]
                # B = B - learning_rate * v
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * self.v["db" + str(l)]
            elif optimizer == "rmsprop":
                ## For the RMSprop optimizer, we use the exponentially weighted averages of the squared gradients to update the parameters
                self.s["dW" + str(l)] = beta1 * self.s["dW" + str(l)] + (1 - beta1) * np.square(grads['dW' + str(l)])
                self.s["db" + str(l)] = beta1 * self.s["db" + str(l)] + (1 - beta1) * np.square(grads['db' + str(l)])
                # Update the parameters using the exponentially weighted averages of the squared gradients
                # W = W - learning_rate * (dW / sqrt(s) + epsilon)
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * (grads['dW' + str(l)] / (np.sqrt(self.s["dW" + str(l)]) + epsilon))
                # B = B - learning_rate * (db / sqrt(s) + epsilon)
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * (grads['db' + str(l)] / (np.sqrt(self.s["db" + str(l)]) + epsilon))
            elif optimizer == "adam":
                ## for the adam optimizer, we use the exponentially weighted averages of both the gradients and the squared gradients to update the parameters
                self.v["dW" + str(l)] = beta1 * self.v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
                self.v["db" + str(l)] = beta1 * self.v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]
                self.s["dW" + str(l)] = beta2 * self.s["dW" + str(l)] + (1 - beta2) * np.square(grads['dW' + str(l)])
                self.s["db" + str(l)] = beta2 * self.s["db" + str(l)] + (1 - beta2) * np.square(grads['db' + str(l)])
                # correct the values of v and s
                v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - np.power(beta1, adam_counter))
                v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - np.power(beta1, adam_counter))
                s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - np.power(beta2, adam_counter))
                s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - np.power(beta2, adam_counter))
                # Update the parameters
                # W = W - learning_rate * (v_corrected / sqrt(s_corrected) + epsilon)
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
                # B = B - learning_rate * (v_corrected / sqrt(s_corrected) + epsilon)
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))


            
    # define the train function
    def train(self, X, Y , learning_rate, num_epochs, batch_size = 64, lambd = 0, keep_prob = None, print_cost = True, loss = 'BinaryCrossEntropy', optimizer = "gd", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        # preparing some important variables
        seed = 10 # The seed for randomly generating the mini batches each epoch
        #M = X.shape[1] # number of training examples in the whole dataset, we use it to divide the accumulated cost by it to get the average cost of all examples
        adam_counter = 0 # this variable is used to count the number of iterations of the adam optimizer, it is used to correct the values of v and s in the adam optimizer
        if keep_prob != None:
            assert len(keep_prob) == self.num_layers - 1, "The number of keep probabilities must be the same as the number of hidden layers + the input layer"
        
        
        ##  Initialize the exponentially weighted averages if the optimizer is momentum or RMSprop or adam 
        if optimizer == "momentum":
            self.v = self.initialize_averages(self.parameters, optimizer)
        elif optimizer == "rmsprop":
            self.s = self.initialize_averages(self.parameters, optimizer)
        elif optimizer == "adam":
            self.v, self.s = self.initialize_averages(self.parameters, optimizer)
        
        
        
        # Start the training loop
        for i in range(num_epochs):

            ## Define and prepare the random mini batches
            seed = seed + 1 # so that we generate different mini batches each epoch
            mini_batches = self.random_mini_batches(X, Y, batch_size=batch_size, seed=seed)
            #total_cost = 0 # we will accummulate the cost of all the mini batches in this variable

            ## For each epoch we will loop over all the different mini batches
            for mini_batch in mini_batches:
                X_batch, Y_batch = mini_batch
                
                ## Forward propagation
                Y_prid_batch, cashes = self.forward_propagation(X_batch,keep_prob)
                ## Compute cost
                cost = self.compute_cost(Y_prid_batch, Y_batch, lambd, loss) # The cost will be of the mini-batch if we use mini-batch GD (won't be very useful) and the whole dataset if we use batch GD (batch size = M)
                #total_cost += cost # If we are using batch GD, it would be useful
                self.costs.append(cost)
                ## Backward propagation
                grads = self.backward_propagation(Y_prid_batch, Y_batch, cashes, lambd, keep_prob, loss)
                ## Update parameters
                if optimizer == "adam":
                    adam_counter += 1
                self.update_parameters(grads, learning_rate,adam_counter=adam_counter,optimizer=optimizer,beta1=beta1,beta2=beta2,epsilon=epsilon)
            
            #self.costs.append(total_cost)
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost)) 
        return self.parameters,self.costs
    

    def random_mini_batches(self, X, Y, batch_size = 64, seed = 0):
        # Some variable initializations 
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        
        ## Shuffle the data
        # Create a list of random numbers from 0 to m-1
        permutation = list(np.random.permutation(m))
        # Use that list to shuffle the data by indexing the columns of X and Y with the random numbers order
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        ## partition the data
        num_complete_batches = m // batch_size
        #print("num_complete_batches = ", num_complete_batches)
        for k in range(num_complete_batches):
            mini_batch_X = shuffled_X[:, k*batch_size : (k+1)*batch_size]
            mini_batch_Y = shuffled_Y[:, k*batch_size : (k+1)*batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # handle the case when the last batch is not complete
        if m % batch_size != 0:
            #print("from inside the incomplete loop")
            mini_batch_X = shuffled_X[:, num_complete_batches*batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_batches*batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    
    # optimizers helper functions
    def initialize_averages(self, parameters, optimizer):
        # the number of layers except the input layer will be half the number of the parameters
        L = len(parameters) // 2
        

        if optimizer == "momentum":
            v = {} 
            # The variable keeps track of the exponentialy weighted averages of the gradients in case of momentum and the squared gradients in case of RMSprop, but they are initialized in the same way to zeros
            for l in range(1, L + 1):
                v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            return v
        elif optimizer == "rmsprop":
            s = {}
            # in here we will keep track of the exponentially weighted averages of the squared gradients and initialize them to zeros
            for l in range(1, L + 1):
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            return s
        elif optimizer == "adam":
            v = {}
            s = {}
            # in here we will keep track of both the exponentially weighted averages of the gradients and the squared gradients and initialize them to zeros
            for l in range(1, L + 1):
                v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            return v, s

    
    # define the accuracy function
    def accuracy(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y) # 1 if they are equal, 0 if they are not, so the mean is the accuracy (count of 1s / total count)
        return accuracy
    
    # define the predict function
    def predict(self, X):
        A, cashes = self.forward_propagation(X)
        predictions = (A > 0.5)
        return predictions

    def gradient_check(self, X, Y, epsilon = 1e-7):
        # take the current parameters and reshape them into a vector
        #print("parameters keys:" , self.parameters.keys())
        parameters_values = self.dictionary_to_vector(self.parameters)
        
        ## get the gradients
        # apply forward propagation on the current parameters
        A, cashes = self.forward_propagation(X)
        # compute the gradients using backward propagation
        grads = self.backward_propagation(A, Y, cashes)
        # reshape the gradients into a vector
        # reverse the order of the grads keys to match the order of the parameters keys
        grads = {key:grads[key] for key in reversed(grads.keys())}
        #print("grads keys:" , grads.keys())
        grads_values = self.dictionary_to_vector(grads)

        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        grad_approx = np.zeros((num_parameters, 1))

        for i in range(num_parameters):
            # compute J_plus[i]
            thetaplus = np.copy(parameters_values) # copy the parameters values to avoid changing the original values
            thetaplus[i][0] += epsilon # nudge only the intended parameter of derivative and leave the rest as they are 
            # calculate the cost after nudging the parameter to the right and fixing the rest of the parameters
            A, cashes = self.forward_propagation(X, parameters = self.vector_to_dictionary(thetaplus))
            J_plus[i] = self.compute_cost(A, Y)
            
            # compute J_minus[i]
            thetaminus = np.copy(parameters_values) # copy the parameters values to avoid changing the original values
            thetaminus[i][0] -= epsilon # nudge only the intended parameter of derivative and leave the rest as they are
            # calculate the cost after nudging the parameter to the left and fixing the rest of the parameters
            A, cashes = self.forward_propagation(X, parameters = self.vector_to_dictionary(thetaminus))
            J_minus[i] = self.compute_cost(A, Y)
            
            # compute grad_approx[i]
            grad_approx[i] = (J_plus[i] - J_minus[i])/ ( 2 * epsilon)

        numerator = np.linalg.norm(grad_approx - grads_values)
        denominator = np.linalg.norm(grads_values) + np.linalg.norm(grad_approx)
        difference = numerator/denominator

        if difference > 2e-7:
            print("\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92m" + "The backward propagation works fine! difference = " + str(difference) + "\033[0m")
        
    
    def dictionary_to_vector(self, parameters):
        count = 0
        for key in parameters.keys():
            #print("key = ", key)
            new_vector = np.reshape(parameters[key], (-1,1))
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count += 1
        return theta
    
    def vector_to_dictionary(self, theta):
        parameters = {}
        L = len(self.layer_dims)
        start = 0
        for l in range(1, L):
            cuurrent_W_shape = self.layer_dims[l]*self.layer_dims[l-1]
            current_b_shape = self.layer_dims[l]
            parameters['W' + str(l)] = theta[start:start + cuurrent_W_shape].reshape((self.layer_dims[l], self.layer_dims[l-1]))
            parameters['b' + str(l)] = theta[start + cuurrent_W_shape: start + cuurrent_W_shape +current_b_shape].reshape((self.layer_dims[l], 1))
            start += cuurrent_W_shape + current_b_shape
        return parameters        


# if __name__ == "__main__":
#     from opt_utils_v1a import load_dataset
#     train_X, train_Y = load_dataset()
#     my_model = ClassifierNN(layer_dims=[train_X.shape[0], 5, 2, 1], activations=["relu", "relu", "sigmoid"])
#     params, costs = my_model.train(train_X, train_Y, optimizer="gd", learning_rate=0.0007, num_epochs=5000, print_cost=True)

