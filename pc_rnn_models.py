import torch
from torch import nn
import numpy as np

class PC_RNN_V(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_h, alpha_x):
        super(PC_RNN_V, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        
    def forward(self, x, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
          
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
                
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()
                    
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
    
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
        return errors

####################################################################################################################################    
    
class PC_RNN_GC(nn.Module):
    
    def __init__(self, states_dim, output_dim, alpha_h_prime, alpha_x, tau):
        super(PC_RNN_GC, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.alpha_h_prime = alpha_h_prime
        self.alpha_x = alpha_x
        self.tau = tau
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10

        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10

        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.h_prime = None
        self.h_prior = None
        self.h_post = None
        self.error = None
        self.error_h_primes = None
        
    def forward(self, x, h_init=0, store=True, with_h_inference=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        - with_h_inference : whether to infer h based on the h', Boolen
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_primes = torch.zeros(seq_len, batch_size, self.states_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_h_primes = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        h_prime = 0
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute the h_prime_error according to h_prime and old_h_post
            error_h_prime = h_prime - (1/self.tau)*(
                torch.mm(torch.tanh(old_h_post), self.w_r.T) \
                + self.b_r.unsqueeze(0).repeat(batch_size, 1)\
                - old_h_post
            )
            if store:
                error_h_primes[t] = error_h_prime.detach()
            
            # Compute h_prime according to h_prime and h_prime_error
            h_prime = h_prime - self.alpha_h_prime * error_h_prime
            if store:
                h_primes[t] = h_prime.detach()
                
            # Compute h_prior according to old_h_post and h_prime
            h_prior = old_h_post + h_prime
            
            if with_h_inference:
                h_prior = h_prior + (self.alpha_h_prime/self.tau) * (
                    (1 - torch.tanh(old_h_post)**2) * torch.mm(error_h_prime, self.w_r) \
                    - error_h_prime
                )
                
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
            
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                old_h_post = h_post
                
            else:
                old_h_post = h_prior
        
        if store:
            self.error = errors.detach()
            self.error_h_prime = error_h_primes
            self.x_pred = x_preds
            self.h_prior = h_priors
            self.h_post = h_posts
        return errors
    
####################################################################################################################################    
    
class PC_RNN_HC_A(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_A, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        self.alpha_h = alpha_h
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_c = torch.randn(self.states_dim, self.causes_dim) / self.causes_dim
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + \
                torch.mm(
                    c,
                    self.w_c.T
                ) + \
                self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
            
            # Bottom-up pass
            if self.alpha_x > 0:

                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(error_h, self.w_c)
                if store:
                    cs[t] = c
                             
                old_h_post = h_post
            
            else:
                old_h_post = h_prior
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors
    
    def learn(self, lr_o, lr_h, lr_c):
        """
        Updating the model parameters based on the prediction error
        Parameters :        
        - lr_o : learning rate for the output weights
        - lr_h : learning rate for the recurrent weights
        - lr_c : learning rate for the causes to hidden weights
        """
        
        seq_len, batch_size, _ = self.x_pred.shape
        
        delta_w_o = torch.mm(
            self.error.reshape(seq_len*batch_size, self.output_dim).T, 
            torch.tanh(self.h_prior.reshape(seq_len*batch_size, self.states_dim))
        )
        self.w_o -= lr_o * delta_w_o
        
        delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), axis=0)
        self.b_o -= lr_o * delta_b_o
        
        delta_w_c = torch.mm(
            self.error_h.reshape(seq_len*batch_size, self.states_dim).T,
            self.c.reshape(seq_len*batch_size, self.causes_dim)
        )
        self.w_c -= lr_c * delta_w_c
        
        delta_w_r = torch.mm(
            self.error_h[1:].reshape((seq_len-1)*batch_size, self.states_dim).T,
            torch.tanh(self.h_post[:-1].reshape((seq_len-1)*batch_size, self.states_dim))
        )
        self.w_r -= lr_h * delta_w_r
            
        delta_b_r = torch.sum(self.error_h.reshape(seq_len*batch_size, self.states_dim), axis=0)
        self.b_r -= lr_h * delta_b_r
    
####################################################################################################################################

class PC_RNN_HC_M(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, factor_dim, tau_h, alpha_x, alpha_h):
        super(PC_RNN_HC_M, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.factor_dim = factor_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        self.alpha_h = alpha_h
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_pd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_fd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_cd = torch.nn.Softmax(1)(0.5*torch.randn(self.causes_dim, self.factor_dim))*self.factor_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h_prior = (1-1/self.tau_h) * old_h_post + (1/self.tau_h) * (
                torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    ) * torch.mm(
                        c,
                        self.w_cd
                    ),
                    self.w_fd.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[t] = h_prior.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error

            # Bottom-up pass
            if self.alpha_x>0:
            
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_h*torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    )* torch.mm(
                        error_h,
                        self.w_fd
                    ),
                    self.w_cd.T
                )
                if store:
                    cs[t] = c

                old_h_post = h_post
                
            else:
                old_h_post = h_prior
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                             
        return errors
    
####################################################################################################################################    
    
class PC_RNN_GC_HC_A(nn.Module):
    
    def __init__(self, causes_dim, states_dim, output_dim, tau_h, alpha_x, alpha_h, alpha_hp):
        super(PC_RNN_GC_HC_A, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        self.alpha_h = alpha_h
        self.alpha_hp = alpha_hp
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_c = torch.randn(self.states_dim, self.causes_dim) / self.causes_dim
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.error_h = None
        self.hp_prior = None
        self.hp_post = None
        self.error_hp = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
            hp_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            hp_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hps = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        old_hp_post = 0
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute hp_prior according to past h_post
            hp_prior = (1-1/self.tau_h) * old_hp_post + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(old_h_post),
                    self.w_r.T
                ) + \
                torch.mm(
                    c,
                    self.w_c.T
                ) + \
                self.b_r.unsqueeze(0).repeat(batch_size, 1) - \
                old_h_post
            )
            if store:
                hp_priors[t] = hp_prior.detach()
                
            # Compute h_prior according to hp_prior
            h_prior = old_h_post + hp_prior
            if store:
                h_priors[t] = h_prior.detach()
                
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
            
            # Bottom-up pass
            if self.alpha_x > 0:
           
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer hp_post according to hp_prior and the error on the h level
                hp_post = hp_prior - self.alpha_h*error_h
                if store:
                    hp_posts[t] = hp_post.detach()

                # Compute the error on the h' level
                error_hp = hp_prior - hp_post
                if store:
                    error_hps[t] = error_hp.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_hp*torch.mm(error_hp, self.w_c)
                if store:
                    cs[t] = c

                old_h_post = h_post
                old_hp_post = hp_post
                
            else:
                old_h_post = h_prior
                old_hp_post = hp_prior
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h_prior = h_priors
            self.h_post = h_posts
            self.error_h = error_hs
            self.hp_prior = hp_priors
            self.h_post = hp_posts
            self.error_hp = error_hps
            self.c = cs
                             
        return errors
    
####################################################################################################################################    
    
class PC_RNN_GC_HC_M(nn.Module):
    
    def __init__(self, causes_dim, states_dim, factor_dim, output_dim, tau_h, alpha_x, alpha_h, alpha_hp):
        super(PC_RNN_GC_HC_M, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.factor_dim = factor_dim
        self.states_dim = states_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        self.alpha_h = alpha_h
        self.alpha_hp = alpha_hp
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim) / 10
        
        # Recurrent weights initialization
        self.w_pd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_fd = torch.randn(self.states_dim, self.factor_dim) / np.sqrt(self.factor_dim)
        self.w_cd = torch.nn.Softmax(1)(0.5*torch.randn(self.causes_dim, self.factor_dim))*self.factor_dim
        self.b_r = torch.randn(self.states_dim) / 10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = None
        self.h_post = None
        self.error_h = None
        self.hp_prior = None
        self.hp_post = None
        self.error_hp = None
        self.c = None
        
    def forward(self, x, c_init, h_init=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            h_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            h_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hs = torch.zeros(seq_len, batch_size, self.states_dim)
            hp_priors = torch.zeros(seq_len, batch_size, self.states_dim)
            hp_posts = torch.zeros(seq_len, batch_size, self.states_dim)
            error_hps = torch.zeros(seq_len, batch_size, self.states_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_h_post = h_init
        old_hp_post = 0
        c = c_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute hp_prior according to past h_post
            hp_prior = (1-1/self.tau_h) * old_hp_post + (1/self.tau_h) * (
                torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    ) * torch.mm(
                        c,
                        self.w_cd
                    ),
                    self.w_fd.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1) - \
                old_h_post
            )
            if store:
                hp_priors[t] = hp_prior.detach()
                
            # Compute h_prior according to hp_prior
            h_prior = old_h_post + hp_prior
            if store:
                h_priors[t] = h_prior.detach()
                
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h_prior), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error

            # Bottom-up pass
            if self.alpha_x > 0:
                
                # Infer h_post according to h_prior and the error on the sensory level
                h_post = h_prior - self.alpha_x*(1-torch.tanh(h_prior)**2)*torch.mm(error, self.w_o)
                if store:
                    h_posts[t] = h_post.detach()

                # Compute the error on the hidden state level
                error_h = h_prior - h_post
                if store:
                    error_hs[t] = error_h.detach()

                # Infer hp_post according to hp_prior and the error on the h level
                hp_post = hp_prior - self.alpha_h*error_h
                if store:
                    hp_posts[t] = hp_post.detach()

                # Compute the error on the h' level
                error_hp = hp_prior - hp_post
                if store:
                    error_hps[t] = error_hp.detach()

                # Infer c according to its past value and the error on the hidden state level
                c = c - self.alpha_hp*torch.mm(
                    torch.mm(
                        torch.tanh(old_h_post),
                        self.w_pd
                    )* torch.mm(
                        error_hp,
                        self.w_fd
                    ),
                    self.w_cd.T
                )
                if store:
                    cs[t] = c

                old_h_post = h_post
                old_hp_post = hp_post
                
            else:
                old_h_post = h_prior
                old_hp_post = hp_prior
        
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h_prior = h_priors
            self.h_post = h_posts
            self.error_h = error_hs
            self.hp_prior = hp_priors
            self.h_post = hp_posts
            self.error_hp = error_hps
            self.c = cs
                             
        return errors