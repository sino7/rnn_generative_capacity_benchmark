import torch
from torch import nn
import numpy as np

class TRNN(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_h):
        super(TRNN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_h = tau_h
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h = (1-1/self.tau_h) * h + (1/self.tau_h) * (
                torch.mm(
                    torch.tanh(h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs

        return errors
    

class LSTM(nn.Module):
    
    def __init__(self, states_dim, output_dim):
        super(LSTM, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent cell
        self.cell = torch.nn.LSTMCell(output_dim, states_dim)
        
        # Predictions, states and errors
        self.x_pred = None
        self.error = None
        self.h = None
        self.c = None
        
    def forward(self, x, c_init, store=True, teacher_forcing=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - c_init : initial cell state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        - teacher_forcing : whether to reinject the prediction as input, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        c = c_init
        h = torch.tanh(c)
        
        # Input
        x_in = torch.zeros(batch_size, self.output_dim)
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h, c = self.cell(x_in, (h, c))
            if store:
                hs[t] = h.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(h, self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
            if teacher_forcing:
                x_in = x[t]

            else:
                x_in = x_pred.detach()
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs

        return errors
    
    
class GRU(nn.Module):
    
    def __init__(self, states_dim, output_dim):
        super(GRU, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent cell
        self.cell = torch.nn.GRUCell(output_dim, states_dim)
        
        # Predictions, states and errors
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True, teacher_forcing=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        - teacher_forcing : whether to reinject the prediction as input, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
        
        # Input
        x_in = torch.zeros(batch_size, self.output_dim)
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h = self.cell(x_in, h)
            if store:
                hs[t] = h.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(h, self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
            if teacher_forcing:
                x_in = x[t]

            else:
                x_in = x_pred.detach()
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs

        return errors
    
class ESN(nn.Module):
    
    def __init__(self, states_dim, output_dim, proba, sigma, tau):
        super(ESN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.proba = proba  # probability of recurrent connection
        self.sigma = sigma  # scale of the recurrent connection
        self.tau = tau
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_r = torch.randn(self.states_dim, self.states_dim) * sigma
        self.connect = torch.rand(self.states_dim, self.states_dim) < proba
        self.w_r *= self.connect
        self.b_r = torch.randn(self.states_dim)/10
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
                
        for t in range(seq_len):
            
            # Top-down pass

            # Compute h_prior according to past h_post
            h = (1-1/self.tau) * h + (1/self.tau) * (
                torch.mm(
                    torch.tanh(h),
                    self.w_r.T
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()

            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()

            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.h = hs
            self.x_pred = x_preds
            self.error = errors
            
        return errors
    
    
class MTRNN(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau_slow, tau_fast):
        super(MTRNN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau_slow = tau_slow
        self.tau_fast = tau_fast
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim) / self.states_dim
        self.b_o = torch.randn(self.output_dim)/10
        
        # Recurrent weights initialization
        self.w_ss = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_s = torch.randn(self.states_dim)/10

        self.w_ff = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        self.b_f = torch.randn(self.states_dim)/10
        
        self.w_sf = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        
        self.w_fs = torch.randn(self.states_dim, self.states_dim) / self.states_dim
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.hf = None
        self.hs = None
        
    def forward(self, x, hs_init, hf_init, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        """

        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hss = torch.zeros(seq_len, batch_size, self.states_dim)
            hfs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        old_hs = hs_init
        old_hf = hf_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            hs = (1-1/self.tau_slow) * old_hs + (1/self.tau_slow) * (
                torch.mm(
                    torch.tanh(old_hs),
                    self.w_ss
                ) + torch.mm(
                    torch.tanh(old_hf),
                    self.w_fs
                ) + self.b_s.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hss[t] = hs.detach()
                
            # Compute h_prior according to past h_post
            hf = (1-1/self.tau_fast) * old_hf + (1/self.tau_fast) * (
                torch.mm(
                    torch.tanh(old_hs),
                    self.w_sf
                ) + torch.mm(
                    torch.tanh(old_hf),
                    self.w_ff
                ) + self.b_f.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hfs[t] = hf.detach()
            
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(hf), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
            old_hs = hs
            old_hf = hf
            
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.hs = hss
            self.hf = hfs

        return errors
    
class AntisymetricRNN(nn.Module):
    
    def __init__(self, states_dim, output_dim, tau, gamma):
        super(AntisymetricRNN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        
        # Output weights initialization
        self.w_o = torch.nn.Parameter(torch.randn(self.output_dim, self.states_dim) / self.states_dim)
        self.b_o = torch.nn.Parameter(torch.randn(self.output_dim)/10)
        
        # Recurrent weights initialization
        self.v_r = torch.nn.Parameter(torch.randn(self.states_dim,self.states_dim) / self.states_dim)
        self.b_r = torch.nn.Parameter(torch.randn(self.states_dim)/10)
        
        self.mask = torch.zeros(self.states_dim, self.states_dim)
        k=0
        for i in range(self.states_dim):
            for j in range(i):
                self.mask[i, j] = 1
                k+=1
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """
        
        # Triangular matrix from v_r
        w_r = self.v_r * self.mask
       
        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute h_prior according to past h_post
            h = h + (1/self.tau) * torch.tanh(
                torch.mm(
                    torch.tanh(h),
                    (w_r - w_r.transpose(0, 1))
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1) \
                - self.gamma * h
            )
            if store:
                hs[t] = h.detach()
                
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs

        return errors
    
class UGRNN(nn.Module):
    
    def __init__(self, states_dim, output_dim):
        super(UGRNN, self).__init__()
        
        self.states_dim = states_dim
        self.output_dim = output_dim
        
        # Output weights initialization
        self.w_o = torch.nn.Parameter(torch.randn(self.output_dim, self.states_dim) / self.states_dim)
        self.b_o = torch.nn.Parameter(torch.randn(self.output_dim)/10)
        
        # Recurrent weights initialization
        self.w_u = torch.nn.Parameter(torch.randn(self.states_dim,self.states_dim) / self.states_dim)
        self.b_u = torch.nn.Parameter(torch.randn(self.states_dim)/10)
        
        # Recurrent weights initialization
        self.w_r = torch.nn.Parameter(torch.randn(self.states_dim,self.states_dim) / self.states_dim)
        self.b_r = torch.nn.Parameter(torch.randn(self.states_dim)/10)
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h = None
        
    def forward(self, x, h_init, store=True):
        """
        Forward pass through the network. The target is part of this method only to
        directly compute the prediction error, it is not involved in the recurrent
        computations of the forward pass.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : initial hidden state, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """
       
        seq_len, batch_size, _ = x.shape
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            hs = torch.zeros(seq_len, batch_size, self.states_dim)
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initial hidden state and hidden causes
        h = h_init
        
        for t in range(seq_len):
            
            # Top-down pass
            
            # Compute the update gate according to h
            u = torch.sigmoid(
                torch.mm(
                    torch.tanh(h),
                    self.w_u
                ) + self.b_u.unsqueeze(0).repeat(batch_size, 1)
            )
            
            # Compute h according to past h and u
            h = u * h + (1 - u) * torch.tanh(
                torch.mm(
                    torch.tanh(h),
                    self.w_r
                ) + self.b_r.unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                hs[t] = h.detach()
                
            # Compute x_pred according to h_prior
            x_pred =  torch.mm(torch.tanh(h), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
                
            error = x_pred - x[t]
            errors[t] = error
            
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.h = hs

        return errors