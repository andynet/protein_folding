import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def calc_prob(x, _mean, _sd):
    """
    Calculates the output of a probability function at value x. The probability function
    is normal with center at _mean and standard deviation _sd
    """
    
    if x > 22:
        if _mean >= 22:
            return torch.tensor(1.0)
        else:
            x = 22
    return torch.exp(- (1 / 2) * ((x - _mean) / _sd) ** 2) / (_sd * torch.sqrt(torch.tensor(2 * np.pi)))


def loss(pred, real, sd = torch.tensor(4)):
    """NLL likelihood"""
    
    loss = 0
    
    for i in range(len(pred) - 1):
        for j in range(i + 1, len(pred)):
            prob = calc_prob(pred[i, j], real[i, j], sd)
            loss += torch.log(prob)
    return -loss


def optimize(structure, 
             iterations=100, 
             lr=1e-3, 
             lr_decay=1,
             min_lr=1e-10,
             decay_frequency=10,
             normalize_gradients=True,
             momentum=0,
             nesterov=False,
             verbose=1, 
             img_dir=None):
    """
    Optimize structure (phi and psi angles) given a label
    """
    history = []
    min_loss = np.inf
    
    structure = copy(structure0)
    
    if momentum > 1 or momentum < 0:
        print('Momentum parameter has to be between 0 and 1')
        return
    
    # initialize V for momentum
    V = torch.zeros((len(structure.torsion)))
    
    for i in range(iterations):
        if structure.torsion.grad is not None:
            structure.torsion.grad.zero_()
        
        if nesterov:
            structure.torsion = (structure.torsion + momentum * V).detach().requires_grad_()
            
        temp_distmap = structure.G()
        L = loss(temp_distmap, structure.Y)
        L.backward()
        
        #print(structure.phi.grad[:5], structure.phi.grad[:5])
        if normalize_gradients:
            # normalize gradients
            structure.torsion.grad = (structure.torsion.grad - torch.mean(structure.torsion.grad)) / torch.std(structure.torsion.grad)
        
        #print(structure.phi.grad[:5], structure.phi.grad[:5])
        
        # Implementing momentum
        V = momentum * V - lr * structure.torsion.grad
        
        structure.torsion = (structure.torsion + V).detach().requires_grad_()
        
        if verbose is not None:
            if i % verbose == 0:
                print(f'Iteration {i}, Loss: {L.item()}')
                
        history.append([i, L.item()])
        
        if i % decay_frequency == 0 and i > 0:
            lr *= lr_decay
        
        if L.item() < min_loss:
            best_structure = copy(structure)
            min_loss = L.item()
            
        if img_dir is not None:
            structure.visualize_structure('{}/iter_{:04d}.png'.format(img_dir, i))
        
        if lr < min_lr:
            break
            
    return best_structure, min_loss, np.array(history)


def optimize0(structure0, iterations, 
             lr=1e-3, 
             lr_decay=1, 
             decay_frequency=10,
             normalize_gradients=True,
             momentum=0,
             verbose=1, 
             img_dir=None):
    """
    Optimize structure (phi and psi angles) given a label
    """
    history = []
    min_loss = np.inf
    
    structure = copy(structure0)
    
    if momentum > 1 or momentum < 0:
        print('Momentum parameter has to be between 0 and 1')
        return
    
    # initialize V for momentum
    V_phi, V_psi = torch.zeros((len(structure.phi))), torch.zeros((len(structure.psi)))
    
    for i in range(iterations):
        if structure.phi.grad is not None:
            structure.phi.grad.zero_()
        if structure.psi.grad is not None:
            structure.psi.grad.zero_()
        
        temp_distmap = structure.G()
        L = loss(temp_distmap, structure.Y)
        L.backward()
        
        if normalize_gradients:
            # normalize gradients
            structure.phi.grad = (structure.phi.grad - torch.mean(structure.phi.grad)) / torch.std(structure.phi.grad)
            structure.psi.grad = (structure.psi.grad - torch.mean(structure.psi.grad)) / torch.std(structure.psi.grad)
        print(structure.phi.grad[:5], structure.phi.grad[:5])
        # Implementing momentum
        V_phi = momentum * V_phi - lr * structure.phi.grad
        V_psi = momentum * V_psi - lr * structure.phi.grad
        
        structure.phi = (structure.phi + V_phi).detach().requires_grad_()
        structure.psi = (structure.psi + V_psi).detach().requires_grad_()
        
        if verbose is not None:
            if i % verbose == 0:
                print(f'Iteration {i}, Loss: {L.item()}')
                
        history.append([i, L.item()])
        
        if i % decay_frequency == 0 and i > 0:
            lr *= lr_decay
        
        if L.item() < min_loss:
            best_structure = copy(structure)
            min_loss = L.item()
            
        if img_dir is not None:
            structure.visualize_structure('{}/iter_{:04d}.png'.format(img_dir, i))
    
    return best_structure, min_loss, np.array(history)