
# coding: utf-8

# In[4]:


import numpy as np
from scipy.linalg import eigvals,solve


# In[44]:


# model parameters.params fitted to East Africa (egypt paper)
alpha_bm = 1.41
beta_bm = 3.73
gamma_bm = -1.87

S = 30


# In[52]:


# building an interaction matrix by the mass ratio model

def build_food_web(species_number, model, alpha, beta, gamma):
    species_masses = generate_species(species_number)
    pred_matrix = build_predation_matrix(species_masses, model)
    return pred_matrix

def generate_species(species_number):
    n = np.random.uniform(0, 1, species_number)
    #n.sort()
    return n

def mass_ratio_model(mass_i, mass_j, alpha, beta, gamma):
    p = np.exp(alpha + beta * np.log(mass_i/mass_j) + gamma * np.log(mass_i/mass_j)**2)
    prob_interaction = p / (1 + p)
    return prob_interaction

def build_predation_matrix(species_masses, model):
    S = len(species_masses)
    int_matrix = np.zeros((S,S))
    for i in range(S):
        for j in range(S):
            int_prob = model(species_masses[i], species_masses[j], alpha_bm, beta_bm, gamma_bm)
            rand_draw = np.random.uniform(0,1)
            if rand_draw < int_prob:
                int_matrix[i,j] = 1
                int_matrix[j,i] = -1
    for k in range(S):
        int_matrix[k,k] = -1
    return int_matrix


# In[53]:


# generalized modelling jacobian and stability analysis

def web_stability(interaction_matrix):
    primary_producers, top_predators = producers_and_predators(interaction_matrix)
    alpha, gamma, lambdas, mu, phi, psi, rho, sigma, chi, beta = make_random_model(interaction_matrix, 
                                                                                   primary_producers, top_predators)
    jacobian = make_jacobian(interaction_matrix, alpha, gamma, lambdas, mu, phi, psi, rho, sigma, chi, beta)
    stability = test_stability(jacobian)
    return stability
    

def producers_and_predators(interaction_matrix):  
    N = len(interaction_matrix)  #NUMBER OF SPECIES 

    #Provide Primary producer and top predator array
    primary_producers = np.ones(N)
    top_predators = np.ones(N)
    for i in range(N):
        if sum(interaction_matrix[i,:]) != interaction_matrix[i,i]:  top_predators[i] = 0
        if sum(interaction_matrix[:,i]) != interaction_matrix[i,i]:  primary_producers[i] = 0      
      
    return primary_producers, top_predators 

def make_random_model(interaction_matrix, primary_producers, top_predators):     #SETTING UP A MODEL WITH RANDOM PARAMETERS!!
    N = interaction_matrix.shape[0]
    alpha = np.random.uniform(0.01, 1., size=N) #Alpha Uniform range
    gamma = np.random.uniform(0.5, 1.5, size=N) #Gamma Uniform range
    lambdas = np.ones((N,N))                    #Passive Prey switiching
    mu = np.random.uniform(1., 2., size=N)     #EUniform mortality exponent
    phi = np.random.uniform(0., 1., size=N)     #Uniform prim.prod exponent
    psi = np.random.uniform(0.5, 1.5, size=N)   #Uniform predation exponent
    rho = 1.- primary_producers.copy()         # 0 predation gain for primprod, 1 for others
    
    #predation loss is 0 for top predators, uniform else
    sigma = (1.- top_predators.copy()) * np.random.uniform(0.25, 1., size=N)   
      
    chi = np.zeros((N,N))
    beta = np.zeros((N,N))
    for i_pred in range(N):             #Assign values to the nonzero feeding parameters. 
        for i_prey in range(N):         # Can also be done by directly using biomass flows.
            if interaction_matrix[i_prey,i_pred] != 0:
                chi[i_pred, i_prey] = np.random.uniform(0.1, 1.0)  #Note the inverse order of chi,beta and lima!!
                beta[i_pred, i_prey] = np.random.uniform(0.1, 1.0)
                
    return alpha, gamma, lambdas, mu, phi, psi, rho, sigma, chi, beta
           # _normalizeFlows()
    
def make_jacobian(interaction_matrix, alpha, gamma, lambdas, mu, phi, psi, rho, sigma, chi, beta ):
    N = interaction_matrix.shape[0]
    jacobian = np.zeros((N, N))
    
    for n in range(N):
        for i in range(N):
            if (i != n): # off diagonal
                dsum = 0; # for calculating the sum in the loss by predation part for mutualistic effects
                for m in range (N):
                    dsum +=  beta[m][n] * lambdas[m][i] * (gamma[m] - 1.) * chi[m][i]
                
                    # gain by predation -  loss by predation
                    jacobian[n][i] = alpha[n] * (rho[n] * gamma[n] * chi[n][i] * lambdas[n][i] 
                                                 - sigma[n] * (beta[i][n] * psi[i] + dsum) )
        for i in range(N):# diagonals
            dsum = 0;
            for m in range(N):
                dsum  += beta[m][i] * lambdas[m][i] * ( (gamma[m] - 1.) * chi[m][i]  + 1. );
            
               # primary production + gain by predation  - mortality- loss by predation
                jacobian[i][i] = alpha[i]*( (1.-rho[i])*phi[i] + rho[i] 
                                             * (gamma[i] * chi[i][i]  * lambdas[i][i] 
                                             + psi[i] ) - (1.- sigma[i]) * mu[i] - sigma[i] 
                                             * ( beta[i][i] * psi[i] + dsum ) )                         

    return jacobian

def test_stability(jacobian):  
    if max(eigvals(jacobian))<- 1e-6:  return True
    return False

def get_eigenvalues(jacobian):
    return sorted(eigvals(jacobian))


# In[54]:


def percent_stable_webs(species_number, alpha, beta, gamma, iterations):
    stability_list = []
    for i in range(iterations):
        m = build_food_web(S, mass_ratio_model, alpha, beta, gamma)
        s = web_stability
        stability_list.append(s)
    number_stable = stability_list.count(True)
    percent_stable = number_stable / iterations
    return percent_stable 


# In[48]:


m = build_food_web(S, mass_ratio_model, alpha_bm, beta_bm, gamma_bm)


# In[49]:


web_stability(m)


# In[55]:


psw = percent_stable_webs(S, alpha_bm, beta_bm, gamma_bm, 1000)


# In[56]:


psw

