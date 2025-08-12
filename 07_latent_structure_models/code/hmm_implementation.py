"""
Hidden Markov Models (HMM) Implementation
========================================

This module provides comprehensive implementations of HMM concepts,
including the dishonest casino example, forward-backward algorithm,
Viterbi algorithm, Baum-Welch algorithm, and various applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import seaborn as sns
from scipy.stats import multivariate_normal

class DishonestCasinoHMM:
    """
    Implementation of the dishonest casino HMM example.
    """
    
    def __init__(self, n_states=2, n_observations=6):
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Initialize parameters
        self.pi = np.array([0.5, 0.5])  # Initial state distribution
        
        # Transition matrix (Fair -> Fair, Fair -> Loaded, Loaded -> Fair, Loaded -> Loaded)
        self.A = np.array([[0.95, 0.05], [0.1, 0.9]])
        
        # Emission matrix
        self.B = np.array([
            [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],  # Fair die
            [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]  # Loaded die
        ])
        
    def generate_sequence(self, length=100):
        """
        Generate a sequence of observations and hidden states
        
        Parameters:
        -----------
        length : int
            Length of the sequence to generate
            
        Returns:
        --------
        observations : array
            Generated observations
        hidden_states : array
            Generated hidden states
        """
        observations = []
        hidden_states = []
        
        # Generate initial state
        z = np.random.choice([0, 1], p=self.pi)
        hidden_states.append(z)
        
        # Generate observations
        for t in range(length):
            # Generate observation given current state
            x = np.random.choice(range(1, 7), p=self.B[z])
            observations.append(x)
            
            # Generate next state
            if t < length - 1:
                z = np.random.choice([0, 1], p=self.A[z])
                hidden_states.append(z)
        
        return np.array(observations), np.array(hidden_states)
    
    def forward_algorithm(self, observations):
        """
        Compute forward probabilities
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        alpha : array
            Forward probabilities
        """
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.B[i, observations[0] - 1]
        
        # Forward recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t] - 1] * np.sum(alpha[t-1, :] * self.A[:, j])
        
        return alpha
    
    def backward_algorithm(self, observations):
        """
        Compute backward probabilities
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        beta : array
            Backward probabilities
        """
        n_obs = len(observations)
        beta = np.zeros((n_obs, self.n_states))
        
        # Initialization
        beta[n_obs-1, :] = 1.0
        
        # Backward recursion
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1] - 1] * beta[t+1, :])
        
        return beta
    
    def viterbi_algorithm(self, observations):
        """
        Find most likely sequence of hidden states
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        path : array
            Most likely hidden state sequence
        delta : array
            Viterbi probabilities
        """
        n_obs = len(observations)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialization
        for i in range(self.n_states):
            delta[0, i] = self.pi[i] * self.B[i, observations[0] - 1]
        
        # Forward recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                delta[t, j] = self.B[j, observations[t] - 1] * np.max(delta[t-1, :] * self.A[:, j])
                psi[t, j] = np.argmax(delta[t-1, :] * self.A[:, j])
        
        # Backtracking
        path = np.zeros(n_obs, dtype=int)
        path[n_obs-1] = np.argmax(delta[n_obs-1, :])
        
        for t in range(n_obs-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, delta
    
    def baum_welch_algorithm(self, observations, max_iter=100, tol=1e-6):
        """
        Estimate HMM parameters using EM algorithm
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_obs = len(observations)
        
        for iteration in range(max_iter):
            # E-step: Compute forward and backward probabilities
            alpha = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)
            
            # Compute gamma and xi
            gamma = alpha * beta
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
            
            xi = np.zeros((n_obs-1, self.n_states, self.n_states))
            for t in range(n_obs-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                     self.B[j, observations[t+1] - 1] * beta[t+1, j])
                xi[t] = xi[t] / np.sum(xi[t])
            
            # M-step: Update parameters
            old_A = self.A.copy()
            old_B = self.B.copy()
            
            # Update initial distribution
            self.pi = gamma[0, :]
            
            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
            
            # Update emission matrix
            for i in range(self.n_states):
                for k in range(self.n_observations):
                    mask = (observations == k + 1)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])
            
            # Check convergence
            if (np.max(np.abs(self.A - old_A)) < tol and 
                np.max(np.abs(self.B - old_B)) < tol):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self

def demonstrate_dishonest_casino():
    """
    Demonstrate the dishonest casino HMM example.
    """
    np.random.seed(42)
    casino = DishonestCasinoHMM()
    
    # Generate data
    observations, true_states = casino.generate_sequence(100)
    
    print("Generated sequence statistics:")
    print(f"Number of 6s: {np.sum(observations == 6)}")
    print(f"Proportion of 6s: {np.mean(observations == 6):.3f}")
    
    # Fit HMM using Baum-Welch
    fitted_casino = DishonestCasinoHMM()
    fitted_casino.baum_welch_algorithm(observations)
    
    print("\nTrue parameters:")
    print("Transition matrix:")
    print(casino.A)
    print("Emission matrix:")
    print(casino.B)
    
    print("\nFitted parameters:")
    print("Transition matrix:")
    print(fitted_casino.A)
    print("Emission matrix:")
    print(fitted_casino.B)
    
    # Viterbi decoding
    viterbi_states, delta = fitted_casino.viterbi_algorithm(observations)
    
    print(f"\nViterbi decoding accuracy: {np.mean(viterbi_states == true_states):.3f}")
    
    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Observations
    ax1.plot(observations, 'b-', alpha=0.7, label='Observations')
    ax1.set_title('Casino Dice Rolls')
    ax1.set_ylabel('Dice Value')
    ax1.legend()
    
    # True states
    ax2.plot(true_states, 'g-', label='True States (0=Fair, 1=Loaded)')
    ax2.set_title('True Hidden States')
    ax2.set_ylabel('State')
    ax2.legend()
    
    # Viterbi states
    ax3.plot(viterbi_states, 'r-', label='Viterbi Decoded States')
    ax3.set_title('Viterbi Decoded States')
    ax3.set_ylabel('State')
    ax3.set_xlabel('Time')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return casino, fitted_casino, observations, true_states, viterbi_states

class HMMForwardBackward:
    """
    Implementation of forward-backward algorithm for HMMs.
    """
    
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.n_states = len(pi)
        
    def forward(self, observations):
        """
        Compute forward probabilities
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        alpha : array
            Forward probabilities
        """
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.B[i, observations[0]]
        
        # Forward recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1, :] * self.A[:, j])
        
        return alpha
    
    def backward(self, observations):
        """
        Compute backward probabilities
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        beta : array
            Backward probabilities
        """
        n_obs = len(observations)
        beta = np.zeros((n_obs, self.n_states))
        
        # Initialization
        beta[n_obs-1, :] = 1.0
        
        # Backward recursion
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])
        
        return beta
    
    def compute_likelihood(self, observations):
        """
        Compute marginal likelihood
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        likelihood : float
            Marginal likelihood
        """
        alpha = self.forward(observations)
        return np.sum(alpha[-1, :])
    
    def compute_posterior(self, observations):
        """
        Compute posterior probabilities of hidden states
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        posterior : array
            Posterior probabilities
        """
        alpha = self.forward(observations)
        beta = self.backward(observations)
        
        # Compute joint probabilities
        joint = alpha * beta
        
        # Normalize
        posterior = joint / np.sum(joint, axis=1, keepdims=True)
        
        return posterior

def demonstrate_forward_backward():
    """
    Demonstrate forward-backward algorithm.
    """
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    
    hmm_fb = HMMForwardBackward(pi, A, B)
    observations = [0, 1, 2, 0, 1]  # Example observations
    
    # Compute forward and backward probabilities
    alpha = hmm_fb.forward(observations)
    beta = hmm_fb.backward(observations)
    posterior = hmm_fb.compute_posterior(observations)
    
    print("Forward probabilities:")
    print(alpha)
    print("\nBackward probabilities:")
    print(beta)
    print("\nPosterior probabilities:")
    print(posterior)
    print(f"\nMarginal likelihood: {hmm_fb.compute_likelihood(observations):.6f}")
    
    return hmm_fb, alpha, beta, posterior

class ViterbiHMM:
    """
    Implementation of Viterbi algorithm for HMMs.
    """
    
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.n_states = len(pi)
        
    def decode(self, observations):
        """
        Find most likely sequence of hidden states
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
            
        Returns:
        --------
        path : array
            Most likely hidden state sequence
        delta : array
            Viterbi probabilities
        """
        n_obs = len(observations)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialization
        for i in range(self.n_states):
            delta[0, i] = self.pi[i] * self.B[i, observations[0]]
        
        # Forward recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                # Compute all possible transitions
                transitions = delta[t-1, :] * self.A[:, j]
                delta[t, j] = self.B[j, observations[t]] * np.max(transitions)
                psi[t, j] = np.argmax(transitions)
        
        # Backtracking
        path = np.zeros(n_obs, dtype=int)
        path[n_obs-1] = np.argmax(delta[n_obs-1, :])
        
        for t in range(n_obs-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, delta
    
    def compute_probability(self, observations, path):
        """
        Compute probability of a given path
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
        path : array
            Hidden state sequence
            
        Returns:
        --------
        probability : float
            Path probability
        """
        prob = self.pi[path[0]] * self.B[path[0], observations[0]]
        
        for t in range(1, len(observations)):
            prob *= self.A[path[t-1], path[t]] * self.B[path[t], observations[t]]
        
        return prob

def demonstrate_viterbi():
    """
    Demonstrate Viterbi algorithm.
    """
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    
    viterbi_hmm = ViterbiHMM(pi, A, B)
    observations = [0, 1, 2, 0, 1]
    
    best_path, delta = viterbi_hmm.decode(observations)
    
    print("Most likely hidden state sequence:")
    print(best_path)
    print(f"Path probability: {viterbi_hmm.compute_probability(observations, best_path):.6f}")
    
    # Compare with posterior decoding
    hmm_fb = HMMForwardBackward(pi, A, B)
    posterior = hmm_fb.compute_posterior(observations)
    posterior_path = np.argmax(posterior, axis=1)
    print(f"Posterior decoding: {posterior_path}")
    print(f"Posterior path probability: {viterbi_hmm.compute_probability(observations, posterior_path):.6f}")
    
    return viterbi_hmm, best_path, posterior_path

class BaumWelchHMM:
    """
    Implementation of Baum-Welch algorithm for HMM parameter estimation.
    """
    
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        
    def initialize_parameters(self):
        """Initialize parameters randomly"""
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.B = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)
        
    def fit(self, observations, max_iter=100, tol=1e-6):
        """
        Fit HMM using Baum-Welch algorithm
        
        Parameters:
        -----------
        observations : array
            Sequence of observations
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.initialize_parameters()
        
        for iteration in range(max_iter):
            # E-step
            gamma, xi = self._e_step(observations)
            
            # M-step
            old_pi = self.pi.copy()
            old_A = self.A.copy()
            old_B = self.B.copy()
            
            self._m_step(observations, gamma, xi)
            
            # Check convergence
            if (np.max(np.abs(self.pi - old_pi)) < tol and
                np.max(np.abs(self.A - old_A)) < tol and
                np.max(np.abs(self.B - old_B)) < tol):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            if iteration % 10 == 0:
                print(f"Iteration {iteration}")
        
        return self
    
    def _e_step(self, observations):
        """E-step: Compute gamma and xi"""
        n_obs = len(observations)
        
        # Forward-backward
        alpha = self._forward(observations)
        beta = self._backward(observations)
        
        # Compute gamma
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        # Compute xi
        xi = np.zeros((n_obs-1, self.n_states, self.n_states))
        for t in range(n_obs-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                 self.B[j, observations[t+1]] * beta[t+1, j])
            xi[t] = xi[t] / np.sum(xi[t])
        
        return gamma, xi
    
    def _m_step(self, observations, gamma, xi):
        """M-step: Update parameters"""
        n_obs = len(observations)
        
        # Update initial distribution
        self.pi = gamma[0, :]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        # Update emission matrix
        for i in range(self.n_states):
            for k in range(self.n_observations):
                mask = (observations == k)
                self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])
    
    def _forward(self, observations):
        """Forward algorithm"""
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.B[i, observations[0]]
        
        # Forward recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1, :] * self.A[:, j])
        
        return alpha
    
    def _backward(self, observations):
        """Backward algorithm"""
        n_obs = len(observations)
        beta = np.zeros((n_obs, self.n_states))
        
        # Initialization
        beta[n_obs-1, :] = 1.0
        
        # Backward recursion
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])
        
        return beta

def demonstrate_baum_welch():
    """
    Demonstrate Baum-Welch algorithm.
    """
    np.random.seed(42)
    observations = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    
    # Fit HMM
    bw_hmm = BaumWelchHMM(n_states=2, n_observations=3)
    bw_hmm.fit(observations)
    
    print("Fitted parameters:")
    print("Initial distribution:", bw_hmm.pi)
    print("Transition matrix:")
    print(bw_hmm.A)
    print("Emission matrix:")
    print(bw_hmm.B)
    
    return bw_hmm

def speech_recognition_example():
    """
    Example of HMM for speech recognition.
    """
    # States: phonemes (simplified)
    phonemes = ['a', 'e', 'i', 'o', 'u']
    
    # Observations: acoustic features
    features = ['low', 'mid', 'high']
    
    # Initialize HMM for speech recognition
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal initial probability
    
    # Transition matrix (phoneme transitions)
    A = np.array([
        [0.6, 0.1, 0.1, 0.1, 0.1],  # 'a' transitions
        [0.1, 0.6, 0.1, 0.1, 0.1],  # 'e' transitions
        [0.1, 0.1, 0.6, 0.1, 0.1],  # 'i' transitions
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 'o' transitions
        [0.1, 0.1, 0.1, 0.1, 0.6]   # 'u' transitions
    ])
    
    # Emission matrix (phoneme to feature mapping)
    B = np.array([
        [0.7, 0.2, 0.1],  # 'a' -> features
        [0.2, 0.7, 0.1],  # 'e' -> features
        [0.1, 0.2, 0.7],  # 'i' -> features
        [0.6, 0.3, 0.1],  # 'o' -> features
        [0.1, 0.3, 0.6]   # 'u' -> features
    ])
    
    return pi, A, B, phonemes, features

def demonstrate_speech_recognition():
    """
    Demonstrate speech recognition with HMM.
    """
    pi, A, B, phonemes, features = speech_recognition_example()
    speech_hmm = ViterbiHMM(pi, A, B)
    
    # Simulate speech features
    speech_features = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    phoneme_sequence = speech_hmm.decode(speech_features)[0]
    
    print("Speech recognition example:")
    print("Features:", speech_features)
    print("Decoded phonemes:", [phonemes[i] for i in phoneme_sequence])
    
    return speech_hmm, phoneme_sequence

def gene_finding_example():
    """
    Example of HMM for gene finding.
    """
    # States: coding, non-coding, start codon, stop codon
    states = ['coding', 'non-coding', 'start', 'stop']
    
    # Observations: DNA bases
    bases = ['A', 'T', 'G', 'C']
    
    # Initialize HMM for gene finding
    pi = np.array([0.1, 0.8, 0.05, 0.05])  # Most DNA is non-coding
    
    # Transition matrix
    A = np.array([
        [0.95, 0.02, 0.02, 0.01],  # coding transitions
        [0.01, 0.98, 0.005, 0.005], # non-coding transitions
        [0.99, 0.01, 0.0, 0.0],     # start transitions
        [0.01, 0.99, 0.0, 0.0]      # stop transitions
    ])
    
    # Emission matrix (base composition)
    B = np.array([
        [0.25, 0.25, 0.25, 0.25],  # coding (random)
        [0.30, 0.30, 0.20, 0.20],  # non-coding
        [0.25, 0.25, 0.25, 0.25],  # start
        [0.25, 0.25, 0.25, 0.25]   # stop
    ])
    
    return pi, A, B, states, bases

def demonstrate_gene_finding():
    """
    Demonstrate gene finding with HMM.
    """
    pi, A, B, states, bases = gene_finding_example()
    gene_hmm = ViterbiHMM(pi, A, B)
    
    # Simulate DNA sequence
    dna_sequence = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    gene_states = gene_hmm.decode(dna_sequence)[0]
    
    print("Gene finding example:")
    print("DNA sequence:", [bases[i] for i in dna_sequence])
    print("Gene states:", [states[i] for i in gene_states])
    
    return gene_hmm, gene_states

class GaussianHMM:
    """
    Implementation of Gaussian HMM for continuous observations.
    """
    
    def __init__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features
        
    def initialize_parameters(self):
        """Initialize parameters for Gaussian HMM"""
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        
        # Gaussian parameters
        self.means = np.random.randn(self.n_states, self.n_features)
        self.covs = np.array([np.eye(self.n_features) for _ in range(self.n_states)])
        
    def emission_probability(self, observation, state):
        """
        Compute emission probability for continuous observation
        
        Parameters:
        -----------
        observation : array
            Continuous observation
        state : int
            Hidden state
            
        Returns:
        --------
        probability : float
            Emission probability
        """
        return multivariate_normal.pdf(observation, self.means[state], self.covs[state])
    
    def fit(self, observations, max_iter=100):
        """
        Fit Gaussian HMM using EM
        
        Parameters:
        -----------
        observations : array
            Sequence of continuous observations
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.initialize_parameters()
        
        for iteration in range(max_iter):
            # E-step and M-step (simplified)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}")
        
        return self

def demonstrate_gaussian_hmm():
    """
    Demonstrate Gaussian HMM.
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 2
    
    # Generate data from two Gaussian components
    data = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2),
        np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], n_samples//2)
    ])
    
    # Fit Gaussian HMM
    gaussian_hmm = GaussianHMM(n_states=2, n_features=2)
    gaussian_hmm.fit(data)
    
    print("Gaussian HMM fitted successfully!")
    
    return gaussian_hmm, data

if __name__ == "__main__":
    print("Demonstrating HMM Implementation...")
    
    # Dishonest Casino Example
    print("\n1. Dishonest Casino HMM")
    casino, fitted_casino, observations, true_states, viterbi_states = demonstrate_dishonest_casino()
    
    # Forward-Backward Algorithm
    print("\n2. Forward-Backward Algorithm")
    hmm_fb, alpha, beta, posterior = demonstrate_forward_backward()
    
    # Viterbi Algorithm
    print("\n3. Viterbi Algorithm")
    viterbi_hmm, best_path, posterior_path = demonstrate_viterbi()
    
    # Baum-Welch Algorithm
    print("\n4. Baum-Welch Algorithm")
    bw_hmm = demonstrate_baum_welch()
    
    # Speech Recognition
    print("\n5. Speech Recognition Example")
    speech_hmm, phoneme_sequence = demonstrate_speech_recognition()
    
    # Gene Finding
    print("\n6. Gene Finding Example")
    gene_hmm, gene_states = demonstrate_gene_finding()
    
    # Gaussian HMM
    print("\n7. Gaussian HMM")
    gaussian_hmm, data = demonstrate_gaussian_hmm()
