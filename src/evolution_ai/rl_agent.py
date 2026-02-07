"""
Reinforcement Learning agent for simulating HIV viral evolution.

This module implements an RL agent that learns to evolve the virus to survive
under different drug pressures and immune responses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from typing import List, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass


@dataclass
class VirusState:
    """Represents the state of the virus in the environment."""
    viral_load: float  # Current viral load
    cd4_count: float  # CD4+ T cell count
    drug_concentrations: List[float]  # Concentrations of different drugs
    immune_response: float  # Strength of immune response
    resistance_profile: Dict[str, float]  # Resistance to different drugs
    mutation_rate: float  # Current mutation rate
    replication_capacity: float  # Ability to replicate despite mutations


class HIVEnvironment(gym.Env):
    """
    Custom gym environment for HIV evolution simulation.
    
    The environment simulates the interaction between HIV and the host,
    including drug treatments and immune responses.
    """
    
    def __init__(self, initial_viral_load: float = 1e5, initial_cd4: float = 500,
                 num_drugs: int = 4):
        super(HIVEnvironment, self).__init__()
        
        self.initial_viral_load = initial_viral_load
        self.initial_cd4 = initial_cd4
        self.num_drugs = num_drugs
        
        # Action space: which mutation to acquire (or no mutation)
        # +1 for no mutation action
        self.action_space = spaces.Discrete(20 + 1)  # 20 common mutations + no mutation
        
        # Observation space: virus state vector
        obs_dim = 3 + num_drugs + 20 + 2  # viral_load, cd4, immune_response, drugs, resistances, mutation_rate, replication_capacity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Common HIV mutations (simplified)
        self.mutations = [
            'K103N', 'K65R', 'M184V', 'L90M', 'Y181C',
            'G190A', 'V106A', 'E138K', 'K219Q', 'M41L',
            'M184I', 'T215Y', 'K70R', 'D67N', 'L210W',
            'T215F', 'M41L', 'D67G', 'K103S', 'Y188L'
        ]
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.viral_load = self.initial_viral_load
        self.cd4_count = self.initial_cd4
        self.drug_concentrations = [0.0] * self.num_drugs
        self.immune_response = 0.1
        self.resistance_profile = {mut: 0.0 for mut in self.mutations}
        self.mutation_rate = 0.001
        self.replication_capacity = 1.0
        self.step_count = 0
        self.max_steps = 100
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation vector."""
        resistances = [self.resistance_profile[mut] for mut in self.mutations]
        obs = np.array([
            np.log(self.viral_load + 1),  # Log-transformed viral load
            self.cd4_count / 1000.0,      # Normalized CD4 count
            self.immune_response,
            *self.drug_concentrations,    # Drug concentrations
            *resistances,                 # Resistance levels
            self.mutation_rate,
            self.replication_capacity
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Apply action (acquire mutation)
        if action < len(self.mutations):
            mutation = self.mutations[action]
            self._apply_mutation(mutation)
        
        # Update viral dynamics based on current state
        self._update_dynamics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminal()
        truncated = self.step_count >= self.max_steps
        
        self.step_count += 1
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _apply_mutation(self, mutation: str):
        """Apply a mutation to the virus."""
        # Increase resistance to relevant drugs based on mutation
        mutation_effects = {
            'K103N': {'NNRTI': 0.8},  # Strong resistance to NNRTIs
            'K65R': {'NRTI': 0.6},    # Resistance to NRTIs
            'M184V': {'NRTI': 0.9},   # High resistance to M184-containing regimens
            'L90M': {'PI': 0.7},      # Resistance to PIs
            'Y181C': {'NNRTI': 0.7},  # Resistance to NNRTIs
            'G190A': {'NNRTI': 0.8},  # Resistance to NNRTIs
            'V106A': {'NNRTI': 0.75}, # Resistance to NNRTIs
            'E138K': {'NNRTI': 0.6},  # Resistance to NNRTIs
            'K219Q': {'NRTI': 0.3},   # Low-level NRTI resistance
            'M41L': {'NRTI': 0.4},    # Thymidine analog mutation
            'M184I': {'NRTI': 0.85},  # Similar to M184V
            'T215Y': {'NRTI': 0.5},   # TAM mutation
            'K70R': {'NRTI': 0.3},    # TAM mutation
            'D67N': {'NRTI': 0.2},    # TAM mutation
            'L210W': {'NRTI': 0.4},   # TAM mutation
            'T215F': {'NRTI': 0.6},   # TAM mutation
            'D67G': {'NRTI': 0.25},   # TAM mutation
            'K103S': {'NNRTI': 0.75}, # Resistance to NNRTIs
            'Y188L': {'NNRTI': 0.85}  # High-level NNRTI resistance
        }
        
        # Apply the mutation effect
        if mutation in mutation_effects:
            for drug_class, resistance_increase in mutation_effects[mutation].items():
                # Update resistance for drugs in this class
                for i, drug_conc in enumerate(self.drug_concentrations):
                    if drug_conc > 0:  # If drug is present
                        # Simplified mapping: assume first 2 drugs are NRTI, next 2 are NNRTI
                        if i < 2 and drug_class == 'NRTI':
                            self.resistance_profile[mutation] = min(1.0, 
                                self.resistance_profile[mutation] + resistance_increase * 0.5)
                        elif i >= 2 and drug_class == 'NNRTI':
                            self.resistance_profile[mutation] = min(1.0, 
                                self.resistance_profile[mutation] + resistance_increase * 0.5)
                        elif i == 2 and drug_class == 'PI':
                            self.resistance_profile[mutation] = min(1.0, 
                                self.resistance_profile[mutation] + resistance_increase * 0.5)
        
        # Update replication capacity (mutations often come with fitness cost)
        self.replication_capacity *= 0.95  # Fitness cost of mutation
    
    def _update_dynamics(self):
        """Update viral and immune dynamics."""
        # Calculate overall effectiveness of treatment
        treatment_effectiveness = 0.0
        for i, drug_conc in enumerate(self.drug_concentrations):
            if drug_conc > 0:
                # Calculate resistance-adjusted drug efficacy
                resistance_sum = sum(
                    self.resistance_profile[mut] 
                    for mut in self.mutations 
                    if self.resistance_profile[mut] > 0.1
                )
                
                # Higher resistance reduces drug efficacy
                adjusted_efficacy = drug_conc * (1 - resistance_sum / len(self.mutations))
                treatment_effectiveness += min(adjusted_efficacy, drug_conc)
        
        treatment_effectiveness = min(treatment_effectiveness, 1.0)
        
        # Update viral load based on treatment and immune response
        base_replication = self.replication_capacity * 2.0  # Base replication rate
        treatment_suppression = treatment_effectiveness * 0.9  # 90% suppression per effective drug
        immune_suppression = self.immune_response * 0.3  # Immune system suppression
        
        net_growth_rate = base_replication * (1 - treatment_suppression) * (1 - immune_suppression)
        
        # Update viral load with logistic growth
        max_viral_load = 1e6
        growth_term = net_growth_rate * (1 - self.viral_load / max_viral_load)
        self.viral_load *= (1 + growth_term)
        self.viral_load = max(1, min(self.viral_load, max_viral_load))
        
        # Update CD4 count based on viral load
        cd4_decline = min(0.1 * (self.viral_load / 1e5), 0.5)  # Max 50% decline
        self.cd4_count *= (1 - cd4_decline)
        self.cd4_count = max(10, min(self.cd4_count, 1200))  # Keep within bounds
        
        # Update immune response based on viral load
        self.immune_response = min(1.0, self.immune_response + 0.01 * (self.viral_load / 1e5))
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state."""
        # Reward for maintaining high viral load (virus survival)
        viral_reward = np.log(self.viral_load / self.initial_viral_load)
        
        # Penalty for low CD4 count (but virus benefits from destroying CD4 cells)
        cd4_penalty = -max(0, (500 - self.cd4_count) / 500)
        
        # Reward for acquiring resistance mutations
        resistance_reward = sum(self.resistance_profile.values()) * 10
        
        # Penalty for reduced replication capacity
        fitness_penalty = -(1.0 - self.replication_capacity) * 20
        
        # Total reward
        reward = viral_reward + resistance_reward + fitness_penalty
        
        return reward
    
    def _is_terminal(self) -> bool:
        """Check if the episode is terminal."""
        # Episode ends if CD4 count drops too low or viral load becomes negligible
        return self.cd4_count < 20 or self.viral_load < 50


class ViralEvolutionNetwork(nn.Module):
    """
    Neural network for the viral evolution policy.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 21):
        super(ViralEvolutionNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class ViralEvolutionAgent:
    """
    Reinforcement Learning agent for evolving HIV virus.
    
    This agent learns to acquire beneficial mutations to survive under
    different drug pressures and immune responses.
    """
    
    def __init__(self, env: HIVEnvironment, learning_rate: float = 0.001):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network dimensions
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n  # Number of possible mutations + no mutation
        
        # Policy network
        self.policy_network = ViralEvolutionNetwork(input_size, 256, output_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.eps = np.finfo(np.float32).eps.item()  # Small value for numerical stability
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select an action based on the current policy.

        Args:
            state: Current state of the environment

        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get action probabilities from policy network
        action_probs = torch.softmax(self.policy_network(state_tensor), dim=1)
        dist = Categorical(action_probs)

        # Sample action from distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob  # Return the tensor, not the item for backprop
    
    def update_policy(self, log_probs: List[torch.Tensor], rewards: List[float]) -> float:
        """
        Update the policy using REINFORCE algorithm.

        Args:
            log_probs: Log probabilities of taken actions
            rewards: Rewards received at each step

        Returns:
            Loss value for monitoring
        """
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # Compute loss - log_probs are already tensors
        loss = -(torch.stack(log_probs) * returns.detach()).sum()

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, episodes: int = 1000, render: bool = False) -> List[float]:
        """
        Train the agent for a number of episodes.
        
        Args:
            episodes: Number of training episodes
            render: Whether to render the environment
            
        Returns:
            List of episode rewards for monitoring
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            
            done = False
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
                done = terminated or truncated
            
            # Update policy after each episode
            loss = self.update_policy(log_probs, rewards)
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Last Loss: {loss:.4f}")
        
        return episode_rewards
    
    def test(self, episodes: int = 10) -> List[Dict[str, Any]]:
        """
        Test the trained agent.
        
        Args:
            episodes: Number of test episodes
            
        Returns:
            List of episode statistics
        """
        results = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            viral_trajectory = []
            cd4_trajectory = []
            
            done = False
            while not done:
                action, _ = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                
                total_reward += reward
                steps += 1
                viral_trajectory.append(self.env.viral_load)
                cd4_trajectory.append(self.env.cd4_count)
                
                done = terminated or truncated
            
            results.append({
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'final_viral_load': self.env.viral_load,
                'final_cd4_count': self.env.cd4_count,
                'viral_trajectory': viral_trajectory,
                'cd4_trajectory': cd4_trajectory
            })
        
        return results


def run_evolution_simulation():
    """
    Run a complete viral evolution simulation.
    """
    print("Starting HIV Viral Evolution Simulation...")
    
    # Create environment
    env = HIVEnvironment(initial_viral_load=1e5, initial_cd4=500, num_drugs=3)
    
    # Create agent
    agent = ViralEvolutionAgent(env, learning_rate=0.001)
    
    print("Training agent...")
    # Train the agent
    rewards = agent.train(episodes=500)
    
    print("\nTesting trained agent...")
    # Test the trained agent
    test_results = agent.test(episodes=5)
    
    for result in test_results:
        print(f"Test Episode {result['episode']}: "
              f"Reward={result['total_reward']:.2f}, "
              f"Steps={result['steps']}, "
              f"Final VL={result['final_viral_load']:.2e}, "
              f"Final CD4={result['final_cd4_count']:.1f}")
    
    return agent, test_results


if __name__ == "__main__":
    # Example usage
    agent, results = run_evolution_simulation()
    
    print("\nSimulation completed!")