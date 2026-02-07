"""
Realistic mutation rates and effects module for Project Genesis-HIV.

This module provides realistic models for HIV mutation rates, effects,
and resistance patterns based on clinical and experimental data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class MutationType(Enum):
    """Types of mutations that can occur in HIV."""
    POINT_MUTATION = "point_mutation"
    DELETION = "deletion"
    INSERTION = "insertion"
    RECOMBINATION = "recombination"


@dataclass
class MutationEffect:
    """Represents the effect of a mutation."""
    mutation_code: str
    gene: str
    position: int
    wildtype_aa: str
    mutant_aa: str
    fitness_cost: float  # Reduction in replicative capacity (0-1)
    resistance_fold_change: Dict[str, float]  # Drug resistance fold changes
    clinical_significance: str  # Major/minor/accessory
    prevalence: float  # Global prevalence (0-1)
    geographic_distribution: Dict[str, float]  # Regional prevalence
    epistatic_effects: List[str]  # Other mutations that affect this mutation's effect


@dataclass
class MutationRate:
    """Represents mutation rates for different contexts."""
    context: str  # Reverse transcription, integration, etc.
    base_rate: float  # Per nucleotide per replication
    hotspots: Dict[int, float]  # Position-specific rates
    sequence_context_effects: Dict[str, float]  # Effects of flanking sequences
    treatment_effects: Dict[str, float]  # How treatments affect rates


class MutationEngine:
    """
    A class to handle realistic HIV mutation modeling.
    
    This includes:
    - Accurate mutation rates based on RT fidelity
    - Fitness costs of mutations
    - Resistance profiles
    - Epistatic interactions
    - Geographic variation
    """
    
    def __init__(self):
        """Initialize the MutationEngine with realistic parameters."""
        self.mutation_rates = self._initialize_mutation_rates()
        self.mutation_effects = self._initialize_mutation_effects()
        self.epistatic_interactions = self._initialize_epistatic_interactions()
        
        # HIV-specific parameters
        self.genome_length = 9749  # Average HIV-1 genome length
        self.replication_rate = 1e10  # Virions produced per day in untreated infection
        self.target_cell_turnover = 1e8  # Target cells infected per day
        
        # Treatment-specific mutation rate modifiers
        self.treatment_modifiers = {
            "none": 1.0,
            "nrti_only": 0.7,
            "nnrti_only": 0.8,
            "pi_only": 0.9,
            "combined": 0.5
        }
    
    def _initialize_mutation_rates(self) -> Dict[str, MutationRate]:
        """Initialize realistic mutation rates based on literature."""
        return {
            "reverse_transcription": MutationRate(
                context="reverse_transcription",
                base_rate=1.6e-5,  # Per nucleotide per replication cycle
                hotspots={
                    184: 5.0e-5,  # M184V hotspot
                    103: 3.0e-5,  # K103N hotspot
                    181: 2.5e-5,  # Y181C hotspot
                },
                sequence_context_effects={
                    "AAA": 1.5,  # More prone to mutations
                    "TTT": 1.3,
                    "ATA": 1.2,
                },
                treatment_effects={
                    "3TC": 0.8,  # Lamivudine reduces M184V rate
                    "AZT": 0.9,  # Zidovudine affects certain sites
                }
            ),
            "integration": MutationRate(
                context="integration",
                base_rate=1e-7,  # Much lower than RT errors
                hotspots={},
                sequence_context_effects={},
                treatment_effects={}
            )
        }
    
    def _initialize_mutation_effects(self) -> Dict[str, MutationEffect]:
        """Initialize known mutation effects based on clinical data."""
        return {
            # Major resistance mutations
            "K103N": MutationEffect(
                mutation_code="K103N",
                gene="RT",
                position=103,
                wildtype_aa="K",
                mutant_aa="N",
                fitness_cost=0.1,  # 10% reduction in replication
                resistance_fold_change={
                    "NVP": 100.0,
                    "EFV": 100.0,
                    "ETR": 10.0,
                    "RPV": 50.0
                },
                clinical_significance="major",
                prevalence=0.15,  # 15% global prevalence
                geographic_distribution={
                    "Sub-Saharan Africa": 0.18,
                    "North America": 0.12,
                    "Europe": 0.14
                },
                epistatic_effects=["L100I", "V106A"]
            ),
            "M184V": MutationEffect(
                mutation_code="M184V",
                gene="RT",
                position=184,
                wildtype_aa="M",
                mutant_aa="V",
                fitness_cost=0.3,  # Significant fitness cost
                resistance_fold_change={
                    "3TC": 100.0,
                    "FTC": 100.0,
                    "ddI": 5.0,
                    "ABC": 2.0
                },
                clinical_significance="major",
                prevalence=0.08,
                geographic_distribution={
                    "Sub-Saharan Africa": 0.10,
                    "North America": 0.06,
                    "Europe": 0.07
                },
                epistatic_effects=["M184I"]
            ),
            "K65R": MutationEffect(
                mutation_code="K65R",
                gene="RT",
                position=65,
                wildtype_aa="K",
                mutant_aa="R",
                fitness_cost=0.15,
                resistance_fold_change={
                    "TDF": 3.0,
                    "ABC": 5.0,
                    "3TC": 4.0,
                    "FTC": 5.0,
                    "D4T": 2.0
                },
                clinical_significance="major",
                prevalence=0.03,
                geographic_distribution={
                    "Sub-Saharan Africa": 0.02,
                    "North America": 0.04,
                    "Europe": 0.03
                },
                epistatic_effects=["S68G", "E44D"]
            ),
            "L90M": MutationEffect(
                mutation_code="L90M",
                gene="PR",
                position=90,
                wildtype_aa="L",
                mutant_aa="M",
                fitness_cost=0.05,
                resistance_fold_change={
                    "IDV": 15.0,
                    "NFV": 10.0,
                    "SQV": 8.0,
                    "APV": 12.0,
                    "LPV": 5.0
                },
                clinical_significance="major",
                prevalence=0.07,
                geographic_distribution={
                    "Sub-Saharan Africa": 0.05,
                    "North America": 0.08,
                    "Europe": 0.09
                },
                epistatic_effects=["M46I", "I54V"]
            ),
            # Minor/accessory mutations
            "M46I": MutationEffect(
                mutation_code="M46I",
                gene="PR",
                position=46,
                wildtype_aa="M",
                mutant_aa="I",
                fitness_cost=0.02,
                resistance_fold_change={
                    "IDV": 3.0,
                    "NFV": 2.5,
                    "APV": 2.0
                },
                clinical_significance="minor",
                prevalence=0.12,
                geographic_distribution={
                    "Sub-Saharan Africa": 0.10,
                    "North America": 0.14,
                    "Europe": 0.13
                },
                epistatic_effects=["L90M", "I54V"]
            )
        }
    
    def _initialize_epistatic_interactions(self) -> Dict[str, Dict[str, float]]:
        """Initialize known epistatic interactions between mutations."""
        return {
            "K103N": {
                "L100I": 0.7,  # L100I reduces fitness cost of K103N
                "V106A": 0.8,
            },
            "M184V": {
                "K65R": 0.6,  # K65R partially compensates for M184V fitness cost
                "L74V": 0.7,
            },
            "K65R": {
                "M184V": 0.5,  # M184V partially compensates for K65R fitness cost
                "S68G": 0.9,  # S68G enhances K65R resistance
            },
            "L90M": {
                "M46I": 1.5,  # M46I enhances resistance of L90M
                "I54V": 1.4,
            }
        }
    
    def calculate_mutation_probability(self, position: int, context: str = "reverse_transcription", 
                                     treatment_regimen: List[str] = None) -> float:
        """
        Calculate the probability of mutation at a specific position.
        
        Args:
            position: Genomic position (1-indexed)
            context: Context of mutation (reverse_transcription, integration, etc.)
            treatment_regimen: Current treatment regimen
            
        Returns:
            Probability of mutation at this position
        """
        if context not in self.mutation_rates:
            raise ValueError(f"Unknown context: {context}")
        
        rate_info = self.mutation_rates[context]
        prob = rate_info.base_rate
        
        # Apply hotspot effects
        if position in rate_info.hotspots:
            prob = rate_info.hotspots[position]
        
        # Apply sequence context effects (simplified)
        # In a real implementation, we would look at the actual sequence
        prob *= 1.0  # Placeholder for sequence context
        
        # Apply treatment effects
        if treatment_regimen:
            for drug in treatment_regimen:
                if drug in rate_info.treatment_effects:
                    prob *= rate_info.treatment_effects[drug]
        
        return min(prob, 1.0)  # Cap at 100%
    
    def generate_mutations(self, sequence_length: int, treatment_regimen: List[str] = None,
                          num_replications: int = 1000) -> List[Tuple[int, str, str]]:
        """
        Generate mutations based on realistic rates and context.
        
        Args:
            sequence_length: Length of sequence to mutate
            treatment_regimen: Current treatment regimen
            num_replications: Number of replication cycles to simulate
            
        Returns:
            List of (position, wildtype, mutant) tuples
        """
        mutations = []
        
        # Calculate effective mutation rate based on treatment
        treatment_key = "none" if not treatment_regimen else "combined"
        rate_modifier = self.treatment_modifiers.get(treatment_key, 1.0)
        
        for pos in range(1, sequence_length + 1):
            # Calculate mutation probability for this position
            mut_prob = self.calculate_mutation_probability(
                pos, "reverse_transcription", treatment_regimen
            ) * rate_modifier
            
            # Determine number of mutations at this position
            expected_mutations = mut_prob * num_replications
            actual_mutations = np.random.poisson(expected_mutations)
            
            for _ in range(int(actual_mutations)):
                # Determine the type of mutation
                mut_type = np.random.choice(
                    ["transition", "transversion", "deletion", "insertion"],
                    p=[0.6, 0.3, 0.05, 0.05]
                )
                
                # For simplicity, we'll just pick a random alternate base
                wildtype_base = np.random.choice(["A", "T", "G", "C"])
                possible_bases = ["A", "T", "G", "C"]
                possible_bases.remove(wildtype_base)
                mutant_base = np.random.choice(possible_bases)
                
                mutations.append((pos, wildtype_base, mutant_base))
        
        return mutations
    
    def calculate_fitness_impact(self, mutation_list: List[str]) -> float:
        """
        Calculate the overall fitness impact of a set of mutations.
        
        Args:
            mutation_list: List of mutation codes (e.g., ["K103N", "M184V"])
            
        Returns:
            Overall fitness (0-1, where 1 is wildtype fitness)
        """
        if not mutation_list:
            return 1.0
        
        # Start with wildtype fitness
        fitness = 1.0
        
        # Apply individual mutation effects
        for mut_code in mutation_list:
            if mut_code in self.mutation_effects:
                effect = self.mutation_effects[mut_code]
                fitness *= (1 - effect.fitness_cost)
        
        # Apply epistatic interactions
        for i, mut1 in enumerate(mutation_list):
            for j, mut2 in enumerate(mutation_list):
                if i != j and mut1 in self.epistatic_interactions:
                    if mut2 in self.epistatic_interactions[mut1]:
                        modifier = self.epistatic_interactions[mut1][mut2]
                        # Adjust fitness based on interaction
                        base_cost = self.mutation_effects[mut1].fitness_cost
                        adjusted_cost = base_cost * modifier
                        fitness = fitness / (1 - base_cost) * (1 - adjusted_cost)
        
        # Ensure fitness is between 0 and 1
        return max(0.0, min(1.0, fitness))
    
    def predict_drug_resistance(self, mutation_list: List[str], drug: str) -> Dict[str, float]:
        """
        Predict drug resistance based on a list of mutations.
        
        Args:
            mutation_list: List of mutation codes
            drug: Drug name to predict resistance for
            
        Returns:
            Dictionary with resistance predictions
        """
        if not mutation_list:
            return {"fold_change": 1.0, "phenotype": "Susceptible", "confidence": 0.9}
        
        total_fold_change = 1.0
        contributing_mutations = []
        
        for mut_code in mutation_list:
            if mut_code in self.mutation_effects:
                effect = self.mutation_effects[mut_code]
                if drug in effect.resistance_fold_change:
                    fold_change = effect.resistance_fold_change[drug]
                    if fold_change > 1.0:  # Only count resistance mutations
                        total_fold_change *= fold_change
                        contributing_mutations.append(mut_code)
        
        # Determine phenotype based on fold change
        if total_fold_change >= 100:
            phenotype = "High-level resistance"
        elif total_fold_change >= 10:
            phenotype = "Intermediate resistance"
        elif total_fold_change >= 3:
            phenotype = "Low-level resistance"
        else:
            phenotype = "Susceptible"
        
        # Confidence based on number of contributing mutations
        confidence = min(0.95, 0.7 + 0.1 * len(contributing_mutations))
        
        return {
            "fold_change": total_fold_change,
            "phenotype": phenotype,
            "contributing_mutations": contributing_mutations,
            "confidence": confidence
        }
    
    def simulate_evolution(self, initial_mutations: List[str], 
                          treatment_history: List[Tuple[List[str], int]],
                          generations: int = 100) -> Dict:
        """
        Simulate viral evolution under treatment pressure.
        
        Args:
            initial_mutations: Starting set of mutations
            treatment_history: List of (treatment_regimen, duration) tuples
            generations: Number of evolutionary generations to simulate
            
        Returns:
            Evolution simulation results
        """
        current_mutations = initial_mutations.copy()
        fitness_trajectory = []
        resistance_trajectories = {}
        mutation_trajectories = []
        
        # Initialize resistance tracking for key drugs
        key_drugs = ["NVP", "EFV", "3TC", "FTC", "TDF", "ABC", "IDV", "NFV"]
        for drug in key_drugs:
            resistance_trajectories[drug] = []
        
        current_generation = 0
        treatment_idx = 0
        treatment_remaining = treatment_history[0][1] if treatment_history else generations
        
        for gen in range(generations):
            # Update treatment if needed
            if treatment_remaining <= 0 and treatment_idx < len(treatment_history) - 1:
                treatment_idx += 1
                treatment_remaining = treatment_history[treatment_idx][1]
            
            current_treatment = treatment_history[treatment_idx][0] if treatment_history else []
            
            # Calculate current fitness
            current_fitness = self.calculate_fitness_impact(current_mutations)
            fitness_trajectory.append(current_fitness)
            
            # Track resistance to key drugs
            for drug in key_drugs:
                resistance_result = self.predict_drug_resistance(current_mutations, drug)
                resistance_trajectories[drug].append(resistance_result["fold_change"])
            
            # Generate new mutations based on current treatment
            # This is a simplified model - in reality, mutation generation would be more complex
            if random.random() < (1 - current_fitness) * 0.1:  # Higher mutation rate when fitness is low
                # Add a random mutation that might improve fitness under current treatment
                possible_mutations = [k for k, v in self.mutation_effects.items() 
                                    if any(drug in v.resistance_fold_change for drug in current_treatment)]
                
                if possible_mutations and random.random() < 0.3:  # 30% chance to acquire resistance
                    new_mutation = random.choice(possible_mutations)
                    if new_mutation not in current_mutations:
                        current_mutations.append(new_mutation)
            
            mutation_trajectories.append(current_mutations.copy())
            treatment_remaining -= 1
        
        return {
            "fitness_trajectory": fitness_trajectory,
            "resistance_trajectories": resistance_trajectories,
            "mutation_trajectories": mutation_trajectories,
            "final_mutations": current_mutations,
            "final_fitness": self.calculate_fitness_impact(current_mutations)
        }


def run_mutation_demo():
    """Demo function showing how to use the MutationEngine."""
    print("Starting HIV Mutation Modeling Demo...")
    
    # Initialize the engine
    mut_engine = MutationEngine()
    
    # Calculate mutation probability at specific positions
    print("\nMutation probabilities:")
    for pos in [65, 103, 184, 90]:
        prob = mut_engine.calculate_mutation_probability(pos)
        print(f"  Position {pos}: {prob:.2e} per replication")
    
    # Generate some mutations
    print("\nGenerating mutations...")
    mutations = mut_engine.generate_mutations(1000, num_replications=100)
    print(f"Generated {len(mutations)} mutations in 1000 bp sequence")
    
    # Calculate fitness impact
    test_mutations = ["K103N", "M184V"]
    fitness = mut_engine.calculate_fitness_impact(test_mutations)
    print(f"\nFitness with {test_mutations}: {fitness:.3f}")
    
    # Predict drug resistance
    print("\nDrug resistance predictions:")
    for drug in ["NVP", "3TC", "TDF", "IDV"]:
        resistance = mut_engine.predict_drug_resistance(test_mutations, drug)
        print(f"  {drug}: {resistance['fold_change']:.1f}x, {resistance['phenotype']}")
    
    # Simulate evolution
    print("\nSimulating viral evolution...")
    treatment_history = [
        (["3TC", "AZT", "NVP"], 50),  # Initial treatment
        (["ABC", "TDF", "EFV"], 50)   # Switch after 50 generations
    ]
    
    evolution_results = mut_engine.simulate_evolution(
        initial_mutations=["M184V"],
        treatment_history=treatment_history,
        generations=100
    )
    
    print(f"Final mutations: {evolution_results['final_mutations']}")
    print(f"Final fitness: {evolution_results['final_fitness']:.3f}")
    print(f"Final NVP resistance: {evolution_results['resistance_trajectories']['NVP'][-1]:.1f}x")
    print(f"Final 3TC resistance: {evolution_results['resistance_trajectories']['3TC'][-1]:.1f}x")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_mutation_demo()