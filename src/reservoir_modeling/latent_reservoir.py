"""
Enhanced latent reservoir modeling module for Project Genesis-HIV.

This module provides detailed modeling of HIV latent reservoirs,
including establishment, maintenance, reactivation, and therapeutic targeting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class ReservoirType(Enum):
    """Types of latent reservoirs."""
    RESTING_MEMORY_CD4 = "resting_memory_CD4"
    TISSUE_RESIDENT_MEMORY = "tissue_resident_memory"
    FOLLICULAR_HELPER_TCELLS = "follicular_helper_Tcells"
    STEM_CELL_LIKE = "stem_cell_like"
    TISSUE_COMPARTMENTS = "tissue_compartments"


@dataclass
class LatentlyInfectedCell:
    """Represents a latently infected cell."""
    cell_id: str
    reservoir_type: ReservoirType
    integration_site: str  # Genomic location of provirus
    proviral_load: int  # Number of integrated copies
    integration_quality: float  # 0-1 scale (intact vs defective)
    activation_threshold: float  # Signal level required for activation
    decay_rate: float  # Natural decay rate
    proliferation_rate: float  # Homeostatic proliferation rate
    age: int  # Days since infection
    last_reactivated: Optional[int]  # Last reactivation time
    resistance_mutations: List[str]  # Mutations conferring resistance


class LatentReservoirEngine:
    """
    A class to model HIV latent reservoirs with high fidelity.
    
    This includes:
    - Establishment of latent infection
    - Maintenance mechanisms
    - Reactivation dynamics
    - Therapeutic interventions
    - Reservoir decay kinetics
    - Tissue-specific compartments
    """
    
    def __init__(self):
        """Initialize the LatentReservoirEngine."""
        self.reservoirs = {
            ReservoirType.RESTING_MEMORY_CD4: [],
            ReservoirType.TISSUE_RESIDENT_MEMORY: [],
            ReservoirType.FOLLICULAR_HELPER_TCELLS: [],
            ReservoirType.STEM_CELL_LIKE: [],
            ReservoirType.TISSUE_COMPARTMENTS: []
        }
        
        self.total_reservoir_size = 0
        self.reservoir_decay_constants = self._initialize_decay_constants()
        self.reactivation_parameters = self._initialize_reactivation_parameters()
        self.compartmentalization_factors = self._initialize_compartmentalization()
        
        # Biological parameters
        self.hiv_genome_intact_rate = 0.05  # ~5% of integrated genomes are intact
        self.homeostatic_proliferation_rate = 0.01  # ~1% per year
        self.activation_induced_death_rate = 0.8  # ~80% die after activation
        self.baseline_reactivation_rate = 1e-4  # Per cell per day
    
    def _initialize_decay_constants(self) -> Dict[ReservoirType, float]:
        """Initialize decay constants for different reservoir types."""
        return {
            ReservoirType.RESTING_MEMORY_CD4: 0.04,  # ~2-year half-life
            ReservoirType.TISSUE_RESIDENT_MEMORY: 0.02,  # ~4-year half-life
            ReservoirType.FOLLICULAR_HELPER_TCELLS: 0.01,  # ~7-year half-life
            ReservoirType.STEM_CELL_LIKE: 0.001,  # Very long-lived
            ReservoirType.TISSUE_COMPARTMENTS: 0.03  # ~2.3-year half-life
        }
    
    def _initialize_reactivation_parameters(self) -> Dict[str, float]:
        """Initialize parameters for reactivation."""
        return {
            'baseline_reactivation_rate': 1e-4,
            'activation_threshold_min': 0.1,
            'activation_threshold_max': 0.9,
            'tcrc_signal_efficiency': 0.7,
            'cytokine_signal_efficiency': 0.5,
            'histone_modification_efficiency': 0.8,
            'shock_agent_efficiency': 0.9  # For HDAC inhibitors
        }
    
    def _initialize_compartmentalization(self) -> Dict[str, Dict]:
        """Initialize tissue compartmentalization parameters."""
        return {
            'blood': {
                'volume': 5.0,  # Liters
                'cell_density': 800,  # CD4+ T cells per μL
                'turnover_rate': 0.1,  # Per day
                'accessibility': 1.0  # Full drug access
            },
            'lymph_nodes': {
                'volume': 0.5,  # Liters
                'cell_density': 10000,  # Much higher in lymphoid tissue
                'turnover_rate': 0.05,
                'accessibility': 0.8  # Slightly reduced access
            },
            'gut_associated_lymphoid': {
                'volume': 0.3,  # Liters
                'cell_density': 8000,
                'turnover_rate': 0.08,
                'accessibility': 0.6  # Reduced access in gut
            },
            'central_nervous_system': {
                'volume': 0.1,  # Liters
                'cell_density': 100,  # Lower density
                'turnover_rate': 0.01,
                'accessibility': 0.3  # Blood-brain barrier limits access
            }
        }
    
    def establish_latent_infection(self, target_cells: int, 
                                 integration_probability: float = 0.001) -> Dict[ReservoirType, int]:
        """
        Establish latent infection in different reservoir compartments.
        
        Args:
            target_cells: Number of target cells exposed to virus
            integration_probability: Probability of integration per infection
            
        Returns:
            Distribution of latent cells across reservoir types
        """
        # Distribute infections based on known anatomical distribution
        distribution = {
            ReservoirType.RESTING_MEMORY_CD4: 0.60,  # 60% in resting memory CD4+
            ReservoirType.TISSUE_RESIDENT_MEMORY: 0.15,  # 15% in tissue-resident
            ReservoirType.FOLLICULAR_HELPER_TCELLS: 0.10,  # 10% in TFH
            ReservoirType.STEM_CELL_LIKE: 0.05,  # 5% in long-lived cells
            ReservoirType.TISSUE_COMPARTMENTS: 0.10  # 10% in tissues
        }
        
        new_latent_cells = {}
        
        for reservoir_type, proportion in distribution.items():
            # Calculate number of infections in this compartment
            infections = int(target_cells * proportion * integration_probability)
            
            # Create latently infected cells
            latent_cells = []
            for i in range(infections):
                cell_id = f"{reservoir_type.value}_cell_{len(self.reservoirs[reservoir_type]) + i}"
                
                # Determine integration quality (intact vs defective)
                integration_quality = 1.0 if random.random() < self.hiv_genome_intact_rate else 0.2
                
                # Determine activation threshold (varies by cell type)
                threshold_min = self.reactivation_parameters['activation_threshold_min']
                threshold_max = self.reactivation_parameters['activation_threshold_max']
                activation_threshold = random.uniform(threshold_min, threshold_max)
                
                # Determine decay and proliferation rates
                decay_rate = self.reservoir_decay_constants[reservoir_type]
                proliferation_rate = self.homeostatic_proliferation_rate
                
                cell = LatentlyInfectedCell(
                    cell_id=cell_id,
                    reservoir_type=reservoir_type,
                    integration_site=self._generate_integration_site(),
                    proviral_load=random.randint(1, 3),  # Usually 1-3 copies
                    integration_quality=integration_quality,
                    activation_threshold=activation_threshold,
                    decay_rate=decay_rate,
                    proliferation_rate=proliferation_rate,
                    age=0,
                    last_reactivated=None,
                    resistance_mutations=[]
                )
                
                latent_cells.append(cell)
            
            self.reservoirs[reservoir_type].extend(latent_cells)
            new_latent_cells[reservoir_type] = len(latent_cells)
        
        self.total_reservoir_size = sum(len(cells) for cells in self.reservoirs.values())
        
        return new_latent_cells
    
    def _generate_integration_site(self) -> str:
        """Generate a plausible integration site."""
        # In reality, this would be a specific genomic location
        # For simulation, we'll generate a representative string
        chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        chromosome = random.choice(chromosomes)
        position = random.randint(10000, 240000000)  # Within chromosome bounds
        return f"{chromosome}:{position}"
    
    def update_reservoir_dynamics(self, days: int = 1, 
                                immune_activation: float = 0.1,
                                antiretroviral_coverage: float = 0.99) -> Dict:
        """
        Update reservoir dynamics over time.
        
        Args:
            days: Number of days to simulate
            immune_activation: Level of immune activation (0-1)
            antiretroviral_coverage: Effectiveness of ART (0-1)
            
        Returns:
            Summary of reservoir changes
        """
        changes = {
            'new_activations': 0,
            'new_deaths': 0,
            'new_divisions': 0,
            'net_change': 0,
            'compartment_changes': {}
        }
        
        for reservoir_type, cells in self.reservoirs.items():
            compartment_changes = {
                'activations': 0,
                'deaths': 0,
                'divisions': 0,
                'net_change': 0
            }
            
            # Process each cell in the reservoir
            cells_to_remove = []
            cells_to_add = []
            
            for i, cell in enumerate(cells):
                # Update cell age
                cell.age += days
                
                # Check for spontaneous reactivation
                spont_reactivation_prob = self.baseline_reactivation_rate * days
                if random.random() < spont_reactivation_prob:
                    compartment_changes['activations'] += 1
                    cell.last_reactivated = cell.age
                    
                    # Determine if cell survives activation
                    if random.random() > self.activation_induced_death_rate:
                        # Cell survives but transitions out of latency
                        # For simplicity, we'll remove it from the latent pool
                        cells_to_remove.append(i)
                    else:
                        # Cell dies during activation
                        compartment_changes['deaths'] += 1
                        cells_to_remove.append(i)
                
                # Check for activation by immune signals
                elif immune_activation > cell.activation_threshold:
                    activation_prob = (immune_activation - cell.activation_threshold) * \
                                    self.reactivation_parameters['tcrc_signal_efficiency']
                    if random.random() < activation_prob * days:
                        compartment_changes['activations'] += 1
                        cell.last_reactivated = cell.age
                        
                        # Determine if cell survives activation
                        if random.random() > self.activation_induced_death_rate:
                            cells_to_remove.append(i)
                        else:
                            compartment_changes['deaths'] += 1
                            cells_to_remove.append(i)
                
                # Check for natural decay
                else:
                    decay_prob = self.reservoir_decay_constants[reservoir_type] * days
                    if random.random() < decay_prob:
                        compartment_changes['deaths'] += 1
                        cells_to_remove.append(i)
                
                # Check for homeostatic proliferation
                prolif_prob = cell.proliferation_rate * days
                if random.random() < prolif_prob:
                    compartment_changes['divisions'] += 1
                    # Create a daughter cell with similar properties
                    daughter_cell = LatentlyInfectedCell(
                        cell_id=f"{cell.cell_id}_daughter_{cell.age}",
                        reservoir_type=cell.reservoir_type,
                        integration_site=cell.integration_site,  # Same integration site
                        proviral_load=cell.proviral_load,
                        integration_quality=cell.integration_quality,
                        activation_threshold=cell.activation_threshold,
                        decay_rate=cell.decay_rate,
                        proliferation_rate=cell.proliferation_rate,
                        age=0,
                        last_reactivated=cell.last_reactivated,
                        resistance_mutations=cell.resistance_mutations.copy()
                    )
                    cells_to_add.append(daughter_cell)
            
            # Apply changes (process in reverse order to maintain indices)
            for idx in sorted(cells_to_remove, reverse=True):
                del cells[idx]
            
            # Add new cells
            cells.extend(cells_to_add)
            
            compartment_changes['net_change'] = len(cells_to_add) - len(cells_to_remove)
            changes['compartment_changes'][reservoir_type.value] = compartment_changes
            changes['new_activations'] += compartment_changes['activations']
            changes['new_deaths'] += compartment_changes['deaths']
            changes['new_divisions'] += compartment_changes['divisions']
            changes['net_change'] += compartment_changes['net_change']
        
        self.total_reservoir_size = sum(len(cells) for cells in self.reservoirs.values())
        
        return changes
    
    def simulate_shock_and_kill(self, shock_agents: List[str], 
                              kill_efficiency: float = 0.5,
                              duration_days: int = 7) -> Dict:
        """
        Simulate shock and kill intervention.
        
        Args:
            shock_agents: List of shock agents used (e.g., ["Vorinostat", "Romidepsin"])
            kill_efficiency: Efficiency of killing activated cells (0-1)
            duration_days: Duration of intervention
            
        Returns:
            Intervention outcome
        """
        outcome = {
            'cells_activated': 0,
            'cells_eliminated': 0,
            'remaining_reservoir': 0,
            'compartment_outcomes': {}
        }
        
        for reservoir_type, cells in self.reservoirs.items():
            compartment_outcome = {
                'cells_activated': 0,
                'cells_eliminated': 0
            }
            
            # Calculate activation enhancement factor
            activation_enhancement = self.reactivation_parameters['shock_agent_efficiency'] * len(shock_agents)
            
            # Process cells
            cells_to_remove = []
            for i, cell in enumerate(cells):
                # Enhanced activation probability due to shock agents
                enhanced_activation_prob = min(1.0, 
                    cell.activation_threshold + activation_enhancement)
                
                if random.random() < enhanced_activation_prob:
                    compartment_outcome['cells_activated'] += 1
                    
                    # Attempt to eliminate the cell
                    if random.random() < kill_efficiency:
                        compartment_outcome['cells_eliminated'] += 1
                        cells_to_remove.append(i)
            
            # Apply changes
            for idx in sorted(cells_to_remove, reverse=True):
                del cells[idx]
            
            outcome['compartment_outcomes'][reservoir_type.value] = compartment_outcome
            outcome['cells_activated'] += compartment_outcome['cells_activated']
            outcome['cells_eliminated'] += compartment_outcome['cells_eliminated']
        
        self.total_reservoir_size = sum(len(cells) for cells in self.reservoirs.values())
        outcome['remaining_reservoir'] = self.total_reservoir_size
        
        return outcome
    
    def estimate_reservoir_size(self) -> Dict[str, int]:
        """
        Estimate the size of different reservoir compartments.
        
        Returns:
            Reservoir size estimates by compartment
        """
        size_estimate = {}
        
        for reservoir_type, cells in self.reservoirs.items():
            # Count only cells with intact provirus
            intact_cells = sum(1 for cell in cells if cell.integration_quality > 0.5)
            size_estimate[reservoir_type.value] = intact_cells
        
        size_estimate['total_intact'] = sum(size_estimate.values())
        size_estimate['total_all'] = sum(len(cells) for cells in self.reservoirs.values())
        
        return size_estimate
    
    def simulate_analytical_treatment_interruption(self, 
                                                ati_duration_days: int = 180) -> Dict:
        """
        Simulate analytical treatment interruption to measure reservoir reactivation.
        
        Args:
            ati_duration_days: Duration of treatment interruption
            
        Returns:
            ATI outcome with viral rebound data
        """
        # During ATI, no ART coverage
        ati_results = {
            'days_to_rebound': None,
            'rebound_viral_load': None,
            'reservoir_reductions': {},
            'rebound_kinetics': []
        }
        
        # Track reservoir changes during ATI
        for day in range(ati_duration_days):
            changes = self.update_reservoir_dynamics(
                days=1, 
                immune_activation=0.3,  # Higher activation without ART
                antiretroviral_coverage=0.0  # No ART coverage
            )
            
            # Check for viral rebound (simplified model)
            if ati_results['days_to_rebound'] is None:
                # If many activations occurred, assume viral rebound
                if changes['new_activations'] > 10:  # Threshold for rebound
                    ati_results['days_to_rebound'] = day
                    ati_results['rebound_viral_load'] = changes['new_activations'] * 1000  # Simplified
            
            # Store daily kinetics
            daily_reservoir = self.estimate_reservoir_size()
            ati_results['rebound_kinetics'].append({
                'day': day,
                'activated_cells': changes['new_activations'],
                'reservoir_size': daily_reservoir['total_intact']
            })
        
        return ati_results
    
    def predict_reservoir_depletion_kinetics(self, 
                                           intervention_schedule: List[Tuple[str, int, float]]) -> Dict:
        """
        Predict reservoir depletion under intervention schedule.
        
        Args:
            intervention_schedule: List of (intervention_type, start_day, intensity) tuples
            
        Returns:
            Predicted depletion kinetics
        """
        kinetics = {
            'time_points': [],
            'reservoir_sizes': [],
            'interventions_applied': []
        }
        
        current_day = 0
        intervention_idx = 0
        
        # Run simulation for 365 days
        for day in range(365):
            # Check if a new intervention should start
            if (intervention_idx < len(intervention_schedule) and 
                day >= intervention_schedule[intervention_idx][1]):
                
                int_type, start_day, intensity = intervention_schedule[intervention_idx]
                
                if int_type == "shock_and_kill":
                    outcome = self.simulate_shock_and_kill(
                        shock_agents=["Vorinostat"], 
                        kill_efficiency=intensity,
                        duration_days=7
                    )
                    kinetics['interventions_applied'].append({
                        'day': day,
                        'type': int_type,
                        'eliminated': outcome['cells_eliminated'],
                        'efficiency': intensity
                    })
                
                intervention_idx += 1
            
            # Update reservoir dynamics
            self.update_reservoir_dynamics(
                days=1,
                immune_activation=0.1,
                antiretroviral_coverage=0.99  # Assuming ongoing ART
            )
            
            # Record reservoir size
            current_size = self.estimate_reservoir_size()['total_intact']
            kinetics['time_points'].append(day)
            kinetics['reservoir_sizes'].append(current_size)
        
        return kinetics


def run_latent_reservoir_demo():
    """Demo function showing how to use the LatentReservoirEngine."""
    print("Starting HIV Latent Reservoir Modeling Demo...")
    
    # Initialize the engine
    reservoir_engine = LatentReservoirEngine()
    
    # Example 1: Establish latent infection
    print("\n1. Establishing Latent Infection:")
    new_infections = reservoir_engine.establish_latent_infection(
        target_cells=1000000,  # 1 million target cells
        integration_probability=0.001  # 0.1% integration rate
    )
    
    for compartment, count in new_infections.items():
        print(f"   {compartment.value}: {count} new latently infected cells")
    
    initial_size = reservoir_engine.estimate_reservoir_size()
    print(f"   Total intact reservoir: {initial_size['total_intact']} cells")
    
    # Example 2: Update reservoir dynamics over time
    print("\n2. Updating Reservoir Dynamics (30 days):")
    changes = reservoir_engine.update_reservoir_dynamics(
        days=30,
        immune_activation=0.1,
        antiretroviral_coverage=0.99
    )
    
    print(f"   Activations: {changes['new_activations']}")
    print(f"   Deaths: {changes['new_deaths']}")
    print(f"   Divisions: {changes['new_divisions']}")
    print(f"   Net change: {changes['net_change']}")
    
    current_size = reservoir_engine.estimate_reservoir_size()
    print(f"   Current intact reservoir: {current_size['total_intact']} cells")
    
    # Example 3: Shock and kill intervention
    print("\n3. Shock and Kill Intervention:")
    shock_outcome = reservoir_engine.simulate_shock_and_kill(
        shock_agents=["Vorinostat"],
        kill_efficiency=0.3,  # 30% kill efficiency
        duration_days=7
    )
    
    print(f"   Cells activated: {shock_outcome['cells_activated']}")
    print(f"   Cells eliminated: {shock_outcome['cells_eliminated']}")
    print(f"   Remaining reservoir: {shock_outcome['remaining_reservoir']} cells")
    
    # Example 4: Analytical treatment interruption
    print("\n4. Analytical Treatment Interruption Simulation:")
    # Create a new reservoir for this demo
    reservoir_engine_2 = LatentReservoirEngine()
    reservoir_engine_2.establish_latent_infection(500000, 0.001)
    
    ati_results = reservoir_engine_2.simulate_analytical_treatment_interruption(90)
    print(f"   Days to rebound: {ati_results['days_to_rebound']}")
    print(f"   Rebound viral load: {ati_results['rebound_viral_load']}")
    
    # Example 5: Predict depletion kinetics
    print("\n5. Predicting Reservoir Depletion Kinetics:")
    intervention_schedule = [
        ("shock_and_kill", 30, 0.2),   # Week 5
        ("shock_and_kill", 90, 0.25),  # Month 3
        ("shock_and_kill", 180, 0.3)   # Month 6
    ]
    
    depletion_kinetics = reservoir_engine.predict_reservoir_depletion_kinetics(
        intervention_schedule
    )
    
    print(f"   Initial reservoir: {depletion_kinetics['reservoir_sizes'][0]} cells")
    print(f"   Final reservoir: {depletion_kinetics['reservoir_sizes'][-1]} cells")
    print(f"   Reduction: {(1 - depletion_kinetics['reservoir_sizes'][-1]/depletion_kinetics['reservoir_sizes'][0])*100:.1f}%")
    
    if depletion_kinetics['interventions_applied']:
        print(f"   Interventions applied: {len(depletion_kinetics['interventions_applied'])}")
        for interv in depletion_kinetics['interventions_applied']:
            print(f"     Day {interv['day']}: {interv['eliminated']} cells eliminated")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_latent_reservoir_demo()