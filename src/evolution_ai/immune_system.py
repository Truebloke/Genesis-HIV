"""
Immune system modeling module for Project Genesis-HIV.

This module provides realistic modeling of the immune response to HIV infection,
including cellular immunity, humoral immunity, and immune activation dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class ImmuneCellType(Enum):
    """Types of immune cells involved in HIV response."""
    CD4_T_HELPER = "CD4_T_helper"
    CD8_T_CYTOTOXIC = "CD8_T_cytotoxic"
    B_CELL = "B_cell"
    NK_CELL = "NK_cell"
    DENDRITIC_CELL = "dendritic_cell"
    MACROPHAGE = "macrophage"
    MONOCYTE = "monocyte"


@dataclass
class ImmuneResponse:
    """Represents an immune response component."""
    cell_type: ImmuneCellType
    activation_state: float  # 0-1 scale
    proliferation_rate: float
    cytotoxic_activity: float  # For CD8+ T cells
    antibody_production: float  # For B cells
    cytokine_secretion: Dict[str, float]  # Various cytokines
    hla_restriction: Optional[str]  # HLA type restriction


@dataclass
class AntigenPresentingCell:
    """Represents an antigen-presenting cell."""
    cell_type: ImmuneCellType
    mhc_expression: Dict[str, float]  # MHC class I and II expression
    costimulatory_molecules: Dict[str, float]  # CD80, CD86, etc.
    antigen_loading: Dict[str, float]  # Peptides loaded on MHC
    migration_capacity: float  # Ability to migrate to lymph nodes


class ImmuneSystemEngine:
    """
    A class to model the immune system response to HIV.
    
    This includes:
    - CD4+ T cell dynamics
    - CD8+ T cell responses
    - B cell and antibody responses
    - Cytokine networks
    - Immune activation and exhaustion
    - HLA-mediated effects
    """
    
    def __init__(self, hla_type: str = "HLA-A*02:01"):
        """
        Initialize the ImmuneSystemEngine.
        
        Args:
            hla_type: Host HLA type affecting immune response
        """
        self.hla_type = hla_type
        self.immune_cells = {}
        self.cytokines = {}
        self.antibodies = {}
        self.antigen_presenting_cells = {}
        
        # Initialize immune compartments
        self._initialize_immune_compartments()
        
        # HIV-specific immune parameters
        self.viral_antigen_threshold = 100  # Copies/mL to trigger response
        self.immune_activation_max = 10.0   # Maximum activation level
        self.exhaustion_threshold = 0.8     # Activation level leading to exhaustion
        
        # HLA-specific parameters
        self.hla_restrictions = {
            "HLA-A*02:01": ["KIR3DS1", "B*57:01"],
            "HLA-B*57:01": ["protective", "slow_progression"],
            "HLA-B*35:01": ["risk", "fast_progression"]
        }
    
    def _initialize_immune_compartments(self):
        """Initialize immune cell compartments with baseline values."""
        # CD4+ T helper cells
        self.immune_cells[ImmuneCellType.CD4_T_HELPER] = {
            'count': 800,  # cells/μL
            'activation_state': 0.1,
            'proliferation_rate': 0.1,
            'survival_rate': 0.95,
            'hiv_susceptibility': 1.0,  # Relative susceptibility to HIV
            'epitope_recognition': self._define_cd4_epitopes()
        }
        
        # CD8+ T cytotoxic cells
        self.immune_cells[ImmuneCellType.CD8_T_CYTOTOXIC] = {
            'count': 500,  # cells/μL
            'activation_state': 0.05,
            'cytotoxic_activity': 0.1,
            'proliferation_rate': 0.05,
            'hiv_specificity': 0.01,  # Baseline HIV-specific response
            'epitope_recognition': self._define_cd8_epitopes()
        }
        
        # B cells
        self.immune_cells[ImmuneCellType.B_CELL] = {
            'count': 150,  # cells/μL
            'activation_state': 0.02,
            'differentiation_rate': 0.01,
            'isotype_switching': 0.1,
            'memory_b_cells': 0.05
        }
        
        # NK cells
        self.immune_cells[ImmuneCellType.NK_CELL] = {
            'count': 200,  # cells/μL
            'activation_state': 0.05,
            'cytotoxic_activity': 0.2,
            'adcc_activity': 0.15
        }
        
        # Initialize cytokine network
        self.cytokines = {
            'IL2': {'concentration': 0.1, 'source': ['CD4_T_helper'], 'effect': 'proliferation'},
            'IL4': {'concentration': 0.05, 'source': ['CD4_T_helper'], 'effect': 'isotype_switching'},
            'IL6': {'concentration': 0.2, 'source': ['macrophage'], 'effect': 'inflammation'},
            'IL7': {'concentration': 0.3, 'source': ['stromal'], 'effect': 'homeostasis'},
            'IL10': {'concentration': 0.1, 'source': ['regulatory'], 'effect': 'suppression'},
            'TNFa': {'concentration': 0.15, 'source': ['macrophage'], 'effect': 'inflammation'},
            'IFNg': {'concentration': 0.08, 'source': ['CD8_T', 'NK'], 'effect': 'activation'},
            'TGFb': {'concentration': 0.12, 'source': ['regulatory'], 'effect': 'suppression'}
        }
        
        # Initialize antibodies
        self.antibodies = {
            'IgG': {'titer': 0.01, 'hiv_neutralization': 0.05, 'half_life': 21},  # days
            'IgA': {'titer': 0.02, 'mucosal_protection': 0.03, 'half_life': 5},
            'IgM': {'titer': 0.05, 'early_response': 0.1, 'half_life': 5}
        }
    
    def _define_cd4_epitopes(self) -> Dict[str, Dict]:
        """Define CD4+ T cell epitope recognition."""
        return {
            "Gag_p24": {
                "peptide": "Gag_p24_1-10",
                "hla_restriction": self.hla_type,
                "immunogenicity": 0.8,
                "crossreactivity": 0.1
            },
            "Pol_RT": {
                "peptide": "Pol_RT_1-10",
                "hla_restriction": self.hla_type,
                "immunogenicity": 0.6,
                "crossreactivity": 0.15
            }
        }
    
    def _define_cd8_epitopes(self) -> Dict[str, Dict]:
        """Define CD8+ T cell epitope recognition."""
        return {
            "Gag_p24_KW": {
                "peptide": "KW10",  # A*02:01 restricted
                "hla_restriction": "HLA-A*02:01",
                "avidity": 0.9,
                "cytotoxic_potency": 0.85,
                "escape_mutations": ["K20R", "W22G"]
            },
            "Pol_RT_IL": {
                "peptide": "IL9",   # B*57:01 restricted
                "hla_restriction": "HLA-B*57:01",
                "avidity": 0.95,
                "cytotoxic_potency": 0.9,
                "escape_mutations": ["I13V", "L14M"]
            }
        }
    
    def update_immune_status(self, viral_load: float, hiv_genome: str = "") -> Dict:
        """
        Update immune system status based on viral load and genome.
        
        Args:
            viral_load: Current viral load (copies/mL)
            hiv_genome: Current HIV genome sequence
            
        Returns:
            Updated immune status
        """
        # Calculate immune activation based on viral load
        viral_signal = min(viral_load / self.viral_antigen_threshold, self.immune_activation_max)
        
        # Update CD4+ T cell compartment
        cd4_compartment = self.immune_cells[ImmuneCellType.CD4_T_HELPER]
        baseline_cd4 = cd4_compartment['count']
        
        # CD4 depletion due to HIV infection
        if viral_load > 1000:
            depletion_factor = min(0.95, 0.1 + 0.8 * (viral_load / 1e6))
            cd4_compartment['count'] *= depletion_factor
        
        # Update CD8+ T cell compartment
        cd8_compartment = self.immune_cells[ImmuneCellType.CD8_T_CYTOTOXIC]
        
        # CD8 activation in response to viral load
        if viral_load > self.viral_antigen_threshold:
            activation_boost = min(2.0, viral_signal * 0.5)
            cd8_compartment['activation_state'] = min(
                1.0, cd8_compartment['activation_state'] + activation_boost * 0.1
            )
            
            # Increase HIV-specific response
            cd8_compartment['hiv_specificity'] = min(
                0.5, cd8_compartment['hiv_specificity'] + viral_signal * 0.01
            )
        
        # Update B cell compartment
        b_compartment = self.immune_cells[ImmuneCellType.B_CELL]
        if viral_load > self.viral_antigen_threshold:
            b_compartment['activation_state'] = min(
                1.0, b_compartment['activation_state'] + viral_signal * 0.05
            )
            
            # Increase differentiation to plasma cells
            b_compartment['differentiation_rate'] = min(
                0.5, b_compartment['differentiation_rate'] + viral_signal * 0.005
            )
        
        # Update cytokine network
        self._update_cytokine_network(viral_signal)
        
        # Update antibody titers
        self._update_antibody_responses(viral_load)
        
        # Check for immune exhaustion
        self._check_immune_exhaustion()
        
        return {
            'cd4_count': cd4_compartment['count'],
            'cd8_activation': cd8_compartment['activation_state'],
            'b_cell_activation': b_compartment['activation_state'],
            'cytokine_levels': self.cytokines,
            'antibody_titers': self.antibodies,
            'hla_effect': self._get_hla_specific_effects()
        }
    
    def _update_cytokine_network(self, viral_signal: float):
        """Update cytokine concentrations based on viral signal."""
        # Increase inflammatory cytokines
        self.cytokines['TNFa']['concentration'] = min(
            2.0, self.cytokines['TNFa']['concentration'] + viral_signal * 0.1
        )
        self.cytokines['IL6']['concentration'] = min(
            1.5, self.cytokines['IL6']['concentration'] + viral_signal * 0.08
        )
        self.cytokines['IFNg']['concentration'] = min(
            1.2, self.cytokines['IFNg']['concentration'] + viral_signal * 0.05
        )
        
        # Anti-inflammatory response
        if self.cytokines['TNFa']['concentration'] > 0.5:
            self.cytokines['IL10']['concentration'] = min(
                1.0, self.cytokines['IL10']['concentration'] + 0.05
            )
            self.cytokines['TGFb']['concentration'] = min(
                1.0, self.cytokines['TGFb']['concentration'] + 0.03
            )
    
    def _update_antibody_responses(self, viral_load: float):
        """Update antibody titers based on viral exposure."""
        if viral_load > self.viral_antigen_threshold:
            # Increase IgG titer (primary neutralizing antibodies)
            self.antibodies['IgG']['titer'] = min(
                1.0, self.antibodies['IgG']['titer'] + min(0.1, viral_load / 1e6)
            )
            
            # Increase neutralization activity
            self.antibodies['IgG']['hiv_neutralization'] = min(
                0.8, self.antibodies['IgG']['hiv_neutralization'] + min(0.05, viral_load / 5e6)
            )
    
    def _check_immune_exhaustion(self):
        """Check for signs of immune exhaustion."""
        cd8_compartment = self.immune_cells[ImmuneCellType.CD8_T_CYTOTOXIC]
        
        # Check if activation is too high for too long (sign of exhaustion)
        if cd8_compartment['activation_state'] > self.exhaustion_threshold:
            # Increase expression of inhibitory receptors
            cd8_compartment['exhaustion_markers'] = getattr(cd8_compartment, 'exhaustion_markers', 0.0)
            cd8_compartment['exhaustion_markers'] = min(
                1.0, cd8_compartment['exhaustion_markers'] + 0.05
            )
            
            # Reduce cytotoxic activity
            cd8_compartment['cytotoxic_activity'] = max(
                0.05, cd8_compartment['cytotoxic_activity'] * 0.95
            )
    
    def _get_hla_specific_effects(self) -> Dict:
        """Get effects of host HLA type on immune response."""
        effects = {
            'ctl_efficiency': 0.5,  # Baseline
            'disease_progression': 'normal',  # Baseline
            'viral_escape_likelihood': 0.3  # Baseline
        }
        
        if self.hla_type in self.hla_restrictions:
            hla_features = self.hla_restrictions[self.hla_type]
            
            if "protective" in hla_features or "B*57:01" in hla_features:
                effects['ctl_efficiency'] = 0.8
                effects['disease_progression'] = 'slow'
                effects['viral_escape_likelihood'] = 0.1
            elif "risk" in hla_features or "B*35:01" in hla_features:
                effects['ctl_efficiency'] = 0.3
                effects['disease_progression'] = 'fast'
                effects['viral_escape_likelihood'] = 0.6
        
        return effects
    
    def calculate_ctl_response(self, hiv_peptides: List[str]) -> Dict:
        """
        Calculate CTL response to specific HIV peptides.
        
        Args:
            hiv_peptides: List of HIV peptides presented
            
        Returns:
            CTL response characteristics
        """
        cd8_compartment = self.immune_cells[ImmuneCellType.CD8_T_CYTOTOXIC]
        
        response_strength = 0.0
        recognized_epitopes = []
        escape_likelihood = 0.0
        
        for peptide in hiv_peptides:
            # Check if this peptide is recognized by host HLA
            for epitope_name, epitope_info in cd8_compartment['epitope_recognition'].items():
                if epitope_info['hla_restriction'] == self.hla_type:
                    # Calculate recognition strength
                    recognition = epitope_info['avidity'] * cd8_compartment['activation_state']
                    response_strength += recognition
                    recognized_epitopes.append(epitope_name)
                    
                    # Calculate likelihood of escape mutations
                    escape_likelihood += (1 - recognition) * 0.1
        
        return {
            'response_strength': min(1.0, response_strength),
            'recognized_epitopes': recognized_epitopes,
            'escape_likelihood': min(1.0, escape_likelihood),
            'cytotoxic_potential': cd8_compartment['cytotoxic_activity'] * response_strength
        }
    
    def calculate_antibody_response(self, viral_proteins: List[str]) -> Dict:
        """
        Calculate antibody response to specific viral proteins.
        
        Args:
            viral_proteins: List of viral proteins (Env, Gag, Pol, etc.)
            
        Returns:
            Antibody response characteristics
        """
        antibody_response = {
            'neutralization_titer': 0.0,
            'binding_titer': 0.0,
            'breadth': 0.0,
            'maturation_index': 0.0
        }
        
        for protein in viral_proteins:
            if protein == "Env":
                # Env-specific antibodies are most important for neutralization
                antibody_response['neutralization_titer'] += self.antibodies['IgG']['hiv_neutralization']
                antibody_response['binding_titer'] += self.antibodies['IgG']['titer']
            elif protein == "Gag":
                # Gag antibodies are important for ADCC
                antibody_response['binding_titer'] += self.antibodies['IgG']['titer'] * 0.5
        
        # Calculate breadth (number of different proteins recognized)
        antibody_response['breadth'] = min(1.0, len(set(viral_proteins)) / 10.0)
        
        # Calculate maturation index based on chronicity
        antibody_response['maturation_index'] = min(1.0, self.antibodies['IgG']['titer'] * 2.0)
        
        return antibody_response
    
    def simulate_immune_response(self, viral_load_trajectory: List[float],
                               duration_days: int = 365) -> Dict:
        """
        Simulate immune response over time given viral load trajectory.
        
        Args:
            viral_load_trajectory: List of viral loads over time
            duration_days: Duration of simulation in days
            
        Returns:
            Complete immune response simulation results
        """
        immune_trajectories = {
            'cd4_count': [],
            'cd8_activation': [],
            'b_cell_activation': [],
            'tnf_alpha': [],
            'ifn_gamma': [],
            'igg_titer': [],
            'ctl_response': []
        }
        
        # Pad viral load trajectory if needed
        if len(viral_load_trajectory) < duration_days:
            # Extend with last value
            extended_vl = viral_load_trajectory + [viral_load_trajectory[-1]] * (duration_days - len(viral_load_trajectory))
        else:
            extended_vl = viral_load_trajectory[:duration_days]
        
        for day, vl in enumerate(extended_vl):
            # Update immune status
            immune_status = self.update_immune_status(vl)
            
            # Record trajectories
            immune_trajectories['cd4_count'].append(immune_status['cd4_count'])
            immune_trajectories['cd8_activation'].append(immune_status['cd8_activation'])
            immune_trajectories['b_cell_activation'].append(immune_status['b_cell_activation'])
            immune_trajectories['tnf_alpha'].append(immune_status['cytokine_levels']['TNFa']['concentration'])
            immune_trajectories['ifn_gamma'].append(immune_status['cytokine_levels']['IFNg']['concentration'])
            immune_trajectories['igg_titer'].append(immune_status['antibody_titers']['IgG']['titer'])
            
            # Calculate CTL response to representative peptides
            # In a real simulation, these would come from the viral sequence
            ctl_response = self.calculate_ctl_response(["Gag_p24_KW", "Pol_RT_IL"])
            immune_trajectories['ctl_response'].append(ctl_response['response_strength'])
        
        # Calculate summary statistics
        summary_stats = {
            'mean_cd4': np.mean(immune_trajectories['cd4_count']),
            'peak_cd4_activation': max(immune_trajectories['cd8_activation']),
            'mean_antibody_titer': np.mean(immune_trajectories['igg_titer']),
            'auc_viral_load': np.trapz(viral_load_trajectory[:duration_days]),
            'immune_correlations': self._calculate_immune_correlations(immune_trajectories)
        }
        
        return {
            'trajectories': immune_trajectories,
            'summary_statistics': summary_stats,
            'hla_effects': self._get_hla_specific_effects()
        }
    
    def _calculate_immune_correlations(self, trajectories: Dict) -> Dict:
        """Calculate correlations between immune parameters."""
        # Convert trajectories to numpy arrays for correlation calculation
        cd4 = np.array(trajectories['cd4_count'])
        cd8_act = np.array(trajectories['cd8_activation'])
        igg = np.array(trajectories['igg_titer'])
        tnf = np.array(trajectories['tnf_alpha'])
        
        # Calculate correlations
        correlations = {
            'cd4_vs_cd8': np.corrcoef(cd4, cd8_act)[0, 1] if len(cd4) > 1 else 0,
            'antibody_vs_inflammation': np.corrcoef(igg, tnf)[0, 1] if len(igg) > 1 else 0,
            'cd8_vs_inflammation': np.corrcoef(cd8_act, tnf)[0, 1] if len(cd8_act) > 1 else 0
        }
        
        return correlations


def run_immune_system_demo():
    """Demo function showing how to use the ImmuneSystemEngine."""
    print("Starting HIV Immune System Modeling Demo...")
    
    # Initialize the engine with a protective HLA type
    immune_engine = ImmuneSystemEngine(hla_type="HLA-B*57:01")
    
    # Simulate a viral load trajectory (copies/mL over time)
    days = 180
    time_points = list(range(days))
    
    # Create a realistic viral load trajectory with acute phase, peak, and set point
    viral_loads = []
    for t in time_points:
        if t < 14:  # Acute phase
            vl = 10 ** (2 + 3 * (t / 14))  # Exponential increase
        elif t < 60:  # Peak and decline
            vl = 10 ** (5 - 0.05 * (t - 14))  # Slow decline to set point
        else:  # Chronic phase
            vl = 10 ** (4 + 0.5 * np.sin(t / 30))  # Oscillating around set point
    
    print(f"\nSimulating immune response to {len(viral_loads)} time points of viral load...")
    
    # Run immune simulation
    results = immune_engine.simulate_immune_response(viral_loads, duration_days=days)
    
    # Print results
    print(f"\nImmune Response Summary:")
    print(f"  Mean CD4+ T cell count: {results['summary_statistics']['mean_cd4']:.1f} cells/μL")
    print(f"  Peak CD8+ T cell activation: {results['summary_statistics']['peak_cd4_activation']:.3f}")
    print(f"  Mean IgG titer: {results['summary_statistics']['mean_antibody_titer']:.3f}")
    print(f"  AUC of viral load: {results['summary_statistics']['auc_viral_load']:.2e}")
    
    print(f"\nHLA Type Effects:")
    hla_effects = results['hla_effects']
    for key, value in hla_effects.items():
        print(f"  {key}: {value}")
    
    print(f"\nImmune Correlations:")
    correlations = results['summary_statistics']['immune_correlations']
    for key, value in correlations.items():
        print(f"  {key}: {value:.3f}")
    
    # Test CTL response to specific peptides
    print(f"\nCTL Response to Protective Epitopes:")
    ctl_response = immune_engine.calculate_ctl_response(["Gag_p24_KW", "Pol_RT_IL"])
    print(f"  Response Strength: {ctl_response['response_strength']:.3f}")
    print(f"  Recognized Epitopes: {ctl_response['recognized_epitopes']}")
    print(f"  Escape Likelihood: {ctl_response['escape_likelihood']:.3f}")
    print(f"  Cytotoxic Potential: {ctl_response['cytotoxic_potential']:.3f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_immune_system_demo()