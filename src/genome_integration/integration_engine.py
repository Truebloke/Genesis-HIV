"""
Integration engine module for simulating HIV genomic integration into host DNA.

This module handles the simulation of HIV integration process, including
identification of integration sites, modeling of integration efficiency,
and simulation of latent reservoir formation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class IntegrationOutcome(Enum):
    """Possible outcomes of HIV integration."""
    SUCCESSFUL_INTEGRATION = "successful_integration"
    FAILED_INTEGRATION = "failed_integration"
    APOPTOSIS = "apoptosis"
    LATENT_INTEGRATION = "latent_integration"


@dataclass
class HostCell:
    """Represents a host cell in the simulation."""
    cell_id: str
    cell_type: str  # e.g., "CD4_T_CELL", "MACROPHAGE"
    activation_state: float  # 0.0 to 1.0
    chromatin_state: Dict[str, float]  # accessibility of different genomic regions
    proviral_load: int = 0
    integrated_virus: bool = False
    latent_state: bool = False
    integration_site: Optional[Tuple[str, int]] = None  # chromosome, position


@dataclass
class IntegrationEvent:
    """Represents a single integration event."""
    event_id: str
    host_cell: HostCell
    viral_genome: str
    integration_site: Tuple[str, int]  # chromosome, position
    outcome: IntegrationOutcome
    integration_efficiency: float
    latency_probability: float
    timestamp: float


class IntegrationEngine:
    """
    A class to simulate HIV integration into host genome.
    
    This includes:
    - Modeling integration site selection
    - Simulating integration efficiency
    - Modeling latent reservoir formation
    - Tracking viral dynamics
    """
    
    def __init__(self):
        """Initialize the IntegrationEngine."""
        self.integration_sites = {}  # Maps chromosome regions to integration probability
        self.host_genome = {}  # Simplified representation of host genome
        self.integration_events = []
        self.cells = []
        
        # Initialize with known HIV integration preferences
        self._initialize_integration_preferences()
    
    def _initialize_integration_preferences(self) -> None:
        """Initialize known preferences for HIV integration sites."""
        # HIV prefers active transcription units, especially in gene-rich regions
        self.integration_preferences = {
            'promoter_regions': 0.8,
            'transcription_units': 0.6,
            'gene_dense_regions': 0.5,
            'heterochromatin': 0.1,
            'telomeres': 0.2,
            'centromeres': 0.05
        }
        
        # Chromosome-specific integration rates based on gene density
        self.chromosome_bias = {
            'chr1': 0.08, 'chr2': 0.07, 'chr3': 0.06, 'chr4': 0.05, 'chr5': 0.06,
            'chr6': 0.06, 'chr7': 0.05, 'chr8': 0.04, 'chr9': 0.04, 'chr10': 0.04,
            'chr11': 0.04, 'chr12': 0.04, 'chr13': 0.02, 'chr14': 0.02, 'chr15': 0.02,
            'chr16': 0.02, 'chr17': 0.03, 'chr18': 0.02, 'chr19': 0.03, 'chr20': 0.02,
            'chr21': 0.01, 'chr22': 0.02, 'chrX': 0.05, 'chrY': 0.001
        }
    
    def create_host_cell(self, cell_type: str, activation_state: float) -> HostCell:
        """
        Create a host cell for the simulation.
        
        Args:
            cell_type (str): Type of host cell (e.g., CD4_T_CELL)
            activation_state (float): Activation state (0.0 to 1.0)
            
        Returns:
            HostCell: Created host cell object
        """
        cell_id = f"cell_{len(self.cells)}"
        
        # Initialize chromatin state based on cell type and activation
        chromatin_state = self._generate_chromatin_state(cell_type, activation_state)
        
        cell = HostCell(
            cell_id=cell_id,
            cell_type=cell_type,
            activation_state=activation_state,
            chromatin_state=chromatin_state
        )
        
        self.cells.append(cell)
        return cell
    
    def _generate_chromatin_state(self, cell_type: str, activation_state: float) -> Dict[str, float]:
        """
        Generate chromatin accessibility state for a cell.
        
        Args:
            cell_type (str): Type of cell
            activation_state (float): Activation state (0.0 to 1.0)
            
        Returns:
            Dict[str, float]: Chromatin accessibility for different regions
        """
        chromatin_state = {}
        
        # Different cell types have different chromatin accessibility patterns
        for region_type in self.integration_preferences.keys():
            # Base accessibility modified by cell type and activation
            base_accessibility = self.integration_preferences[region_type]
            
            # Activation increases accessibility of transcriptionally active regions
            if region_type in ['promoter_regions', 'transcription_units', 'gene_dense_regions']:
                accessibility = base_accessibility * (0.5 + 0.5 * activation_state)
            else:
                accessibility = base_accessibility * (1.0 - 0.3 * activation_state)
            
            # Clamp between 0 and 1
            chromatin_state[region_type] = max(0.0, min(1.0, accessibility))
        
        return chromatin_state
    
    def simulate_integration(self, host_cell: HostCell, viral_genome: str, 
                           integration_time: float = 0.0) -> IntegrationEvent:
        """
        Simulate a single integration event.
        
        Args:
            host_cell (HostCell): The host cell to integrate into
            viral_genome (str): The viral genome sequence to integrate
            integration_time (float): Time of integration
            
        Returns:
            IntegrationEvent: Result of the integration event
        """
        event_id = f"integration_{len(self.integration_events)}"
        
        # Determine if integration is successful based on cellular factors
        success_probability = self._calculate_integration_success(host_cell)
        
        if random.random() < success_probability:
            # Select integration site based on preferences and chromatin state
            integration_site = self._select_integration_site(host_cell)
            
            # Determine if integration leads to latency
            latency_probability = self._calculate_latency_probability(host_cell, integration_site)
            
            if random.random() < latency_probability:
                outcome = IntegrationOutcome.LATENT_INTEGRATION
                host_cell.latent_state = True
            else:
                outcome = IntegrationOutcome.SUCCESSFUL_INTEGRATION
                host_cell.integrated_virus = True
            
            # Record integration
            host_cell.integration_site = integration_site
            host_cell.proviral_load += 1
            
            integration_event = IntegrationEvent(
                event_id=event_id,
                host_cell=host_cell,
                viral_genome=viral_genome,
                integration_site=integration_site,
                outcome=outcome,
                integration_efficiency=success_probability,
                latency_probability=latency_probability,
                timestamp=integration_time
            )
        else:
            # Integration failed
            integration_event = IntegrationEvent(
                event_id=event_id,
                host_cell=host_cell,
                viral_genome=viral_genome,
                integration_site=None,
                outcome=IntegrationOutcome.FAILED_INTEGRATION,
                integration_efficiency=success_probability,
                latency_probability=0.0,
                timestamp=integration_time
            )
        
        self.integration_events.append(integration_event)
        return integration_event
    
    def _calculate_integration_success(self, host_cell: HostCell) -> float:
        """
        Calculate the probability of successful integration.
        
        Args:
            host_cell (HostCell): The host cell
            
        Returns:
            float: Probability of successful integration (0.0 to 1.0)
        """
        # Factors affecting integration success:
        # 1. Cell activation state (higher activation = higher success)
        activation_factor = host_cell.activation_state
        
        # 2. Chromatin accessibility (more accessible = higher success)
        avg_accessibility = np.mean(list(host_cell.chromatin_state.values()))
        
        # 3. Cell type specific factors
        cell_type_factors = {
            'CD4_T_CELL': 1.0,
            'MACROPHAGE': 0.7,
            'MONOCYTE': 0.5,
            'MICROGLIA': 0.6
        }
        cell_type_factor = cell_type_factors.get(host_cell.cell_type, 0.5)
        
        # Combine factors with weights
        success_prob = (
            0.3 * activation_factor +
            0.4 * avg_accessibility +
            0.3 * cell_type_factor
        )
        
        # Apply upper bound
        return min(success_prob, 0.95)
    
    def _select_integration_site(self, host_cell: HostCell) -> Tuple[str, int]:
        """
        Select an integration site based on preferences and chromatin state.
        
        Args:
            host_cell (HostCell): The host cell
            
        Returns:
            Tuple[str, int]: Selected chromosome and position
        """
        # Weight different chromosome regions by accessibility
        region_weights = []
        region_names = []
        
        for region_type, accessibility in host_cell.chromatin_state.items():
            # Weight by both intrinsic preference and chromatin accessibility
            weight = self.integration_preferences[region_type] * accessibility
            region_weights.append(weight)
            region_names.append(region_type)
        
        # Normalize weights
        total_weight = sum(region_weights)
        if total_weight == 0:
            # Fallback to uniform distribution if no accessible regions
            region_weights = [1.0 / len(region_names)] * len(region_names)
        else:
            region_weights = [w / total_weight for w in region_weights]
        
        # Select region based on weights
        selected_region = np.random.choice(region_names, p=region_weights)
        
        # Select chromosome based on bias
        chromosomes = list(self.chromosome_bias.keys())
        chr_weights = list(self.chromosome_bias.values())
        total_chr_weight = sum(chr_weights)
        chr_probs = [w / total_chr_weight for w in chr_weights]
        
        selected_chromosome = np.random.choice(chromosomes, p=chr_probs)
        
        # Select position within chromosome (simplified)
        # In reality, this would depend on the specific region type
        position = random.randint(1, 100000000)  # Random position up to 100Mbp
        
        return (selected_chromosome, position)
    
    def _calculate_latency_probability(self, host_cell: HostCell, 
                                     integration_site: Tuple[str, int]) -> float:
        """
        Calculate the probability of entering latency.
        
        Args:
            host_cell (HostCell): The host cell
            integration_site: The integration site
            
        Returns:
            float: Probability of latency (0.0 to 1.0)
        """
        # Factors affecting latency:
        # 1. Integration site (some regions favor latency)
        chr, pos = integration_site
        if 'chr19' in chr or 'chr17' in chr:  # Gene-rich chromosomes
            site_factor = 0.3
        elif 'heterochromatin' in self.host_genome.get(chr, {}).get(pos, {}):
            site_factor = 0.8
        else:
            site_factor = 0.5
        
        # 2. Cell activation state (lower activation = higher latency)
        activation_factor = 1.0 - host_cell.activation_state
        
        # 3. Cell type
        cell_type_factors = {
            'CD4_T_CELL': 0.4,  # Memory T cells can go latent
            'MACROPHAGE': 0.7,  # Macrophages often establish latency
            'MONOCYTE': 0.6,
            'MICROGLIA': 0.8
        }
        cell_type_factor = cell_type_factors.get(host_cell.cell_type, 0.5)
        
        # Combine factors
        latency_prob = (
            0.4 * site_factor +
            0.4 * activation_factor +
            0.2 * cell_type_factor
        )
        
        return min(latency_prob, 0.95)
    
    def simulate_multiple_integrations(self, host_cell: HostCell, viral_genome: str, 
                                    num_integrations: int, time_step: float = 1.0) -> List[IntegrationEvent]:
        """
        Simulate multiple integration events over time.
        
        Args:
            host_cell (HostCell): The host cell
            viral_genome (str): The viral genome sequence
            num_integrations (int): Number of integration attempts
            time_step (float): Time interval between integrations
            
        Returns:
            List[IntegrationEvent]: List of integration events
        """
        events = []
        current_time = 0.0
        
        for i in range(num_integrations):
            event = self.simulate_integration(host_cell, viral_genome, current_time)
            events.append(event)
            current_time += time_step
        
        return events
    
    def activate_latent_virus(self, host_cell: HostCell, stimulus_level: float = 1.0) -> bool:
        """
        Simulate activation of latent virus.
        
        Args:
            host_cell (HostCell): The host cell with latent virus
            stimulus_level (float): Level of activation stimulus (0.0 to 1.0)
            
        Returns:
            bool: True if virus was activated, False otherwise
        """
        if not host_cell.latent_state:
            return False
        
        # Probability of activation depends on stimulus and integration site
        activation_probability = stimulus_level * host_cell.activation_state
        
        if random.random() < activation_probability:
            host_cell.latent_state = False
            host_cell.activation_state = min(1.0, host_cell.activation_state + 0.3)
            return True
        
        return False
    
    def get_integration_summary(self) -> Dict[str, any]:
        """
        Get a summary of integration events.
        
        Returns:
            Dict: Summary statistics of integration events
        """
        if not self.integration_events:
            return {"error": "No integration events recorded"}
        
        outcomes = [event.outcome.value for event in self.integration_events]
        successful_count = outcomes.count(IntegrationOutcome.SUCCESSFUL_INTEGRATION.value)
        latent_count = outcomes.count(IntegrationOutcome.LATENT_INTEGRATION.value)
        failed_count = outcomes.count(IntegrationOutcome.FAILED_INTEGRATION.value)
        
        # Calculate average integration efficiency
        efficiencies = [event.integration_efficiency for event in self.integration_events]
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # Count cell types involved
        cell_types = [event.host_cell.cell_type for event in self.integration_events]
        cell_type_counts = {}
        for ct in cell_types:
            cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
        
        summary = {
            "total_events": len(self.integration_events),
            "successful_integrations": successful_count,
            "latent_integrations": latent_count,
            "failed_integrations": failed_count,
            "average_efficiency": avg_efficiency,
            "cell_type_distribution": cell_type_counts,
            "latent_reservoir_size": sum(1 for cell in self.cells if cell.latent_state)
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    engine = IntegrationEngine()
    
    # Create some host cells
    t_cell = engine.create_host_cell("CD4_T_CELL", activation_state=0.3)  # Resting T cell
    macrophage = engine.create_host_cell("MACROPHAGE", activation_state=0.7)  # Activated macrophage
    
    print(f"Created cells: {t_cell.cell_id}, {macrophage.cell_id}")
    print(f"T cell chromatin state: {t_cell.chromatin_state}")
    print(f"Macrophage chromatin state: {macrophage.chromatin_state}")
    
    # Simulate integration events
    viral_genome = "ATGCGATCGTAGCTAGCTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
    
    # Simulate integration in T cell
    t_event = engine.simulate_integration(t_cell, viral_genome, integration_time=0.0)
    print(f"\nT cell integration: {t_event.outcome.value}")
    print(f"Integration efficiency: {t_event.integration_efficiency:.2f}")
    print(f"Latency probability: {t_event.latency_probability:.2f}")
    
    # Simulate integration in macrophage
    m_event = engine.simulate_integration(macrophage, viral_genome, integration_time=1.0)
    print(f"\nMacrophage integration: {m_event.outcome.value}")
    print(f"Integration efficiency: {m_event.integration_efficiency:.2f}")
    print(f"Latency probability: {m_event.latency_probability:.2f}")
    
    # Simulate multiple integrations
    multi_events = engine.simulate_multiple_integrations(t_cell, viral_genome, 5, time_step=1.0)
    print(f"\nSimulated {len(multi_events)} additional integration events in T cell")
    
    # Activate latent virus
    if t_cell.latent_state:
        activated = engine.activate_latent_virus(t_cell, stimulus_level=0.8)
        print(f"\nLatent virus activation attempt: {'Success' if activated else 'Failed'}")
    
    # Print integration summary
    summary = engine.get_integration_summary()
    print(f"\nIntegration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")