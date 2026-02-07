"""
Therapeutic intervention modeling module for Project Genesis-HIV.

This module models various therapeutic interventions for HIV, including:
- Antiretroviral therapy (ART)
- Latency reversing agents (LRAs)
- Immunotherapies
- Gene therapies
- Combination approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class InterventionType(Enum):
    """Types of therapeutic interventions."""
    ART = "antiretroviral_therapy"
    LRA = "latency_reversing_agent"
    IMMUNOTHERAPY = "immunotherapy"
    GENE_THERAPY = "gene_therapy"
    COMBINATION = "combination_therapy"
    VACCINE = "vaccine"


@dataclass
class Intervention:
    """Represents a therapeutic intervention."""
    name: str
    intervention_type: InterventionType
    mechanism_of_action: str
    efficacy_parameters: Dict[str, float]  # Efficacy metrics
    administration_schedule: Dict[str, float]  # Dosing, timing
    resistance_profile: Dict[str, float]  # Resistance mutations and impact
    side_effects: Dict[str, float]  # Side effect profile
    cost: float  # Cost per patient per year


@dataclass
class TreatmentOutcome:
    """Represents an outcome from a treatment intervention."""
    viral_suppression: bool  # Achieved viral suppression
    cd4_recovery: float  # CD4 count improvement
    reservoir_reduction: float  # Log reduction in latent reservoir
    time_to_failure: int  # Days to treatment failure
    adverse_events: List[str]  # List of adverse events
    quality_of_life: float  # Quality of life score (0-1)
    cost_effectiveness: float  # Cost per quality-adjusted life year


class TherapeuticInterventionEngine:
    """
    A class to model therapeutic interventions for HIV.
    
    This includes:
    - Antiretroviral therapy (ART) modeling
    - Latency reversing agents (LRAs)
    - Immunotherapies (CAR-T, vaccines, etc.)
    - Gene therapies (gene editing, etc.)
    - Combination approaches
    - Treatment sequencing and optimization
    - Resistance development modeling
    - Cost-effectiveness analysis
    """
    
    def __init__(self):
        """Initialize the TherapeuticInterventionEngine."""
        self.interventions = self._initialize_interventions()
        self.treatment_sequencing_rules = self._initialize_sequencing_rules()
        self.cost_parameters = self._initialize_cost_parameters()
        
        # Biological parameters
        self.viral_turnover_rate = 2.0  # Virions produced per day per infected cell
        self.infected_cell_death_rate = 0.5  # Per day
        self.latent_cell_reactivation_rate = 1e-4  # Per day
        self.treatment_adherence = 0.95  # 95% adherence
    
    def _initialize_interventions(self) -> Dict[str, Intervention]:
        """Initialize known therapeutic interventions."""
        return {
            # Standard ART regimens
            "TDF-FTC-EFV": Intervention(
                name="Tenofovir-Emtricitabine-Efavirenz",
                intervention_type=InterventionType.ART,
                mechanism_of_action="NRTI+NRTI+NNRTI combination",
                efficacy_parameters={
                    'viral_suppression_rate': 0.85,
                    'cd4_recovery_rate': 100,  # cells/μL/year
                    'time_to_suppression': 14,  # days
                    'resistance_development': 0.15  # per year
                },
                administration_schedule={
                    'frequency': 'daily',
                    'duration': 'lifelong',
                    'timing': 'once_daily'
                },
                resistance_profile={
                    'K103N': 10.0,  # Fold change in IC50
                    'M184V': 5.0,
                    'K65R': 3.0
                },
                side_effects={
                    'neuropsychiatric': 0.20,
                    'renal_toxicity': 0.05,
                    'bone_loss': 0.10
                },
                cost=1200  # USD per year
            ),
            
            "TAF-FTC-DTG": Intervention(
                name="Tenofovir-Alafenamide-Emtricitabine-Dolutegravir",
                intervention_type=InterventionType.ART,
                mechanism_of_action="NRTI+NRTI+INSTI combination",
                efficacy_parameters={
                    'viral_suppression_rate': 0.92,
                    'cd4_recovery_rate': 120,
                    'time_to_suppression': 10,
                    'resistance_development': 0.02
                },
                administration_schedule={
                    'frequency': 'daily',
                    'duration': 'lifelong',
                    'timing': 'once_daily'
                },
                resistance_profile={
                    'R263K': 5.0,
                    'N155H': 15.0
                },
                side_effects={
                    'neuropsychiatric': 0.05,
                    'renal_toxicity': 0.01,
                    'bone_loss': 0.02
                },
                cost=2400
            ),
            
            # Latency reversing agents
            "Vorinostat": Intervention(
                name="Vorinostat (HDAC inhibitor)",
                intervention_type=InterventionType.LRA,
                mechanism_of_action="Histone deacetylase inhibition to reactivate latent HIV",
                efficacy_parameters={
                    'reservoir_reactivation': 0.15,  # 15% of cells reactivated
                    'duration_of_effect': 7,  # days
                    'synergy_with_immune_clearance': 0.30  # 30% additional clearance
                },
                administration_schedule={
                    'frequency': 'intermittent',
                    'duration': 'short_course',
                    'timing': '400mg daily for 14 days every 3 months'
                },
                resistance_profile={},  # No direct resistance to LRA
                side_effects={
                    'fatigue': 0.60,
                    'diarrhea': 0.40,
                    'nausea': 0.50,
                    'thrombocytopenia': 0.20
                },
                cost=8000  # Per course
            ),
            
            "Romidepsin": Intervention(
                name="Romidepsin (HDAC inhibitor)",
                intervention_type=InterventionType.LRA,
                mechanism_of_action="Selective HDAC inhibition to reactivate latent HIV",
                efficacy_parameters={
                    'reservoir_reactivation': 0.25,
                    'duration_of_effect': 5,
                    'synergy_with_immune_clearance': 0.40
                },
                administration_schedule={
                    'frequency': 'intermittent',
                    'duration': 'short_course',
                    'timing': '26mg/m2 IV on days 1, 8, 15 every 4 weeks'
                },
                resistance_profile={},
                side_effects={
                    'fatigue': 0.70,
                    'nausea': 0.60,
                    'hypotension': 0.30,
                    'cardiac_toxicity': 0.10
                },
                cost=12000  # Per course
            ),
            
            # Immunotherapies
            "Anti-PD1": Intervention(
                name="Anti-PD1 antibody (Nivolumab)",
                intervention_type=InterventionType.IMMUNOTHERAPY,
                mechanism_of_action="Checkpoint inhibition to restore T cell function",
                efficacy_parameters={
                    't_cell_function_restoration': 0.40,
                    'reservoir_targeting': 0.10,
                    'viral_suppression_support': 0.20
                },
                administration_schedule={
                    'frequency': 'every_2_weeks',
                    'duration': '6_months',
                    'timing': '240mg IV every 2 weeks'
                },
                resistance_profile={},
                side_effects={
                    'autoimmune_effects': 0.30,
                    'fatigue': 0.40,
                    'rash': 0.25,
                    'colitis': 0.15
                },
                cost=150000  # Per year
            ),
            
            # Gene therapies
            "CCR5_gene_editing": Intervention(
                name="CCR5 gene editing (SB-728-T)",
                intervention_type=InterventionType.GENE_THERAPY,
                mechanism_of_action="CCR5 gene disruption to create HIV-resistant cells",
                efficacy_parameters={
                    'hiv_resistance_rate': 0.60,
                    'engraftment_efficiency': 0.40,
                    'durability': 0.90  # 90% durability at 5 years
                },
                administration_schedule={
                    'frequency': 'single_infusion',
                    'duration': 'permanent',
                    'timing': 'ex-vivo modification and reinfusion'
                },
                resistance_profile={
                    'CXCR4_tropic_virus': 0.10  # Potential for X4 emergence
                },
                side_effects={
                    'infusion_reaction': 0.10,
                    'autoimmune_risk': 0.05,
                    'insertional_mutagenesis': 0.01
                },
                cost=300000  # One-time cost
            ),
            
            # Vaccines
            "Therapeutic_vaccine": Intervention(
                name="Therapeutic HIV vaccine",
                intervention_type=InterventionType.VACCINE,
                mechanism_of_action="Boost HIV-specific immune responses",
                efficacy_parameters={
                    'ctl_response_enhancement': 0.50,
                    'antibody_response': 0.30,
                    'functional_cure_probability': 0.10
                },
                administration_schedule={
                    'frequency': 'prime_boost',
                    'duration': '6_months',
                    'timing': 'Prime at 0, boost at 1, 3, 6 months'
                },
                resistance_profile={},
                side_effects={
                    'injection_site_reaction': 0.70,
                    'flu_like_syndrome': 0.40,
                    'autoimmune_risk': 0.05
                },
                cost=5000  # Per series
            )
        }
    
    def _initialize_sequencing_rules(self) -> Dict[str, List[str]]:
        """Initialize treatment sequencing rules."""
        return {
            "first_line": ["TDF-FTC-EFV", "TAF-FTC-DTG"],
            "second_line": ["TAF-FTC-DTG", "Alternative_PI_based"],
            "salvage": ["Multi_agent_salvage", "Investigational_agents"]
        }
    
    def _initialize_cost_parameters(self) -> Dict[str, float]:
        """Initialize cost parameters."""
        return {
            'quality_of_life_weight': 0.8,  # Weight for QALY calculation
            'cost_per_ae': 5000,  # Cost per adverse event
            'monitoring_cost': 1200,  # Annual monitoring cost
            'discount_rate': 0.03  # For cost-effectiveness analysis
        }
    
    def simulate_single_intervention(self, intervention_name: str, 
                                   patient_characteristics: Dict,
                                   duration_days: int = 365) -> TreatmentOutcome:
        """
        Simulate a single therapeutic intervention.
        
        Args:
            intervention_name: Name of the intervention
            patient_characteristics: Patient characteristics (age, baseline CD4, VL, etc.)
            duration_days: Duration of simulation in days
            
        Returns:
            Treatment outcome
        """
        if intervention_name not in self.interventions:
            raise ValueError(f"Unknown intervention: {intervention_name}")
        
        intervention = self.interventions[intervention_name]
        
        # Calculate viral suppression based on efficacy and patient factors
        base_suppression = intervention.efficacy_parameters['viral_suppression_rate']
        
        # Adjust for patient factors
        age_factor = 1.0 - (patient_characteristics.get('age', 35) - 35) * 0.005  # Older patients may have lower response
        baseline_vl_factor = 1.0 - min(0.2, (patient_characteristics.get('baseline_vl_log10', 4.5) - 4.0) * 0.1)  # Higher VL may reduce response
        adherence_factor = self.treatment_adherence  # Adherence affects efficacy
        
        adjusted_suppression = base_suppression * age_factor * baseline_vl_factor * adherence_factor
        viral_suppression = random.random() < adjusted_suppression
        
        # Calculate CD4 recovery
        base_recovery = intervention.efficacy_parameters['cd4_recovery_rate']
        cd4_recovery = base_recovery * (duration_days / 365.0)  # Scale by duration
        
        # Calculate reservoir reduction (for LRAs and combination approaches)
        reservoir_reduction = 0.0
        if intervention.intervention_type == InterventionType.LRA:
            base_reactivation = intervention.efficacy_parameters['reservoir_reactivation']
            synergy = intervention.efficacy_parameters['synergy_with_immune_clearance']
            reservoir_reduction = np.log10(1 + base_reactivation * synergy)
        
        # Calculate time to failure
        resistance_rate = intervention.efficacy_parameters.get('resistance_development', 0.0)
        if resistance_rate > 0:
            # Convert annual rate to daily rate
            daily_failure_rate = 1 - (1 - resistance_rate) ** (1/365.0)
            time_to_failure = np.random.geometric(daily_failure_rate)
        else:
            time_to_failure = duration_days  # No failure during follow-up
        
        # Determine adverse events
        adverse_events = []
        for side_effect, rate in intervention.side_effects.items():
            if random.random() < rate:
                adverse_events.append(side_effect)
        
        # Calculate quality of life (0-1 scale)
        baseline_qol = patient_characteristics.get('baseline_qol', 0.7)
        qol_penalty = len(adverse_events) * 0.05  # Each AE reduces QOL by 5%
        quality_of_life = max(0.0, baseline_qol - qol_penalty)
        
        # Calculate cost-effectiveness
        intervention_cost = intervention.cost
        if intervention.administration_schedule['frequency'] == 'daily':
            intervention_cost = intervention.cost * (duration_days / 365.0)
        elif intervention.administration_schedule['frequency'] == 'intermittent':
            # Calculate number of courses
            course_interval = intervention.administration_schedule.get('course_interval', 90)
            num_courses = duration_days // course_interval
            intervention_cost = intervention.cost * num_courses
        
        # Add monitoring costs
        monitoring_cost = self.cost_parameters['monitoring_cost'] * (duration_days / 365.0)
        total_cost = intervention_cost + monitoring_cost
        
        # Calculate QALYs gained
        qaly_gained = quality_of_life * (duration_days / 365.0) * self.cost_parameters['quality_of_life_weight']
        
        # Cost per QALY
        cost_per_qaly = total_cost / qaly_gained if qaly_gained > 0 else float('inf')
        
        return TreatmentOutcome(
            viral_suppression=viral_suppression,
            cd4_recovery=cd4_recovery,
            reservoir_reduction=reservoir_reduction,
            time_to_failure=time_to_failure,
            adverse_events=adverse_events,
            quality_of_life=quality_of_life,
            cost_effectiveness=cost_per_qaly
        )
    
    def simulate_combination_intervention(self, intervention_names: List[str],
                                       patient_characteristics: Dict,
                                       duration_days: int = 365) -> TreatmentOutcome:
        """
        Simulate a combination of therapeutic interventions.
        
        Args:
            intervention_names: List of intervention names
            patient_characteristics: Patient characteristics
            duration_days: Duration of simulation
            
        Returns:
            Combined treatment outcome
        """
        outcomes = []
        
        for intervention_name in intervention_names:
            outcome = self.simulate_single_intervention(
                intervention_name, patient_characteristics, duration_days
            )
            outcomes.append(outcome)
        
        # Combine outcomes
        viral_suppression = any(out.viral_suppression for out in outcomes)
        cd4_recovery = sum(out.cd4_recovery for out in outcomes)  # Additive effect
        reservoir_reduction = sum(out.reservoir_reduction for out in outcomes)  # Additive effect
        
        # Time to failure is minimum of individual times
        time_to_failure = min(out.time_to_failure for out in outcomes)
        
        # Combine adverse events
        all_adverse_events = []
        for out in outcomes:
            all_adverse_events.extend(out.adverse_events)
        
        # Quality of life is affected by all adverse events
        baseline_qol = patient_characteristics.get('baseline_qol', 0.7)
        qol_penalty = len(all_adverse_events) * 0.03  # Less penalty per AE in combination
        quality_of_life = max(0.0, baseline_qol - qol_penalty)
        
        # Calculate combined cost
        total_cost = 0
        for i, intervention_name in enumerate(intervention_names):
            intervention = self.interventions[intervention_name]
            intervention_cost = intervention.cost
            if intervention.administration_schedule['frequency'] == 'daily':
                intervention_cost = intervention.cost * (duration_days / 365.0)
            elif intervention.administration_schedule['frequency'] == 'intermittent':
                course_interval = intervention.administration_schedule.get('course_interval', 90)
                num_courses = duration_days // course_interval
                intervention_cost = intervention.cost * num_courses
            
            total_cost += intervention_cost
        
        # Add monitoring costs
        monitoring_cost = self.cost_parameters['monitoring_cost'] * (duration_days / 365.0)
        total_cost += monitoring_cost
        
        # Calculate QALYs and cost-effectiveness
        qaly_gained = quality_of_life * (duration_days / 365.0) * self.cost_parameters['quality_of_life_weight']
        cost_per_qaly = total_cost / qaly_gained if qaly_gained > 0 else float('inf')
        
        return TreatmentOutcome(
            viral_suppression=viral_suppression,
            cd4_recovery=cd4_recovery,
            reservoir_reduction=reservoir_reduction,
            time_to_failure=time_to_failure,
            adverse_events=list(set(all_adverse_events)),  # Remove duplicates
            quality_of_life=quality_of_life,
            cost_effectiveness=cost_per_qaly
        )
    
    def optimize_treatment_sequence(self, patient_characteristics: Dict,
                                  first_line_options: List[str] = None,
                                  duration_days: int = 1095) -> Dict:  # 3 years
        """
        Optimize treatment sequence for a patient.
        
        Args:
            patient_characteristics: Patient characteristics
            first_line_options: Options for first-line therapy
            duration_days: Duration of optimization
            
        Returns:
            Optimized treatment sequence and outcomes
        """
        if first_line_options is None:
            first_line_options = self.treatment_sequencing_rules['first_line']
        
        # For simplicity, we'll evaluate each first-line option and select the best
        best_sequence = None
        best_outcome = None
        best_score = float('-inf')
        
        for first_line in first_line_options:
            # Simulate first-line therapy
            first_outcome = self.simulate_single_intervention(
                first_line, patient_characteristics, duration_days
            )
            
            # Calculate composite score (balance efficacy, safety, cost)
            efficacy_score = (
                (first_outcome.viral_suppression * 0.4) +
                (min(1.0, first_outcome.cd4_recovery / 100) * 0.3) +
                (min(1.0, first_outcome.reservoir_reduction * 10) * 0.1)
            )
            
            safety_score = 1.0 - (len(first_outcome.adverse_events) * 0.1)
            cost_score = 1.0 / (1.0 + np.log(first_outcome.cost_effectiveness + 1))
            
            # Weighted score
            score = (
                efficacy_score * 0.5 +
                safety_score * 0.3 +
                cost_score * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_sequence = [first_line]
                best_outcome = first_outcome
        
        return {
            'optimal_sequence': best_sequence,
            'predicted_outcome': best_outcome,
            'composite_score': best_score,
            'rationale': f"Selected {best_sequence[0]} based on efficacy, safety, and cost-effectiveness profile"
        }
    
    def simulate_shock_and_kill_approach(self, lra_agent: str, 
                                       immune_enhancer: str,
                                       patient_characteristics: Dict,
                                       cycles: int = 4) -> Dict:
        """
        Simulate a shock-and-kill approach.
        
        Args:
            lra_agent: Name of the latency reversing agent
            immune_enhancer: Name of the immune enhancer (e.g., anti-PD1)
            patient_characteristics: Patient characteristics
            cycles: Number of shock-kill cycles
            
        Returns:
            Shock-and-kill outcome
        """
        total_reservoir_reduction = 0.0
        cumulative_adverse_events = []
        quality_of_life = patient_characteristics.get('baseline_qol', 0.7)
        
        # Calculate timing for cycles
        cycle_interval = 90  # days between cycles
        total_duration = cycles * cycle_interval
        
        for cycle in range(cycles):
            # Administer LRA
            lra_outcome = self.simulate_single_intervention(
                lra_agent, patient_characteristics, duration_days=14  # LRA course length
            )
            
            # Administer immune enhancer
            immune_outcome = self.simulate_single_intervention(
                immune_enhancer, patient_characteristics, duration_days=28  # Immune therapy length
            )
            
            # Calculate synergistic reservoir reduction
            lra_reduction = lra_outcome.reservoir_reduction
            immune_potentiation = 0.5  # 50% additional killing with immune enhancement
            cycle_reduction = lra_reduction * (1 + immune_potentiation)
            
            # Apply reduction (log scale)
            total_reservoir_reduction = total_reservoir_reduction + cycle_reduction
            
            # Track adverse events
            cumulative_adverse_events.extend(lra_outcome.adverse_events)
            cumulative_adverse_events.extend(immune_outcome.adverse_events)
            
            # Update quality of life
            qol_penalty = len(set(cumulative_adverse_events)) * 0.04
            quality_of_life = max(0.0, quality_of_life - qol_penalty)
        
        # Calculate final outcomes
        viral_suppression = True  # Assuming maintained ART
        cd4_recovery = patient_characteristics.get('baseline_cd4', 200) * 1.1  # 10% improvement
        
        # Calculate costs
        lra_cost = self.interventions[lra_agent].cost * cycles
        immune_cost = self.interventions[immune_enhancer].cost * (cycles * 28 / 365)  # Prorated
        monitoring_cost = self.cost_parameters['monitoring_cost'] * (total_duration / 365.0)
        total_cost = lra_cost + immune_cost + monitoring_cost
        
        # Calculate cost-effectiveness
        qaly_gained = quality_of_life * (total_duration / 365.0) * self.cost_parameters['quality_of_life_weight']
        cost_per_qaly = total_cost / qaly_gained if qaly_gained > 0 else float('inf')
        
        return {
            'reservoir_reduction_log': total_reservoir_reduction,
            'viral_suppression': viral_suppression,
            'cd4_recovery': cd4_recovery,
            'adverse_events': list(set(cumulative_adverse_events)),
            'quality_of_life': quality_of_life,
            'cost_effectiveness': cost_per_qaly,
            'cycles_completed': cycles,
            'total_duration': total_duration
        }
    
    def compare_interventions(self, intervention_names: List[str],
                            patient_characteristics: Dict,
                            duration_days: int = 365) -> pd.DataFrame:
        """
        Compare multiple interventions head-to-head.
        
        Args:
            intervention_names: List of intervention names to compare
            patient_characteristics: Patient characteristics
            duration_days: Duration for comparison
            
        Returns:
            Comparison dataframe
        """
        comparison_data = []
        
        for intervention_name in intervention_names:
            outcome = self.simulate_single_intervention(
                intervention_name, patient_characteristics, duration_days
            )
            
            intervention = self.interventions[intervention_name]
            
            comparison_data.append({
                'Intervention': intervention_name,
                'Type': intervention.intervention_type.value,
                'Viral Suppression': outcome.viral_suppression,
                'CD4 Recovery': outcome.cd4_recovery,
                'Reservoir Reduction': outcome.reservoir_reduction,
                'Time to Failure (days)': outcome.time_to_failure,
                'Adverse Events': len(outcome.adverse_events),
                'Quality of Life': outcome.quality_of_life,
                'Cost-Effectiveness': outcome.cost_effectiveness,
                'Mechanism': intervention.mechanism_of_action
            })
        
        return pd.DataFrame(comparison_data)


def run_therapeutic_intervention_demo():
    """Demo function showing how to use the TherapeuticInterventionEngine."""
    print("Starting HIV Therapeutic Intervention Modeling Demo...")
    
    # Initialize the engine
    intervention_engine = TherapeuticInterventionEngine()
    
    # Example patient characteristics
    patient = {
        'age': 35,
        'baseline_cd4': 200,
        'baseline_vl_log10': 4.5,
        'baseline_qol': 0.7,
        'hla_type': 'B*57:01',
        'resistance_profile': []
    }
    
    # Example 1: Simulate single intervention
    print("\n1. Single Intervention Simulation:")
    outcome = intervention_engine.simulate_single_intervention(
        "TAF-FTC-DTG", patient, duration_days=365
    )
    
    print(f"   Intervention: TAF-FTC-DTG")
    print(f"   Viral suppression: {outcome.viral_suppression}")
    print(f"   CD4 recovery: {outcome.cd4_recovery:.1f} cells/μL")
    print(f"   Adverse events: {len(outcome.adverse_events)} ({outcome.adverse_events})")
    print(f"   Quality of life: {outcome.quality_of_life:.2f}")
    print(f"   Cost-effectiveness: ${outcome.cost_effectiveness:,.0f}/QALY")
    
    # Example 2: Simulate combination intervention
    print("\n2. Combination Intervention Simulation:")
    combo_outcome = intervention_engine.simulate_combination_intervention(
        ["TAF-FTC-DTG", "Vorinostat"], patient, duration_days=180
    )
    
    print(f"   Combination: TAF-FTC-DTG + Vorinostat")
    print(f"   Viral suppression: {combo_outcome.viral_suppression}")
    print(f"   CD4 recovery: {combo_outcome.cd4_recovery:.1f} cells/μL")
    print(f"   Reservoir reduction: {combo_outcome.reservoir_reduction:.2f} log10")
    print(f"   Adverse events: {len(combo_outcome.adverse_events)}")
    print(f"   Quality of life: {combo_outcome.quality_of_life:.2f}")
    
    # Example 3: Optimize treatment sequence
    print("\n3. Treatment Sequence Optimization:")
    optimization = intervention_engine.optimize_treatment_sequence(
        patient, 
        first_line_options=["TDF-FTC-EFV", "TAF-FTC-DTG"],
        duration_days=1095  # 3 years
    )
    
    print(f"   Optimal sequence: {optimization['optimal_sequence']}")
    print(f"   Composite score: {optimization['composite_score']:.3f}")
    print(f"   Rationale: {optimization['rationale']}")
    
    # Example 4: Shock and kill approach
    print("\n4. Shock and Kill Simulation:")
    shock_kill_result = intervention_engine.simulate_shock_and_kill_approach(
        "Vorinostat", "Anti-PD1", patient, cycles=4
    )
    
    print(f"   Reservoir reduction: {shock_kill_result['reservoir_reduction_log']:.2f} log10")
    print(f"   Adverse events: {len(shock_kill_result['adverse_events'])}")
    print(f"   Quality of life: {shock_kill_result['quality_of_life']:.2f}")
    print(f"   Cost-effectiveness: ${shock_kill_result['cost_effectiveness']:,.0f}/QALY")
    print(f"   Cycles completed: {shock_kill_result['cycles_completed']}")
    
    # Example 5: Compare interventions
    print("\n5. Intervention Comparison:")
    comparison_df = intervention_engine.compare_interventions(
        ["TDF-FTC-EFV", "TAF-FTC-DTG", "Vorinostat", "Anti-PD1"], 
        patient, 
        duration_days=365
    )
    
    print("   Comparison Results:")
    for _, row in comparison_df.iterrows():
        print(f"     {row['Intervention']}: VS={row['Viral Suppression']}, "
              f"CD4={row['CD4 Recovery']:.1f}, AE={row['Adverse Events']}, "
              f"QoL={row['Quality of Life']:.2f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_therapeutic_intervention_demo()