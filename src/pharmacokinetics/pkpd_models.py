"""
Pharmacokinetics/Pharmacodynamics (PK/PD) modeling module for Project Genesis-HIV.

This module provides realistic PK/PD models for HIV medications, including
absorption, distribution, metabolism, excretion, and drug effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from enum import Enum


class DrugClass(Enum):
    """Classes of HIV drugs."""
    NRTI = "NRTI"  # Nucleoside Reverse Transcriptase Inhibitors
    NNRTI = "NNRTI"  # Non-Nucleoside Reverse Transcriptase Inhibitors
    PI = "PI"  # Protease Inhibitors
    INSTI = "INSTI"  # Integrase Strand Transfer Inhibitors
    ENTRY_INHIBITOR = "Entry_Inhibitor"
    FUSION_INHIBITOR = "Fusion_Inhibitor"


@dataclass
class PKParameters:
    """Pharmacokinetic parameters for a drug."""
    dose: float  # mg
    dosing_interval: int  # hours
    absorption_rate: float  # 1/hour
    volume_distribution: float  # L/kg
    clearance: float  # L/hour
    bioavailability: float  # 0-1
    half_life: float  # hours
    protein_binding: float  # fraction bound (0-1)
    elimination_route: str  # hepatic, renal, mixed


@dataclass
class PDEffect:
    """Pharmacodynamic effect parameters."""
    ic50: float  # 50% inhibitory concentration (mg/L)
    ic90: float  # 90% inhibitory concentration (mg/L)
    hill_coefficient: float  # Hill coefficient for sigmoidicity
    max_effect: float  # Maximum achievable effect (0-1)
    resistance_factor: float  # Multiplier for resistant virus
    synergism_factors: Dict[str, float]  # Synergism with other drugs


class PKPDEngine:
    """
    A class to model pharmacokinetics and pharmacodynamics of HIV drugs.
    
    This includes:
    - Absorption, distribution, metabolism, and excretion (ADME)
    - Drug concentration over time
    - Effect on viral replication
    - Drug interactions
    - Individual patient factors
    - Resistance considerations
    """
    
    def __init__(self):
        """Initialize the PKPDEngine with drug parameters."""
        self.drug_parameters = self._initialize_drug_parameters()
        self.patient_factors = self._initialize_patient_factors()
        self.drug_interactions = self._initialize_drug_interactions()
        
        # Simulation parameters
        self.time_step = 0.1  # hours
        self.max_time = 24 * 28  # 28 days in hours
    
    def _initialize_drug_parameters(self) -> Dict[str, Tuple[PKParameters, PDEffect]]:
        """Initialize PK/PD parameters for common HIV drugs."""
        return {
            # NRTIs
            "Tenofovir_DF": (
                PKParameters(
                    dose=300,  # mg
                    dosing_interval=24,  # hours
                    absorption_rate=0.8,  # 1/hour
                    volume_distribution=1.3,  # L/kg
                    clearance=1.8,  # L/hour
                    bioavailability=0.7,  # 70%
                    half_life=17.0,  # hours
                    protein_binding=0.79,  # 79%
                    elimination_route="renal"
                ),
                PDEffect(
                    ic50=0.06,  # mg/L
                    ic90=0.25,  # mg/L
                    hill_coefficient=1.2,
                    max_effect=0.95,
                    resistance_factor=1.0,
                    synergism_factors={"Emtricitabine": 1.3}
                )
            ),
            "Emtricitabine": (
                PKParameters(
                    dose=200,
                    dosing_interval=24,
                    absorption_rate=0.9,
                    volume_distribution=1.5,
                    clearance=0.9,
                    bioavailability=0.94,
                    half_life=10.0,
                    protein_binding=0.04,
                    elimination_route="renal"
                ),
                PDEffect(
                    ic50=0.005,
                    ic90=0.02,
                    hill_coefficient=1.1,
                    max_effect=0.98,
                    resistance_factor=1.0,
                    synergism_factors={"Tenofovir_DF": 1.3}
                )
            ),
            "Lamivudine": (
                PKParameters(
                    dose=300,
                    dosing_interval=24,
                    absorption_rate=0.85,
                    volume_distribution=1.3,
                    clearance=0.8,
                    bioavailability=0.87,
                    half_life=5.0,
                    protein_binding=0.36,
                    elimination_route="renal"
                ),
                PDEffect(
                    ic50=0.006,
                    ic90=0.024,
                    hill_coefficient=1.0,
                    max_effect=0.95,
                    resistance_factor=1.0,
                    synergism_factors={}
                )
            ),
            
            # NNRTIs
            "Efavirenz": (
                PKParameters(
                    dose=600,
                    dosing_interval=24,
                    absorption_rate=0.6,
                    volume_distribution=3.6,
                    clearance=0.4,
                    bioavailability=0.94,
                    half_life=40.0,
                    protein_binding=0.99,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.04,
                    ic90=0.16,
                    hill_coefficient=1.5,
                    max_effect=0.99,
                    resistance_factor=1.0,
                    synergism_factors={}
                )
            ),
            "Rilpivirine": (
                PKParameters(
                    dose=25,
                    dosing_interval=24,
                    absorption_rate=0.7,
                    volume_distribution=10.4,
                    clearance=0.3,
                    bioavailability=0.85,
                    half_life=50.0,
                    protein_binding=0.998,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.045,
                    ic90=0.18,
                    hill_coefficient=1.4,
                    max_effect=0.98,
                    resistance_factor=1.0,
                    synergism_factors={}
                )
            ),
            
            # PIs
            "Atazanavir": (
                PKParameters(
                    dose=300,
                    dosing_interval=24,
                    absorption_rate=0.5,
                    volume_distribution=1.2,
                    clearance=0.25,
                    bioavailability=0.6,
                    half_life=7.0,
                    protein_binding=0.82,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.15,
                    ic90=0.6,
                    hill_coefficient=1.3,
                    max_effect=0.97,
                    resistance_factor=1.0,
                    synergism_factors={"Ritonavir": 3.0}  # Boosted
                )
            ),
            "Darunavir": (
                PKParameters(
                    dose=800,
                    dosing_interval=24,
                    absorption_rate=0.35,
                    volume_distribution=0.9,
                    clearance=0.15,
                    bioavailability=0.38,
                    half_life=15.0,
                    protein_binding=0.95,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.025,
                    ic90=0.1,
                    hill_coefficient=1.4,
                    max_effect=0.99,
                    resistance_factor=1.0,
                    synergism_factors={"Ritonavir": 3.0}  # Boosted
                )
            ),
            
            # INSTIs
            "Dolutegravir": (
                PKParameters(
                    dose=50,
                    dosing_interval=24,
                    absorption_rate=0.4,
                    volume_distribution=0.8,
                    clearance=0.1,
                    bioavailability=0.5,
                    half_life=14.0,
                    protein_binding=0.993,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.0032,
                    ic90=0.013,
                    hill_coefficient=2.0,
                    max_effect=0.995,
                    resistance_factor=1.0,
                    synergism_factors={}
                )
            ),
            "Bictegravir": (
                PKParameters(
                    dose=50,
                    dosing_interval=24,
                    absorption_rate=0.45,
                    volume_distribution=0.7,
                    clearance=0.08,
                    bioavailability=0.55,
                    half_life=17.0,
                    protein_binding=0.995,
                    elimination_route="hepatic"
                ),
                PDEffect(
                    ic50=0.024,
                    ic90=0.096,
                    hill_coefficient=1.8,
                    max_effect=0.99,
                    resistance_factor=1.0,
                    synergism_factors={}
                )
            )
        }
    
    def _initialize_patient_factors(self) -> Dict[str, float]:
        """Initialize patient-specific factors affecting PK/PD."""
        return {
            'weight': 70.0,  # kg
            'age': 35.0,  # years
            'sex': 'male',  # male/female
            'creatinine_clearance': 100.0,  # mL/min
            'albumin_level': 4.0,  # g/dL
            'hepatic_function': 1.0,  # 0-1 scale (1.0 = normal)
            'compliance': 0.95,  # 0-1 scale (fraction of doses taken)
            'food_effect': 1.0,  # Multiplier for food effect
            'drug_interactions': {},  # Other medications affecting clearance
        }
    
    def _initialize_drug_interactions(self) -> Dict[str, Dict[str, float]]:
        """Initialize known drug-drug interactions."""
        return {
            "Ritonavir": {
                "Atazanavir": 3.0,  # Ritonavir boosts Atazanavir
                "Darunavir": 3.0,   # Ritonavir boosts Darunavir
                "Dolutegravir": 1.0, # Minimal effect on INSTIs
            },
            "Cobicistat": {
                "Elvitegravir": 4.0,  # Cobicistat boosts Elvitegravir
                "Atazanavir": 2.5,    # Cobicistat boosts Atazanavir
            },
            "Efavirenz": {
                "Methadone": 0.5,     # Efavirenz induces Methadone metabolism
                "Warfarin": 0.8,      # Minor interaction
            }
        }
    
    def calculate_drug_concentration(self, drug_name: str, time_points: List[float],
                                   doses: List[Tuple[float, float]] = None) -> List[float]:
        """
        Calculate drug concentration over time using PK parameters.
        
        Args:
            drug_name: Name of the drug
            time_points: List of time points (hours) to calculate concentration
            doses: List of (dose_amount, time) tuples for actual dosing
            
        Returns:
            List of drug concentrations at each time point
        """
        if drug_name not in self.drug_parameters:
            raise ValueError(f"Unknown drug: {drug_name}")
        
        pk_params, _ = self.drug_parameters[drug_name]
        
        # If no specific dosing schedule provided, use standard parameters
        if doses is None:
            # Create a standard dosing schedule
            doses = []
            t = 0
            while t <= max(time_points):
                doses.append((pk_params.dose, t))
                t += pk_params.dosing_interval
        
        concentrations = []
        
        for t in time_points:
            conc = 0.0
            
            # Sum contributions from all previous doses
            for dose_amount, dose_time in doses:
                if t >= dose_time:
                    # Time since this dose was administered
                    dt = t - dose_time
                    
                    # Calculate concentration from this dose using PK model
                    # Using a simple one-compartment model with first-order absorption
                    ka = pk_params.absorption_rate
                    kel = pk_params.clearance / pk_params.volume_distribution
                    F = pk_params.bioavailability
                    
                    # Amount absorbed
                    amount_absorbed = F * dose_amount
                    
                    # Concentration calculation (simplified one-compartment model)
                    if ka != kel:
                        conc_contribution = (amount_absorbed * ka * F / 
                                           (pk_params.volume_distribution * (ka - kel))) * \
                                          (np.exp(-kel * dt) - np.exp(-ka * dt))
                    else:
                        # Special case when ka = kel
                        conc_contribution = (amount_absorbed * ka * F * dt * 
                                           np.exp(-kel * dt)) / pk_params.volume_distribution
                    
                    conc += max(0, conc_contribution)
            
            concentrations.append(conc)
        
        return concentrations
    
    def calculate_antiviral_effect(self, drug_name: str, concentration: float,
                                 resistance_mutations: List[str] = None) -> float:
        """
        Calculate the antiviral effect of a drug at a given concentration.
        
        Args:
            drug_name: Name of the drug
            concentration: Current drug concentration (mg/L)
            resistance_mutations: List of resistance mutations present
            
        Returns:
            Fractional inhibition of viral replication (0-1)
        """
        if drug_name not in self.drug_parameters:
            raise ValueError(f"Unknown drug: {drug_name}")
        
        _, pd_effect = self.drug_parameters[drug_name]
        
        # Apply resistance factor if mutations are present
        resistance_multiplier = 1.0
        if resistance_mutations:
            # In a real implementation, this would map specific mutations to resistance factors
            # For now, we'll use a simplified approach
            resistance_multiplier = pd_effect.resistance_factor ** len(resistance_mutations)
        
        # Calculate effect using Hill equation
        ic50_resistant = pd_effect.ic50 * resistance_multiplier
        
        effect = (pd_effect.max_effect * 
                 (concentration / ic50_resistant) ** pd_effect.hill_coefficient) / \
                (1 + (concentration / ic50_resistant) ** pd_effect.hill_coefficient)
        
        return min(effect, pd_effect.max_effect)
    
    def simulate_combination_therapy(self, drug_regimen: List[Tuple[str, List[Tuple[float, float]]]],
                                   time_points: List[float],
                                   resistance_profile: Dict[str, List[str]] = None) -> Dict:
        """
        Simulate combination therapy with multiple drugs.
        
        Args:
            drug_regimen: List of (drug_name, [(dose, time), ...]) tuples
            time_points: Time points to simulate
            resistance_profile: Dict mapping drug names to resistance mutations
            
        Returns:
            Simulation results including concentrations and effects
        """
        results = {
            'time_points': time_points,
            'drug_concentrations': {},
            'individual_effects': {},
            'combined_effect': [],
            'cumulative_exposure': {}
        }
        
        # Calculate concentrations for each drug
        for drug_name, dosing_schedule in drug_regimen:
            concentrations = self.calculate_drug_concentration(
                drug_name, time_points, dosing_schedule
            )
            results['drug_concentrations'][drug_name] = concentrations
            
            # Calculate individual effects
            resistance_muts = resistance_profile.get(drug_name, []) if resistance_profile else []
            individual_effects = [
                self.calculate_antiviral_effect(drug_name, conc, resistance_muts)
                for conc in concentrations
            ]
            results['individual_effects'][drug_name] = individual_effects
        
        # Calculate combined effect using Bliss independence model
        for i in range(len(time_points)):
            individual_contributions = []
            for drug_name, _ in drug_regimen:
                effect = results['individual_effects'][drug_name][i]
                individual_contributions.append(effect)
            
            # Combined effect: 1 - product of survivals
            combined_survival = 1.0
            for effect in individual_contributions:
                combined_survival *= (1 - effect)
            
            combined_effect = 1 - combined_survival
            results['combined_effect'].append(combined_effect)
        
        # Calculate cumulative exposure (AUC) for each drug
        for drug_name, _ in drug_regimen:
            concs = results['drug_concentrations'][drug_name]
            # Simple trapezoidal rule for AUC
            auc = np.trapz(concs, dx=self.time_step)
            results['cumulative_exposure'][drug_name] = auc
        
        return results
    
    def adjust_for_patient_factors(self, drug_name: str, patient_factors: Dict[str, float] = None) -> Dict:
        """
        Adjust PK parameters based on patient-specific factors.
        
        Args:
            drug_name: Name of the drug
            patient_factors: Patient-specific factors
            
        Returns:
            Adjusted PK parameters
        """
        if patient_factors is None:
            patient_factors = self.patient_factors
        
        original_pk, original_pd = self.drug_parameters[drug_name]
        
        # Create adjusted copy
        adjusted_pk = PKParameters(
            dose=original_pk.dose,
            dosing_interval=original_pk.dosing_interval,
            absorption_rate=original_pk.absorption_rate,
            volume_distribution=original_pk.volume_distribution,
            clearance=original_pk.clearance,
            bioavailability=original_pk.bioavailability,
            half_life=original_pk.half_life,
            protein_binding=original_pk.protein_binding,
            elimination_route=original_pk.elimination_route
        )
        
        # Adjust for patient factors
        # Weight adjustment for volume of distribution
        weight_factor = patient_factors['weight'] / 70.0  # Standard 70kg
        adjusted_pk.volume_distribution *= weight_factor
        
        # Renal function adjustment for renally cleared drugs
        if "renal" in original_pk.elimination_route.lower():
            cr_cl_factor = patient_factors['creatinine_clearance'] / 100.0
            adjusted_pk.clearance *= max(0.1, cr_cl_factor)  # Don't go below 10% of normal
        
        # Hepatic function adjustment for hepatically cleared drugs
        if "hepatic" in original_pk.elimination_route.lower():
            hep_factor = patient_factors['hepatic_function']
            adjusted_pk.clearance *= max(0.05, hep_factor)  # Don't go below 5% of normal
        
        # Protein binding adjustment
        albumin_factor = patient_factors['albumin_level'] / 4.0
        adjusted_pk.protein_binding = min(0.95, original_pk.protein_binding * albumin_factor)
        
        # Compliance adjustment (affects effective dose)
        compliance_factor = patient_factors['compliance']
        adjusted_pk.dose *= compliance_factor
        
        # Food effect adjustment
        adjusted_pk.bioavailability *= patient_factors['food_effect']
        
        # Calculate new half-life based on adjusted clearance and volume
        kel_adjusted = adjusted_pk.clearance / adjusted_pk.volume_distribution
        if kel_adjusted > 0:
            adjusted_pk.half_life = np.log(2) / kel_adjusted
        
        return {
            'pk_parameters': adjusted_pk,
            'pd_parameters': original_pd
        }
    
    def predict_adherence_effect(self, drug_name: str, adherence_rate: float,
                               duration_days: int = 30) -> Dict:
        """
        Predict the effect of imperfect adherence on drug exposure.
        
        Args:
            drug_name: Name of the drug
            adherence_rate: Fraction of doses taken (0-1)
            duration_days: Duration to simulate (days)
            
        Returns:
            Adherence effect analysis
        """
        # Create time points
        time_points = np.arange(0, duration_days * 24, self.time_step).tolist()
        
        # Create perfect dosing schedule
        pk_params, _ = self.drug_parameters[drug_name]
        perfect_doses = []
        t = 0
        while t <= duration_days * 24:
            perfect_doses.append((pk_params.dose, t))
            t += pk_params.dosing_interval
        
        # Create imperfect dosing schedule based on adherence
        imperfect_doses = []
        t = 0
        while t <= duration_days * 24:
            if random.random() < adherence_rate:
                imperfect_doses.append((pk_params.dose, t))
            t += pk_params.dosing_interval
        
        # Calculate concentrations for both schedules
        perfect_concs = self.calculate_drug_concentration(drug_name, time_points, perfect_doses)
        imperfect_concs = self.calculate_drug_concentration(drug_name, time_points, imperfect_doses)
        
        # Calculate AUCs
        perfect_auc = np.trapz(perfect_concs, dx=self.time_step)
        imperfect_auc = np.trapz(imperfect_concs, dx=self.time_step)
        
        # Calculate trough levels (minimum concentrations)
        perfect_trough = min(perfect_concs)
        imperfect_trough = min(imperfect_concs)
        
        # Calculate percentage of time above IC50
        ic50 = self.drug_parameters[drug_name][1].ic50
        perfect_above_ic50 = sum(1 for c in perfect_concs if c > ic50) / len(perfect_concs)
        imperfect_above_ic50 = sum(1 for c in imperfect_concs if c > ic50) / len(imperfect_concs)
        
        return {
            'adherence_rate': adherence_rate,
            'perfect_auc': perfect_auc,
            'imperfect_auc': imperfect_auc,
            'auc_ratio': imperfect_auc / perfect_auc if perfect_auc > 0 else 0,
            'perfect_trough': perfect_trough,
            'imperfect_trough': imperfect_trough,
            'perfect_percent_above_ic50': perfect_above_ic50 * 100,
            'imperfect_percent_above_ic50': imperfect_above_ic50 * 100,
            'exposure_loss': (perfect_auc - imperfect_auc) / perfect_auc if perfect_auc > 0 else 0
        }
    
    def model_drug_interactions(self, drug1: str, drug2: str) -> Dict:
        """
        Model the interaction between two drugs.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            Interaction analysis
        """
        interaction_factor = 1.0
        
        # Check if there's a known interaction
        if drug1 in self.drug_interactions and drug2 in self.drug_interactions[drug1]:
            interaction_factor = self.drug_interactions[drug1][drug2]
        elif drug2 in self.drug_interactions and drug1 in self.drug_interactions[drug2]:
            interaction_factor = self.drug_interactions[drug2][drug1]
        
        # Get original PK parameters
        pk1, pd1 = self.drug_parameters[drug1]
        pk2, pd2 = self.drug_parameters[drug2]
        
        # Adjust parameters based on interaction
        adj_pk1 = PKParameters(
            dose=pk1.dose,
            dosing_interval=pk1.dosing_interval,
            absorption_rate=pk1.absorption_rate,
            volume_distribution=pk1.volume_distribution,
            clearance=pk1.clearance / interaction_factor,  # Interaction affects clearance
            bioavailability=pk1.bioavailability,
            half_life=pk1.half_life * interaction_factor,  # Longer half-life if clearance decreased
            protein_binding=pk1.protein_binding,
            elimination_route=pk1.elimination_route
        )
        
        return {
            'interaction_factor': interaction_factor,
            'original_drug1_clearance': pk1.clearance,
            'adjusted_drug1_clearance': adj_pk1.clearance,
            'original_drug1_half_life': pk1.half_life,
            'adjusted_drug1_half_life': adj_pk1.half_life,
            'clinical_significance': 'Major' if interaction_factor > 2.0 or interaction_factor < 0.5 else 'Minor'
        }


def run_pkpd_demo():
    """Demo function showing how to use the PKPDEngine."""
    print("Starting HIV PK/PD Modeling Demo...")
    
    # Initialize the engine
    pkpd_engine = PKPDEngine()
    
    # Example 1: Calculate concentration profile for a single drug
    print("\n1. Single Drug PK Simulation:")
    time_points = [i * 0.5 for i in range(0, 48)]  # 48 hours, 30-min intervals
    concentrations = pkpd_engine.calculate_drug_concentration("Dolutegravir", time_points)
    
    print(f"   Dolutegravir concentrations at 24h: {concentrations[48]:.3f} mg/L")
    print(f"   Max concentration: {max(concentrations):.3f} mg/L")
    print(f"   Trough concentration: {min(concentrations):.3f} mg/L")
    
    # Example 2: Calculate antiviral effect
    print("\n2. Antiviral Effect Calculation:")
    effect = pkpd_engine.calculate_antiviral_effect("Dolutegravir", concentrations[48])
    print(f"   Antiviral effect at 24h: {effect:.3f} (fractional inhibition)")
    
    # Example 3: Simulate combination therapy
    print("\n3. Combination Therapy Simulation:")
    regimen = [
        ("Tenofovir_DF", [(300, t) for t in range(0, 72, 24)]),  # Daily dosing
        ("Emtricitabine", [(200, t) for t in range(0, 72, 24)]),
        ("Dolutegravir", [(50, t) for t in range(0, 72, 24)])
    ]
    
    combo_results = pkpd_engine.simulate_combination_therapy(regimen, time_points[:72])  # 72 time points = 36 hours
    print(f"   Combined effect at 24h: {combo_results['combined_effect'][48]:.3f}")
    print(f"   Tenofovir AUC: {combo_results['cumulative_exposure']['Tenofovir_DF']:.2f}")
    print(f"   Emtricitabine AUC: {combo_results['cumulative_exposure']['Emtricitabine']:.2f}")
    print(f"   Dolutegravir AUC: {combo_results['cumulative_exposure']['Dolutegravir']:.2f}")
    
    # Example 4: Patient-specific adjustments
    print("\n4. Patient-Specific Adjustments:")
    patient_factors = {
        'weight': 85.0,  # Heavier patient
        'creatinine_clearance': 80.0,  # Mild renal impairment
        'compliance': 0.85,  # 85% adherence
    }
    
    adjusted_params = pkpd_engine.adjust_for_patient_factors("Tenofovir_DF", patient_factors)
    print(f"   Adjusted clearance: {adjusted_params['pk_parameters'].clearance:.2f} L/h")
    print(f"   Adjusted volume of distribution: {adjusted_params['pk_parameters'].volume_distribution:.2f} L/kg")
    print(f"   Adjusted dose for compliance: {adjusted_params['pk_parameters'].dose:.0f} mg")
    
    # Example 5: Adherence effect
    print("\n5. Adherence Effect Analysis:")
    adherence_results = pkpd_engine.predict_adherence_effect("Dolutegravir", 0.80, duration_days=7)
    print(f"   AUC ratio (imperfect/perfect): {adherence_results['auc_ratio']:.3f}")
    print(f"   Exposure loss due to poor adherence: {adherence_results['exposure_loss']*100:.1f}%")
    print(f"   Time above IC50 (perfect): {adherence_results['perfect_percent_above_ic50']:.1f}%")
    print(f"   Time above IC50 (imperfect): {adherence_results['imperfect_percent_above_ic50']:.1f}%")
    
    # Example 6: Drug interaction
    print("\n6. Drug Interaction Analysis:")
    interaction_results = pkpd_engine.model_drug_interactions("Ritonavir", "Atazanavir")
    print(f"   Interaction factor: {interaction_results['interaction_factor']:.2f}")
    print(f"   Clinical significance: {interaction_results['clinical_significance']}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_pkpd_demo()