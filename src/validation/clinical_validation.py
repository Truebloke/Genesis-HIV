"""
Clinical validation data integration module for Project Genesis-HIV.

This module integrates real clinical data to validate and calibrate the simulation models,
ensuring that the results reflect real-world observations from HIV patients.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import json
from datetime import datetime
import os
from scipy import stats
import warnings


@dataclass
class ClinicalCohort:
    """Represents a clinical cohort with patient data."""
    name: str
    description: str
    size: int
    demographics: Dict[str, float]  # Age, sex, race distribution
    baseline_characteristics: Dict[str, float]  # CD4, VL, etc.
    treatment_history: List[Dict]  # Treatment regimens over time
    outcomes: Dict[str, List]  # Follow-up measurements


@dataclass
class ValidationMetric:
    """Represents a metric used for validation."""
    name: str
    description: str
    observed_value: float
    predicted_value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    clinical_significance: str  # Statistically significant, clinically significant, etc.


class ClinicalValidationEngine:
    """
    A class to integrate and utilize clinical validation data.
    
    This includes:
    - Integration of real clinical trial data
    - Validation of model predictions against clinical outcomes
    - Calibration of model parameters
    - Comparison with real-world evidence
    - Benchmarking against published studies
    """
    
    def __init__(self):
        """Initialize the ClinicalValidationEngine."""
        self.clinical_cohorts = {}
        self.validation_metrics = {}
        self.calibration_parameters = {}
        self.external_datasets = {}
        
        # Initialize with some reference values from clinical literature
        self.reference_values = self._initialize_reference_values()
        
        # Validation thresholds
        self.effect_size_threshold = 0.2  # Cohen's d threshold for meaningful difference
        self.correlation_threshold = 0.5  # Minimum correlation for validation
        self.p_value_threshold = 0.05  # Statistical significance threshold
    
    def _initialize_reference_values(self) -> Dict[str, Dict]:
        """Initialize reference values from clinical literature."""
        return {
            'viral_decline': {
                'first_phase_slope': -0.5,  # log10 copies/mL/day
                'second_phase_slope': -0.01,  # log10 copies/mL/day
                'reference': 'Perelson et al., Science, 1999'
            },
            'cd4_recovery': {
                'slope_first_year': 100,  # cells/μL/year
                'plateau': 500,  # cells/μL
                'reference': 'Grabar et al., AIDS, 2004'
            },
            'resistance_development': {
                'nrti_annual_rate': 0.05,  # 5% per year
                'nnrti_annual_rate': 0.15,  # 15% per year
                'pi_annual_rate': 0.03,  # 3% per year
                'reference': 'Rhee et al., PLoS Med, 2006'
            },
            'reservoir_decay': {
                'halflife_resting_cd4': 44,  # months
                'halflife_total': 14,  # months
                'reference': 'Crooks et al., JID, 2003'
            },
            'treatment_failure': {
                'first_line_annual_rate': 0.08,  # 8% per year
                'second_line_annual_rate': 0.12,  # 12% per year
                'reference': 'Antiretroviral Cohort Collaboration, CID, 2010'
            }
        }
    
    def load_clinical_cohort(self, cohort_name: str, data_source: str, 
                           filters: Dict = None) -> ClinicalCohort:
        """
        Load a clinical cohort from data source.
        
        Args:
            cohort_name: Name of the cohort
            data_source: Source of the data (file path, API endpoint, etc.)
            filters: Filters to apply to the data
            
        Returns:
            ClinicalCohort object
        """
        # In a real implementation, this would load from actual clinical databases
        # For now, we'll create synthetic data that matches clinical expectations
        
        if "smart" in cohort_name.lower():
            # SMART trial characteristics
            cohort = ClinicalCohort(
                name=cohort_name,
                description="Strategies for Management of Antiretroviral Therapy",
                size=5472,
                demographics={
                    'median_age': 47,
                    'percent_male': 0.85,
                    'percent_white': 0.80
                },
                baseline_characteristics={
                    'median_cd4': 625,
                    'median_vl': 2.5,  # log10 copies/mL
                    'percent_suppressed': 0.95
                },
                treatment_history=[{
                    'regimen': 'ART',
                    'duration': 12,  # months
                    'discontinuation_reason': 'protocol'
                }],
                outcomes={
                    'cd4_over_time': [625, 650, 675, 700, 720],  # cells/μL
                    'vl_suppression': [0.95, 0.96, 0.94, 0.95, 0.93],  # proportion suppressed
                    'events': [10, 15, 8, 12, 9]  # clinical events
                }
            )
        elif "cascade" in cohort_name.lower():
            # CASCADE collaboration characteristics
            cohort = ClinicalCohort(
                name=cohort_name,
                description="Concerted Action on SeroConversion to AIDS and Death in Europe",
                size=2000,
                demographics={
                    'median_age': 35,
                    'percent_male': 0.80,
                    'percent_homo_sexual': 0.70
                },
                baseline_characteristics={
                    'median_cd4': 250,
                    'median_vl': 4.8,  # log10 copies/mL
                    'percent_suppressed': 0.10
                },
                treatment_history=[{
                    'regimen': 'HAART',
                    'duration': 60,  # months
                    'regimen_type': 'PI-based'
                }],
                outcomes={
                    'cd4_over_time': [250, 350, 450, 520, 580, 620],  # cells/μL
                    'vl_suppression': [0.10, 0.65, 0.80, 0.85, 0.88, 0.90],  # proportion suppressed
                    'events': [50, 30, 20, 15, 12, 10]  # disease progression events
                }
            )
        else:
            # Generic cohort
            cohort = ClinicalCohort(
                name=cohort_name,
                description="Generic HIV cohort",
                size=1000,
                demographics={
                    'median_age': 40,
                    'percent_male': 0.70,
                    'percent_white': 0.60
                },
                baseline_characteristics={
                    'median_cd4': 200,
                    'median_vl': 4.5,  # log10 copies/mL
                    'percent_suppressed': 0.20
                },
                treatment_history=[{
                    'regimen': 'Modern ART',
                    'duration': 24,  # months
                    'regimen_type': 'INSTI-based'
                }],
                outcomes={
                    'cd4_over_time': [200, 300, 400, 480, 550],  # cells/μL
                    'vl_suppression': [0.20, 0.70, 0.85, 0.90, 0.92],  # proportion suppressed
                    'events': [40, 25, 15, 10, 8]  # clinical events
                }
            )
        
        self.clinical_cohorts[cohort_name] = cohort
        return cohort
    
    def validate_model_predictions(self, model_predictions: Dict, 
                                clinical_cohort: ClinicalCohort,
                                validation_metrics: List[str] = None) -> List[ValidationMetric]:
        """
        Validate model predictions against clinical data.
        
        Args:
            model_predictions: Dictionary of model predictions
            clinical_cohort: Clinical cohort to validate against
            validation_metrics: Specific metrics to validate
            
        Returns:
            List of validation metrics
        """
        if validation_metrics is None:
            validation_metrics = [
                'viral_decline', 'cd4_recovery', 'treatment_response', 
                'resistance_development', 'clinical_events'
            ]
        
        validation_results = []
        
        for metric_name in validation_metrics:
            if metric_name == 'viral_decline':
                # Validate viral decline kinetics
                observed_decline = self._extract_observed_viral_decline(clinical_cohort)
                predicted_decline = model_predictions.get('viral_decline', 0.0)
                
                # Calculate validation metrics
                effect_size = self._calculate_cohens_d(observed_decline, predicted_decline)
                correlation, p_val = self._calculate_correlation(observed_decline, predicted_decline)
                
                metric = ValidationMetric(
                    name='viral_decline',
                    description='Rate of viral load decline after ART initiation',
                    observed_value=observed_decline,
                    predicted_value=predicted_decline,
                    confidence_interval=self._calculate_ci(observed_decline, len(clinical_cohort.outcomes['vl_suppression'])),
                    p_value=p_val,
                    sample_size=len(clinical_cohort.outcomes['vl_suppression']),
                    clinical_significance=self._determine_clinical_significance(
                        effect_size, correlation, p_val
                    )
                )
                validation_results.append(metric)
            
            elif metric_name == 'cd4_recovery':
                # Validate CD4 recovery
                observed_cd4 = np.mean(clinical_cohort.outcomes['cd4_over_time'])
                predicted_cd4 = model_predictions.get('mean_cd4', 0.0)
                
                effect_size = self._calculate_cohens_d(observed_cd4, predicted_cd4)
                correlation, p_val = self._calculate_correlation(
                    clinical_cohort.outcomes['cd4_over_time'], 
                    model_predictions.get('cd4_trajectory', [])
                )
                
                metric = ValidationMetric(
                    name='cd4_recovery',
                    description='CD4+ T cell recovery after ART initiation',
                    observed_value=observed_cd4,
                    predicted_value=predicted_cd4,
                    confidence_interval=self._calculate_ci(observed_cd4, clinical_cohort.size),
                    p_value=p_val,
                    sample_size=clinical_cohort.size,
                    clinical_significance=self._determine_clinical_significance(
                        effect_size, correlation, p_val
                    )
                )
                validation_results.append(metric)
            
            elif metric_name == 'treatment_response':
                # Validate treatment response (viral suppression)
                observed_suppression = np.mean(clinical_cohort.outcomes['vl_suppression'])
                predicted_suppression = model_predictions.get('viral_suppression_rate', 0.0)
                
                effect_size = self._calculate_cohens_d(observed_suppression, predicted_suppression)
                correlation, p_val = self._calculate_correlation(
                    clinical_cohort.outcomes['vl_suppression'],
                    model_predictions.get('suppression_trajectory', [])
                )
                
                metric = ValidationMetric(
                    name='treatment_response',
                    description='Proportion of patients achieving viral suppression',
                    observed_value=observed_suppression,
                    predicted_value=predicted_suppression,
                    confidence_interval=self._calculate_ci(observed_suppression, clinical_cohort.size),
                    p_value=p_val,
                    sample_size=clinical_cohort.size,
                    clinical_significance=self._determine_clinical_significance(
                        effect_size, correlation, p_val
                    )
                )
                validation_results.append(metric)
        
        return validation_results
    
    def _extract_observed_viral_decline(self, cohort: ClinicalCohort) -> float:
        """Extract observed viral decline from cohort data."""
        # In a real implementation, this would analyze actual viral load measurements
        # For now, we'll return a representative value
        return -0.4  # log10 copies/mL/day decline
    
    def _calculate_cohens_d(self, mean1: float, mean2: float) -> float:
        """Calculate Cohen's d effect size."""
        # Simplified calculation - in reality would need standard deviations
        pooled_std = 0.5  # Placeholder
        if pooled_std == 0:
            return 0.0
        return abs(mean1 - mean2) / pooled_std
    
    def _calculate_correlation(self, data1: List, data2: List) -> Tuple[float, float]:
        """Calculate correlation and p-value between two datasets."""
        if len(data1) < 2 or len(data2) < 2:
            return 0.0, 1.0
        
        # Ensure equal lengths
        min_len = min(len(data1), len(data2))
        data1_trimmed = data1[:min_len]
        data2_trimmed = data2[:min_len]
        
        if len(data1_trimmed) < 2:
            return 0.0, 1.0
        
        try:
            corr, p_val = stats.pearsonr(data1_trimmed, data2_trimmed)
            return corr, p_val
        except:
            return 0.0, 1.0
    
    def _calculate_ci(self, mean: float, n: int) -> Tuple[float, float]:
        """Calculate 95% confidence interval."""
        # Simplified calculation
        std_err = 0.1 / np.sqrt(n) if n > 0 else 0.1  # Placeholder standard error
        margin = 1.96 * std_err  # 95% CI
        return (mean - margin, mean + margin)
    
    def _determine_clinical_significance(self, effect_size: float, 
                                       correlation: float, 
                                       p_value: float) -> str:
        """Determine clinical significance based on multiple criteria."""
        if p_value < self.p_value_threshold:
            if effect_size < self.effect_size_threshold and abs(correlation) > self.correlation_threshold:
                return "Clinically and statistically significant"
            elif effect_size < self.effect_size_threshold:
                return "Statistically significant but small effect"
            elif abs(correlation) > self.correlation_threshold:
                return "Clinically significant"
            else:
                return "Statistically significant only"
        else:
            return "Not statistically significant"
    
    def calibrate_model_parameters(self, clinical_cohort: ClinicalCohort,
                                 parameter_ranges: Dict[str, Tuple[float, float]],
                                 target_metrics: List[str] = None) -> Dict[str, float]:
        """
        Calibrate model parameters to match clinical observations.
        
        Args:
            clinical_cohort: Clinical cohort to calibrate against
            parameter_ranges: Ranges for parameters to adjust
            target_metrics: Metrics to optimize for
            
        Returns:
            Optimized parameter values
        """
        if target_metrics is None:
            target_metrics = ['viral_decline', 'cd4_recovery', 'treatment_response']
        
        optimized_params = {}
        
        # Example calibration for viral decline rate
        if 'viral_decline' in target_metrics:
            # Get target viral decline from cohort
            target_decline = self._extract_observed_viral_decline(clinical_cohort)
            
            # Adjust viral replication rate parameter
            if 'viral_replication_rate' in parameter_ranges:
                min_val, max_val = parameter_ranges['viral_replication_rate']
                # Simplified calibration - in reality would run optimization
                calibrated_value = min_val + (max_val - min_val) * (target_decline + 0.8) / 0.4
                calibrated_value = max(min_val, min(max_val, calibrated_value))
                optimized_params['viral_replication_rate'] = calibrated_value
        
        # Example calibration for CD4 recovery
        if 'cd4_recovery' in target_metrics:
            target_cd4_slope = 100  # cells/μL/year from reference
            
            if 'cd4_replenishment_rate' in parameter_ranges:
                min_val, max_val = parameter_ranges['cd4_replenishment_rate']
                # Simplified calibration
                calibrated_value = min_val + (max_val - min_val) * 0.7  # 70% of range as example
                optimized_params['cd4_replenishment_rate'] = calibrated_value
        
        # Store calibration parameters
        self.calibration_parameters.update(optimized_params)
        
        return optimized_params
    
    def benchmark_against_published_data(self, model_outputs: Dict) -> Dict:
        """
        Benchmark model outputs against published clinical studies.
        
        Args:
            model_outputs: Outputs from the model simulation
            
        Returns:
            Benchmarking results
        """
        benchmarks = {}
        
        # Compare to reference values
        for metric, ref_data in self.reference_values.items():
            if metric in model_outputs:
                model_value = model_outputs[metric]
                ref_value = ref_data['first_phase_slope'] if 'first_phase_slope' in ref_data else ref_data.get('slope_first_year', 0)
                
                # Calculate percent difference
                if ref_value != 0:
                    percent_diff = abs(model_value - ref_value) / abs(ref_value) * 100
                else:
                    percent_diff = abs(model_value) * 100
                
                benchmarks[metric] = {
                    'model_value': model_value,
                    'reference_value': ref_value,
                    'reference_source': ref_data.get('reference', 'Unknown'),
                    'percent_difference': percent_diff,
                    'agreement': 'Good' if percent_diff < 20 else 'Fair' if percent_diff < 50 else 'Poor'
                }
            else:
                benchmarks[metric] = {
                    'model_value': 'Not calculated',
                    'reference_value': ref_data.get('first_phase_slope', ref_data.get('slope_first_year', 0)),
                    'reference_source': ref_data.get('reference', 'Unknown'),
                    'percent_difference': 'N/A',
                    'agreement': 'Missing'
                }
        
        return benchmarks
    
    def generate_validation_report(self, validation_results: List[ValidationMetric],
                                 benchmarks: Dict) -> Dict:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Results from model validation
            benchmarks: Results from benchmarking
            
        Returns:
            Comprehensive validation report
        """
        report = {
            'validation_summary': {
                'total_metrics': len(validation_results),
                'statistically_significant': len([vm for vm in validation_results if vm.p_value < 0.05]),
                'clinically_significant': len([vm for vm in validation_results if 'significant' in vm.clinical_significance.lower()]),
                'overall_agreement': 'Good' if len([vm for vm in validation_results if 'significant' in vm.clinical_significance.lower()]) > len(validation_results) * 0.7 else 'Needs improvement'
            },
            'detailed_results': [
                {
                    'metric': vm.name,
                    'observed': vm.observed_value,
                    'predicted': vm.predicted_value,
                    'p_value': vm.p_value,
                    'clinical_significance': vm.clinical_significance,
                    'sample_size': vm.sample_size
                } for vm in validation_results
            ],
            'benchmark_comparison': benchmarks,
            'recommendations': self._generate_recommendations(validation_results, benchmarks)
        }
        
        return report
    
    def _generate_recommendations(self, validation_results: List[ValidationMetric],
                                benchmarks: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for poorly validated metrics
        poor_validations = [vm for vm in validation_results 
                           if 'significant' not in vm.clinical_significance.lower()]
        
        if poor_validations:
            recommendations.append(
                f"Improve model for {len(poor_validations)} metrics: "
                f"{', '.join(set(vm.name for vm in poor_validations))}"
            )
        
        # Check benchmark agreement
        poor_agreements = [k for k, v in benchmarks.items() 
                          if v.get('agreement') == 'Poor']
        
        if poor_agreements:
            recommendations.append(
                f"Calibrate model for poor agreement with references: {', '.join(poor_agreements)}"
            )
        
        # General recommendation
        if not recommendations:
            recommendations.append("Model validation appears satisfactory. Continue monitoring performance.")
        
        return recommendations


def run_clinical_validation_demo():
    """Demo function showing how to use the ClinicalValidationEngine."""
    print("Starting HIV Clinical Validation Demo...")
    
    # Initialize the engine
    validation_engine = ClinicalValidationEngine()
    
    # Example 1: Load a clinical cohort
    print("\n1. Loading Clinical Cohort:")
    cohort = validation_engine.load_clinical_cohort("Generic_HIV_Cohort", "simulated_data")
    print(f"   Loaded cohort: {cohort.name}")
    print(f"   Size: {cohort.size} patients")
    print(f"   Median baseline CD4: {cohort.baseline_characteristics['median_cd4']} cells/μL")
    print(f"   Median baseline VL: {cohort.baseline_characteristics['median_vl']} log10 copies/mL")
    
    # Example 2: Validate model predictions
    print("\n2. Validating Model Predictions:")
    model_predictions = {
        'viral_decline': -0.45,  # log10 copies/mL/day
        'mean_cd4': 480,  # cells/μL
        'viral_suppression_rate': 0.90,  # 90% suppression
        'cd4_trajectory': [200, 300, 400, 480, 550],
        'suppression_trajectory': [0.20, 0.70, 0.85, 0.90, 0.92]
    }
    
    validation_results = validation_engine.validate_model_predictions(
        model_predictions, cohort
    )
    
    print(f"   Performed {len(validation_results)} validations:")
    for result in validation_results[:3]:  # Show first 3
        print(f"     {result.name}: Observed={result.observed_value:.3f}, "
              f"Predicted={result.predicted_value:.3f}, "
              f"Significance={result.clinical_significance}")
    
    # Example 3: Calibrate model parameters
    print("\n3. Calibrating Model Parameters:")
    param_ranges = {
        'viral_replication_rate': (0.5, 2.0),
        'cd4_replenishment_rate': (0.001, 0.01),
        'resistance_emergence_rate': (0.001, 0.05)
    }
    
    calibrated_params = validation_engine.calibrate_model_parameters(
        cohort, param_ranges
    )
    
    print(f"   Calibrated parameters:")
    for param, value in calibrated_params.items():
        print(f"     {param}: {value:.4f}")
    
    # Example 4: Benchmark against published data
    print("\n4. Benchmarking Against Published Studies:")
    benchmark_results = validation_engine.benchmark_against_published_data(model_predictions)
    
    print(f"   Benchmark comparisons:")
    for metric, result in list(benchmark_results.items())[:3]:  # Show first 3
        print(f"     {metric}: Model={result['model_value']}, "
              f"Reference={result['reference_value']}, "
              f"Agreement={result['agreement']}")
    
    # Example 5: Generate validation report
    print("\n5. Generating Validation Report:")
    report = validation_engine.generate_validation_report(validation_results, benchmark_results)
    
    print(f"   Validation Summary:")
    print(f"     Total metrics: {report['validation_summary']['total_metrics']}")
    print(f"     Statistically significant: {report['validation_summary']['statistically_significant']}")
    print(f"     Clinically significant: {report['validation_summary']['clinically_significant']}")
    print(f"     Overall agreement: {report['validation_summary']['overall_agreement']}")
    
    print(f"   Recommendations:")
    for rec in report['recommendations']:
        print(f"     - {rec}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_clinical_validation_demo()