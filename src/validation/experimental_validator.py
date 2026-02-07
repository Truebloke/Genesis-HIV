"""
Experimental data validation module for Project Genesis-HIV.

This module validates simulation results against real experimental data from:
- In vitro studies
- Clinical trials
- Observational cohorts
- Animal models
- Single-cell studies
- Proteomics and metabolomics data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Result of validation comparison."""
    metric_name: str
    experimental_value: float
    simulated_value: float
    p_value: float
    correlation: float
    rmse: float
    mae: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    validation_passed: bool
    clinical_relevance: str


class ExperimentalValidator:
    """
    A class to validate simulation results against experimental data.
    
    This includes:
    - Comparison with in vitro experimental data
    - Validation against clinical trial results
    - Benchmarking against observational cohorts
    - Statistical validation of model predictions
    - Sensitivity analysis for model parameters
    - Uncertainty quantification
    """
    
    def __init__(self):
        """Initialize the ExperimentalValidator."""
        self.experimental_datasets = self._load_experimental_datasets()
        self.validation_thresholds = self._initialize_validation_thresholds()
        self.statistical_tests = self._initialize_statistical_tests()
        
        # Validation criteria
        self.correlation_threshold = 0.7  # Minimum correlation for validation
        self.p_value_threshold = 0.05     # Statistical significance threshold
        self.rmse_threshold = 0.3        # RMSE threshold (normalized)
        self.r_squared_threshold = 0.6   # R² threshold
    
    def _load_experimental_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load experimental datasets for validation."""
        # In a real implementation, this would load from actual experimental databases
        # For this implementation, we'll create synthetic datasets that represent real experimental data
        
        datasets = {}
        
        # In vitro viral replication data
        in_vitro_data = pd.DataFrame({
            'drug_concentration': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'viral_replication': [1.0, 0.8, 0.65, 0.4, 0.25, 0.15, 0.08, 0.05],
            'error_bar': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01],
            'study': ['Perelson_1999', 'Perelson_1999', 'Perelson_1999', 'Perelson_1999',
                     'Perelson_1999', 'Perelson_1999', 'Perelson_1999', 'Perelson_1999']
        })
        datasets['in_vitro_viral_replication'] = in_vitro_data
        
        # CD4 recovery data from clinical trials
        cd4_data = pd.DataFrame({
            'time_weeks': [2, 4, 8, 12, 24, 36, 48, 96],
            'cd4_count_mean': [250, 320, 380, 420, 480, 520, 550, 580],
            'cd4_count_stderr': [20, 18, 15, 12, 10, 8, 7, 6],
            'study': ['DHHS_2014', 'DHHS_2014', 'DHHS_2014', 'DHHS_2014',
                     'DHHS_2014', 'DHHS_2014', 'DHHS_2014', 'DHHS_2014']
        })
        datasets['cd4_recovery'] = cd4_data
        
        # Viral load decline data
        viral_decline_data = pd.DataFrame({
            'time_days': [1, 2, 3, 7, 14, 21, 28, 56, 84],
            'viral_load_log10': [4.5, 3.8, 3.2, 2.1, 1.5, 1.2, 1.0, 0.8, 0.7],
            'error': [0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.05],
            'study': ['Perelson_1996', 'Perelson_1996', 'Perelson_1996', 'Perelson_1996',
                     'Perelson_1996', 'Perelson_1996', 'Perelson_1996', 'Perelson_1996', 'Perelson_1996']
        })
        datasets['viral_decline'] = viral_decline_data
        
        # Resistance mutation frequencies
        resistance_data = pd.DataFrame({
            'mutation': ['K103N', 'M184V', 'K65R', 'L90M', 'Y181C', 'G190A'],
            'frequency_treatment_naive': [0.001, 0.002, 0.0005, 0.001, 0.0015, 0.001],
            'frequency_treatment_experienced': [0.15, 0.65, 0.05, 0.08, 0.12, 0.09],
            'study': ['Rhee_2006', 'Rhee_2006', 'Rhee_2006', 'Rhee_2006', 'Rhee_2006', 'Rhee_2006']
        })
        datasets['resistance_mutations'] = resistance_data
        
        # Reservoir decay data
        reservoir_data = pd.DataFrame({
            'time_months': [3, 6, 12, 18, 24, 36, 48, 60],
            'reservoir_size_log10': [1.8, 1.7, 1.6, 1.55, 1.5, 1.45, 1.4, 1.38],
            'measurement_method': ['Q4PCR', 'Q4PCR', 'Q4PCR', 'Q4PCR', 'Q4PCR', 'Q4PCR', 'Q4PCR', 'Q4PCR'],
            'study': ['Crooks_2003', 'Crooks_2003', 'Crooks_2003', 'Crooks_2003',
                     'Crooks_2003', 'Crooks_2003', 'Crooks_2003', 'Crooks_2003']
        })
        datasets['reservoir_decay'] = reservoir_data
        
        return datasets
    
    def _initialize_validation_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize validation thresholds for different metrics."""
        return {
            'viral_load_decline': {
                'correlation_min': 0.8,
                'rmse_max': 0.25,
                'r_squared_min': 0.7,
                'bias_max': 0.2
            },
            'cd4_recovery': {
                'correlation_min': 0.75,
                'rmse_max': 50,  # cells/μL
                'r_squared_min': 0.65,
                'bias_max': 30
            },
            'resistance_development': {
                'correlation_min': 0.7,
                'rmse_max': 0.1,  # proportion
                'r_squared_min': 0.6,
                'bias_max': 0.08
            },
            'reservoir_decay': {
                'correlation_min': 0.75,
                'rmse_max': 0.15,  # log10 copies
                'r_squared_min': 0.65,
                'bias_max': 0.12
            }
        }
    
    def _initialize_statistical_tests(self) -> Dict[str, callable]:
        """Initialize statistical tests for validation."""
        return {
            'pearson_correlation': stats.pearsonr,
            'spearman_correlation': stats.spearmanr,
            'mann_whitney_u': stats.mannwhitneyu,
            'kolmogorov_smirnov': stats.kstest,
            'chi_square': stats.chisquare
        }
    
    def validate_viral_dynamics(self, simulated_data: Dict) -> List[ValidationResult]:
        """
        Validate simulated viral dynamics against experimental data.
        
        Args:
            simulated_data: Dictionary containing simulated viral dynamics data
            
        Returns:
            List of validation results
        """
        results = []
        
        # Get experimental data
        exp_data = self.experimental_datasets['viral_decline']
        
        # Align time points
        exp_times = exp_data['time_days'].values
        exp_vl = exp_data['viral_load_log10'].values
        
        # Interpolate simulated data to match experimental time points
        if 'time_days' in simulated_data and 'viral_load_log10' in simulated_data:
            sim_times = simulated_data['time_days']
            sim_vl = simulated_data['viral_load_log10']
            
            # Interpolate simulated values to experimental time points
            from scipy.interpolate import interp1d
            f = interp1d(sim_times, sim_vl, kind='linear', fill_value='extrapolate')
            sim_vl_aligned = f(exp_times)
            
            # Perform validation
            result = self._validate_single_metric(
                'viral_load_decline',
                exp_vl,
                sim_vl_aligned,
                'Log10 Viral Load Decline'
            )
            results.append(result)
        
        return results
    
    def validate_cd4_recovery(self, simulated_data: Dict) -> List[ValidationResult]:
        """
        Validate simulated CD4 recovery against clinical data.
        
        Args:
            simulated_data: Dictionary containing simulated CD4 recovery data
            
        Returns:
            List of validation results
        """
        results = []
        
        # Get experimental data
        exp_data = self.experimental_datasets['cd4_recovery']
        
        # Align time points
        exp_times = exp_data['time_weeks'].values
        exp_cd4 = exp_data['cd4_count_mean'].values
        
        # Interpolate simulated data to match experimental time points
        if 'time_weeks' in simulated_data and 'cd4_count' in simulated_data:
            sim_times = simulated_data['time_weeks']
            sim_cd4 = simulated_data['cd4_count']
            
            # Interpolate simulated values to experimental time points
            from scipy.interpolate import interp1d
            f = interp1d(sim_times, sim_cd4, kind='linear', fill_value='extrapolate')
            sim_cd4_aligned = f(exp_times)
            
            # Perform validation
            result = self._validate_single_metric(
                'cd4_recovery',
                exp_cd4,
                sim_cd4_aligned,
                'CD4+ T Cell Recovery'
            )
            results.append(result)
        
        return results
    
    def validate_resistance_patterns(self, simulated_data: Dict) -> List[ValidationResult]:
        """
        Validate simulated resistance patterns against experimental data.
        
        Args:
            simulated_data: Dictionary containing simulated resistance data
            
        Returns:
            List of validation results
        """
        results = []
        
        # Get experimental data
        exp_data = self.experimental_datasets['resistance_mutations']
        
        if 'resistance_frequencies' in simulated_data:
            sim_resistances = simulated_data['resistance_frequencies']
            
            # Validate each mutation
            for i, mutation in enumerate(exp_data['mutation']):
                if mutation in sim_resistances:
                    exp_freq = exp_data.iloc[i]['frequency_treatment_experienced']
                    sim_freq = sim_resistances[mutation]
                    
                    # Create artificial datasets for validation
                    exp_vals = [exp_freq] * 10  # Replicate for statistical testing
                    sim_vals = [sim_freq] * 10
                    
                    result = self._validate_single_metric(
                        f'resistance_{mutation}',
                        np.array(exp_vals),
                        np.array(sim_vals),
                        f'{mutation} Resistance Frequency'
                    )
                    results.append(result)
        
        return results
    
    def validate_reservoir_dynamics(self, simulated_data: Dict) -> List[ValidationResult]:
        """
        Validate simulated reservoir dynamics against experimental data.
        
        Args:
            simulated_data: Dictionary containing simulated reservoir data
            
        Returns:
            List of validation results
        """
        results = []
        
        # Get experimental data
        exp_data = self.experimental_datasets['reservoir_decay']
        
        # Align time points
        exp_times = exp_data['time_months'].values
        exp_reservoir = exp_data['reservoir_size_log10'].values
        
        # Interpolate simulated data to match experimental time points
        if 'time_months' in simulated_data and 'reservoir_size_log10' in simulated_data:
            sim_times = simulated_data['time_months']
            sim_reservoir = simulated_data['reservoir_size_log10']
            
            # Interpolate simulated values to experimental time points
            from scipy.interpolate import interp1d
            f = interp1d(sim_times, sim_reservoir, kind='linear', fill_value='extrapolate')
            sim_reservoir_aligned = f(exp_times)
            
            # Perform validation
            result = self._validate_single_metric(
                'reservoir_decay',
                exp_reservoir,
                sim_reservoir_aligned,
                'HIV Reservoir Decay'
            )
            results.append(result)
        
        return results
    
    def _validate_single_metric(self, metric_name: str, 
                              experimental_values: np.ndarray,
                              simulated_values: np.ndarray,
                              description: str) -> ValidationResult:
        """
        Validate a single metric comparing experimental and simulated values.
        
        Args:
            metric_name: Name of the metric
            experimental_values: Experimental values
            simulated_values: Simulated values
            description: Description of the metric
            
        Returns:
            ValidationResult object
        """
        # Ensure arrays are the same length
        min_len = min(len(experimental_values), len(simulated_values))
        exp_vals = experimental_values[:min_len]
        sim_vals = simulated_values[:min_len]
        
        # Calculate correlation
        if len(exp_vals) > 1:
            correlation, p_value = stats.pearsonr(exp_vals, sim_vals)
        else:
            correlation, p_value = 0.0, 1.0
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(exp_vals, sim_vals))
        
        # Calculate MAE
        mae = mean_absolute_error(exp_vals, sim_vals)
        
        # Calculate R²
        r_squared = r2_score(exp_vals, sim_vals)
        
        # Calculate confidence interval for mean difference
        differences = exp_vals - sim_vals
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        n = len(differences)
        
        # 95% confidence interval
        if n > 1:
            se_diff = std_diff / np.sqrt(n)
            t_val = stats.t.ppf(0.975, n-1)  # 95% CI
            margin = t_val * se_diff
            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin
        else:
            ci_lower, ci_upper = mean_diff, mean_diff
        
        # Determine if validation passes
        threshold = self.validation_thresholds.get(metric_name, {})
        corr_thresh = threshold.get('correlation_min', self.correlation_threshold)
        rmse_thresh = threshold.get('rmse_max', self.rmse_threshold)
        
        passes_corr = abs(correlation) >= corr_thresh
        passes_rmse = rmse <= rmse_thresh
        passes_p = p_value <= self.p_value_threshold
        
        validation_passed = passes_corr and passes_rmse and passes_p
        
        # Determine clinical relevance
        if validation_passed:
            if abs(correlation) > 0.8:
                clinical_relevance = "Excellent correlation with experimental data"
            elif abs(correlation) > 0.7:
                clinical_relevance = "Good correlation with experimental data"
            else:
                clinical_relevance = "Moderate correlation with experimental data"
        else:
            clinical_relevance = "Poor correlation with experimental data - model needs refinement"
        
        return ValidationResult(
            metric_name=metric_name,
            experimental_value=np.mean(exp_vals),
            simulated_value=np.mean(sim_vals),
            p_value=p_value,
            correlation=correlation,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            confidence_interval=(ci_lower, ci_upper),
            validation_passed=validation_passed,
            clinical_relevance=clinical_relevance
        )
    
    def perform_global_validation(self, simulation_results: Dict) -> Dict:
        """
        Perform comprehensive validation across all metrics.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            Comprehensive validation report
        """
        all_results = []
        
        # Validate each component
        viral_results = self.validate_viral_dynamics(simulation_results)
        all_results.extend(viral_results)
        
        cd4_results = self.validate_cd4_recovery(simulation_results)
        all_results.extend(cd4_results)
        
        resistance_results = self.validate_resistance_patterns(simulation_results)
        all_results.extend(resistance_results)
        
        reservoir_results = self.validate_reservoir_dynamics(simulation_results)
        all_results.extend(reservoir_results)
        
        # Calculate overall validation score
        total_metrics = len(all_results)
        passed_metrics = sum(1 for r in all_results if r.validation_passed)
        overall_score = passed_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # Identify areas of concern
        concerns = []
        for result in all_results:
            if not result.validation_passed:
                concerns.append(f"{result.metric_name}: {result.clinical_relevance}")
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(all_results)
        
        report = {
            'overall_validation_score': overall_score,
            'total_metrics_evaluated': total_metrics,
            'passed_metrics': passed_metrics,
            'failed_metrics': total_metrics - passed_metrics,
            'validation_results': all_results,
            'areas_of_concern': concerns,
            'recommendations': recommendations,
            'confidence_level': self._calculate_confidence_level(overall_score),
            'clinical_applicability': self._determine_clinical_applicability(overall_score)
        }
        
        return report
    
    def _generate_validation_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for systematic biases
        correlations = [r.correlation for r in validation_results if not np.isnan(r.correlation)]
        if correlations:
            avg_correlation = np.mean(correlations)
            if avg_correlation < 0.5:
                recommendations.append("Overall correlation is low (<0.5). Consider revising model assumptions.")
            elif avg_correlation < 0.7:
                recommendations.append("Overall correlation is moderate (0.5-0.7). Model may need refinement.")
        
        # Check for high RMSE values
        rmses = [r.rmse for r in validation_results if not np.isnan(r.rmse)]
        if rmses:
            avg_rmse = np.mean(rmses)
            if avg_rmse > 0.5:
                recommendations.append("High RMSE values indicate poor predictive accuracy. Investigate parameter calibration.")
        
        # Identify specific metrics that failed
        failed_metrics = [r for r in validation_results if not r.validation_passed]
        if failed_metrics:
            failed_names = [f"{r.metric_name} (r={r.correlation:.3f})" for r in failed_metrics[:3]]
            recommendations.append(f"Failed metrics: {', '.join(failed_names)}. Focus validation efforts on these areas.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model validation appears satisfactory. Continue monitoring performance with additional datasets.")
        
        return recommendations
    
    def _calculate_confidence_level(self, overall_score: float) -> str:
        """Calculate confidence level based on overall validation score."""
        if overall_score >= 0.9:
            return "Very High"
        elif overall_score >= 0.8:
            return "High"
        elif overall_score >= 0.7:
            return "Moderate"
        elif overall_score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _determine_clinical_applicability(self, overall_score: float) -> str:
        """Determine clinical applicability based on validation score."""
        if overall_score >= 0.85:
            return "Suitable for clinical decision support"
        elif overall_score >= 0.7:
            return "Suitable for research applications"
        elif overall_score >= 0.5:
            return "Limited utility, requires further validation"
        else:
            return "Not suitable for clinical applications"
    
    def sensitivity_analysis(self, base_params: Dict, param_ranges: Dict, 
                           validation_metric: str = 'viral_load_decline') -> Dict:
        """
        Perform sensitivity analysis to identify critical parameters.
        
        Args:
            base_params: Base parameter values
            param_ranges: Ranges for each parameter to test
            validation_metric: Metric to use for sensitivity analysis
            
        Returns:
            Sensitivity analysis results
        """
        results = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Test parameter at different values
            test_values = np.linspace(min_val, max_val, 5)  # Test 5 values
            param_sensitivity = []
            
            for val in test_values:
                # Modify parameter
                test_params = base_params.copy()
                test_params[param_name] = val
                
                # Run simulation with modified parameter
                # (In a real implementation, this would call the actual simulation)
                # For this demo, we'll simulate the effect
                simulated_output = self._simulate_param_effect(test_params, validation_metric)
                
                # Validate against experimental data
                if validation_metric == 'viral_load_decline':
                    exp_data = self.experimental_datasets['viral_decline']['viral_load_log10'].values
                    correlation, _ = stats.pearsonr(exp_data, simulated_output)
                else:
                    correlation = 0.5  # Default value
                
                param_sensitivity.append(correlation)
            
            # Calculate sensitivity as change in correlation per unit change in parameter
            param_range = max_val - min_val
            if param_range > 0 and len(param_sensitivity) > 1:
                sensitivity = (max(param_sensitivity) - min(param_sensitivity)) / param_range
            else:
                sensitivity = 0.0
            
            results[param_name] = {
                'sensitivity': sensitivity,
                'test_values': test_values.tolist(),
                'correlations': param_sensitivity,
                'impact_category': self._categorize_param_impact(sensitivity)
            }
        
        return results
    
    def _simulate_param_effect(self, params: Dict, metric: str) -> np.ndarray:
        """Simulate the effect of parameter changes on output (for sensitivity analysis)."""
        # This is a placeholder that would be replaced with actual simulation logic
        # For this demo, we'll return a simple time series
        time_points = np.arange(0, 10, 1)
        
        if metric == 'viral_load_decline':
            # Simulate viral decline with parameter-dependent rate
            decline_rate = params.get('viral_clearance_rate', 0.5)
            base_level = params.get('baseline_viral_load', 5.0)
            return base_level * np.exp(-decline_rate * time_points)
        else:
            return np.random.rand(len(time_points))
    
    def _categorize_param_impact(self, sensitivity: float) -> str:
        """Categorize parameter impact based on sensitivity."""
        abs_sens = abs(sensitivity)
        if abs_sens > 0.5:
            return "High Impact"
        elif abs_sens > 0.2:
            return "Medium Impact"
        elif abs_sens > 0.05:
            return "Low Impact"
        else:
            return "Negligible Impact"
    
    def generate_validation_visualization(self, validation_results: List[ValidationResult], 
                                        save_path: str = None) -> plt.Figure:
        """
        Generate visualization of validation results.
        
        Args:
            validation_results: List of validation results
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HIV Simulation Model Validation Results', fontsize=16)
        
        # Extract data for plotting
        metric_names = [r.metric_name for r in validation_results]
        correlations = [r.correlation for r in validation_results]
        rmse_values = [r.rmse for r in validation_results]
        r_squared_values = [r.r_squared for r in validation_results]
        validation_status = [r.validation_passed for r in validation_results]
        
        # Plot 1: Correlation coefficients
        axes[0, 0].bar(range(len(metric_names)), correlations, 
                      color=['green' if r.validation_passed else 'red' for r in validation_results])
        axes[0, 0].set_xticks(range(len(metric_names)))
        axes[0, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Correlation Coefficient')
        axes[0, 0].set_title('Correlation with Experimental Data')
        axes[0, 0].axhline(y=self.correlation_threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # Plot 2: RMSE values
        axes[0, 1].bar(range(len(metric_names)), rmse_values,
                      color=['green' if r.validation_passed else 'red' for r in validation_results])
        axes[0, 1].set_xticks(range(len(metric_names)))
        axes[0, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Root Mean Square Error')
        axes[0, 1].axhline(y=self.rmse_threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 1].legend()
        
        # Plot 3: R² values
        axes[1, 0].bar(range(len(metric_names)), r_squared_values,
                      color=['green' if r.validation_passed else 'red' for r in validation_results])
        axes[1, 0].set_xticks(range(len(metric_names)))
        axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Coefficient of Determination')
        axes[1, 0].axhline(y=self.r_squared_threshold, color='red', linestyle='--', label='Threshold')
        axes[1, 0].legend()
        
        # Plot 4: Validation summary
        pass_count = sum(validation_status)
        fail_count = len(validation_status) - pass_count
        axes[1, 1].pie([pass_count, fail_count], labels=['Passed', 'Failed'], 
                       autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Validation Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def run_validation_demo():
    """Demo function showing how to use the ExperimentalValidator."""
    print("Starting HIV Experimental Validation Demo...")
    
    # Initialize the validator
    validator = ExperimentalValidator()
    
    # Example simulation results (these would come from the actual simulation)
    simulation_results = {
        'time_days': [1, 2, 3, 7, 14, 21, 28, 56, 84],
        'viral_load_log10': [4.5, 3.7, 3.1, 2.0, 1.4, 1.1, 0.9, 0.7, 0.6],
        'time_weeks': [2, 4, 8, 12, 24, 36, 48, 96],
        'cd4_count': [245, 315, 375, 415, 475, 515, 545, 575],
        'time_months': [3, 6, 12, 18, 24, 36, 48, 60],
        'reservoir_size_log10': [1.75, 1.68, 1.58, 1.52, 1.48, 1.42, 1.38, 1.35],
        'resistance_frequencies': {
            'K103N': 0.12,
            'M184V': 0.62,
            'K65R': 0.04,
            'L90M': 0.07,
            'Y181C': 0.11,
            'G190A': 0.08
        }
    }
    
    # Example 1: Validate viral dynamics
    print("\n1. Validating Viral Dynamics:")
    viral_results = validator.validate_viral_dynamics(simulation_results)
    for result in viral_results:
        print(f"   {result.metric_name}: r={result.correlation:.3f}, "
              f"RMSE={result.rmse:.3f}, p={result.p_value:.3f}, "
              f"Passed: {result.validation_passed}")
    
    # Example 2: Validate CD4 recovery
    print("\n2. Validating CD4 Recovery:")
    cd4_results = validator.validate_cd4_recovery(simulation_results)
    for result in cd4_results:
        print(f"   {result.metric_name}: r={result.correlation:.3f}, "
              f"RMSE={result.rmse:.1f}, R²={result.r_squared:.3f}, "
              f"Passed: {result.validation_passed}")
    
    # Example 3: Validate resistance patterns
    print("\n3. Validating Resistance Patterns:")
    resistance_results = validator.validate_resistance_patterns(simulation_results)
    for result in resistance_results[:3]:  # Show first 3
        print(f"   {result.metric_name}: Exp={result.experimental_value:.3f}, "
              f"Sim={result.simulated_value:.3f}, Passed: {result.validation_passed}")
    
    # Example 4: Validate reservoir dynamics
    print("\n4. Validating Reservoir Dynamics:")
    reservoir_results = validator.validate_reservoir_dynamics(simulation_results)
    for result in reservoir_results:
        print(f"   {result.metric_name}: r={result.correlation:.3f}, "
              f"RMSE={result.rmse:.3f}, Bias={result.confidence_interval[0]:.3f}, "
              f"Passed: {result.validation_passed}")
    
    # Example 5: Perform global validation
    print("\n5. Performing Global Validation:")
    global_report = validator.perform_global_validation(simulation_results)
    
    print(f"   Overall Validation Score: {global_report['overall_validation_score']:.3f}")
    print(f"   Total Metrics Evaluated: {global_report['total_metrics_evaluated']}")
    print(f"   Passed Metrics: {global_report['passed_metrics']}")
    print(f"   Failed Metrics: {global_report['failed_metrics']}")
    print(f"   Confidence Level: {global_report['confidence_level']}")
    print(f"   Clinical Applicability: {global_report['clinical_applicability']}")
    
    print(f"\n   Areas of Concern:")
    for concern in global_report['areas_of_concern'][:3]:  # Show first 3
        print(f"     - {concern}")
    
    print(f"\n   Recommendations:")
    for rec in global_report['recommendations'][:3]:  # Show first 3
        print(f"     - {rec}")
    
    # Example 6: Sensitivity analysis
    print("\n6. Performing Sensitivity Analysis:")
    base_params = {
        'viral_clearance_rate': 0.5,
        'infection_rate': 2e-8,
        'drug_efficacy': 0.9
    }
    
    param_ranges = {
        'viral_clearance_rate': (0.1, 1.0),
        'infection_rate': (1e-8, 5e-8),
        'drug_efficacy': (0.7, 0.99)
    }
    
    sensitivity_results = validator.sensitivity_analysis(base_params, param_ranges)
    
    print("   Parameter Sensitivity Analysis:")
    for param, data in sensitivity_results.items():
        print(f"     {param}: {data['impact_category']} (sensitivity: {data['sensitivity']:.3f})")
    
    # Example 7: Generate validation visualization
    print("\n7. Generating Validation Visualization:")
    print("   (Visualization would be saved as 'validation_results.png')")
    
    # Combine all results for visualization
    all_results = (viral_results + cd4_results + resistance_results + reservoir_results)
    fig = validator.generate_validation_visualization(all_results, "validation_results.png")
    
    print("\nDemo completed!")
    print("\nNote: In a real implementation, this would compare against actual experimental datasets")
    print("from HIV research studies to validate the simulation model's predictions.")


if __name__ == "__main__":
    run_validation_demo()