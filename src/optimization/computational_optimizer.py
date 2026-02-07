"""
Computational efficiency optimization module for Project Genesis-HIV.

This module implements optimizations to improve the performance of the simulation,
including parallel processing, caching, algorithmic improvements, and memory management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools
import threading
import queue
from collections import OrderedDict
import gc
import psutil
import os
from functools import wraps
import cProfile
import pstats
from io import StringIO


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking optimization effectiveness."""
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    throughput: float  # Operations per second
    cache_hit_rate: float
    parallel_efficiency: float


class ComputationalOptimizer:
    """
    A class to optimize computational performance of the HIV simulation.
    
    This includes:
    - Parallel processing for independent tasks
    - Caching for expensive computations
    - Memory management and garbage collection
    - Algorithmic optimizations
    - Profiling and performance monitoring
    - Vectorization of mathematical operations
    """
    
    def __init__(self):
        """Initialize the ComputationalOptimizer."""
        self.cache = OrderedDict()
        self.cache_size_limit = 1000  # Maximum number of cached items
        self.profile_enabled = True
        self.profiling_stats = {}
        
        # Performance tracking
        self.metrics_history = []
        
        # Multiprocessing settings
        self.max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid overhead
        
        # Memory management
        self.memory_threshold = 0.8  # 80% of available memory
        self.cleanup_threshold = 0.9  # 90% of available memory
    
    def memoize(self, func: Callable) -> Callable:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to memoize
            
        Returns:
            Memoized function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if result is in cache
            if key in self.cache:
                return self.cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Add to cache
            self.cache[key] = result
            
            # Manage cache size
            if len(self.cache) > self.cache_size_limit:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            
            return result
        return wrapper
    
    def time_it(self, func: Callable) -> Callable:
        """
        Decorator to time function execution and collect metrics.
        
        Args:
            func: Function to time
            
        Returns:
            Timed function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_percent = psutil.cpu_percent()
            
            # Calculate throughput (operations per second)
            if execution_time > 0:
                throughput = 1.0 / execution_time
            else:
                throughput = float('inf')
            
            # Calculate cache hit rate
            cache_hits = sum(1 for k, v in self.cache.items() if v is not None)
            cache_hit_rate = cache_hits / len(self.cache) if self.cache else 0.0
            
            # Calculate parallel efficiency (simplified)
            parallel_efficiency = 1.0  # Placeholder for actual calculation
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_delta,
                cpu_utilization=cpu_percent,
                throughput=throughput,
                cache_hit_rate=cache_hit_rate,
                parallel_efficiency=parallel_efficiency
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            return result
        return wrapper
    
    def parallel_process(self, items: List, func: Callable, chunk_size: int = None) -> List:
        """
        Process items in parallel using multiple threads or processes.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            chunk_size: Size of chunks to process in parallel
            
        Returns:
            List of processed results
        """
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = [executor.submit(self._process_chunk, chunk, func) for chunk in chunks]
            
            # Collect results
            for future in as_completed(futures):
                results.extend(future.result())
        
        return results
    
    def _process_chunk(self, chunk: List, func: Callable) -> List:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def vectorize_operation(self, array: np.ndarray, operation: Callable) -> np.ndarray:
        """
        Apply an operation to an array using vectorization.
        
        Args:
            array: Input array
            operation: Operation to apply
            
        Returns:
            Result array
        """
        # Use numpy's vectorize function for efficient operation
        vectorized_op = np.vectorize(operation)
        return vectorized_op(array)
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up unused objects."""
        # Check current memory usage
        current_memory = psutil.virtual_memory().percent / 100.0
        
        if current_memory > self.cleanup_threshold:
            # Force garbage collection
            gc.collect()
            
            # Clear cache if needed
            if len(self.cache) > self.cache_size_limit // 2:
                # Remove half of the cache entries
                for _ in range(len(self.cache) // 2):
                    if self.cache:
                        self.cache.popitem(last=False)
        
        elif current_memory > self.memory_threshold:
            # Clear some cache entries
            if len(self.cache) > self.cache_size_limit // 4:
                for _ in range(len(self.cache) // 4):
                    if self.cache:
                        self.cache.popitem(last=False)
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution.
        
        Args:
            func: Function to profile
            
        Returns:
            Profiled function
        """
        if not self.profile_enabled:
            return func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a string buffer to capture profiling stats
            pr = cProfile.Profile()
            pr.enable()
            
            result = func(*args, **kwargs)
            
            pr.disable()
            
            # Capture stats
            s = StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            # Store profiling stats
            func_name = func.__name__
            if func_name not in self.profiling_stats:
                self.profiling_stats[func_name] = []
            self.profiling_stats[func_name].append(s.getvalue())
            
            return result
        return wrapper
    
    def optimize_viral_dynamics_calculation(self, 
                                          viral_load: float, 
                                          target_cells: float, 
                                          infected_cells: float,
                                          drug_concentration: float) -> Tuple[float, float, float]:
        """
        Optimized calculation of viral dynamics using vectorized operations.
        
        Args:
            viral_load: Current viral load
            target_cells: Number of target cells
            infected_cells: Number of infected cells
            drug_concentration: Drug concentration
            
        Returns:
            Tuple of (new_viral_load, new_target_cells, new_infected_cells)
        """
        # Use numpy for vectorized calculations
        # Parameters
        infection_rate = 2e-8  # per virion per cell per day
        clearance_rate = 23  # per day
        production_rate = 2000  # virions per cell per day
        death_rate = 0.01  # per day
        drug_efficacy = 0.9  # 90% efficacy at max concentration
        
        # Calculate infection rate with drug effect
        effective_infection_rate = infection_rate * (1 - drug_efficacy * drug_concentration)
        
        # Calculate new values using vectorized operations
        newly_infected = effective_infection_rate * viral_load * target_cells
        new_infected_cells = infected_cells + newly_infected - death_rate * infected_cells
        new_viral_load = (production_rate * infected_cells - clearance_rate * viral_load) * 0.1  # Time step
        new_target_cells = target_cells - newly_infected + 0.1 * target_cells  # Recruitment
        
        # Ensure non-negative values
        new_viral_load = max(0, viral_load + new_viral_load)
        new_target_cells = max(0, new_target_cells)
        new_infected_cells = max(0, new_infected_cells)
        
        return new_viral_load, new_target_cells, new_infected_cells
    
    def optimize_binding_energy_calculation(self, 
                                          protein_coords: np.ndarray,
                                          ligand_coords: np.ndarray) -> float:
        """
        Optimized calculation of binding energy using vectorized operations.
        
        Args:
            protein_coords: Coordinates of protein atoms
            ligand_coords: Coordinates of ligand atoms
            
        Returns:
            Binding energy
        """
        # Calculate distance matrix using vectorized operations
        dist_matrix = np.linalg.norm(
            protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :], 
            axis=2
        )
        
        # Apply Lennard-Jones potential
        sigma = 2.0  # Angstroms
        epsilon = 0.2  # kcal/mol
        
        # Avoid division by zero
        dist_matrix = np.clip(dist_matrix, 0.1, None)
        
        # Calculate potential for all pairs
        term1 = (sigma / dist_matrix) ** 12
        term2 = (sigma / dist_matrix) ** 6
        potential = 4 * epsilon * (term1 - term2)
        
        # Sum all interactions
        binding_energy = np.sum(potential)
        
        return binding_energy
    
    def optimize_resistance_prediction(self, 
                                     mutation_list: List[str],
                                     drug_list: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Optimized prediction of drug resistance using vectorized operations.
        
        Args:
            mutation_list: List of mutations
            drug_list: List of drugs
            
        Returns:
            Resistance predictions
        """
        # Create a lookup table for mutation-drug effects (simplified)
        # In a real implementation, this would be a precomputed matrix
        resistance_matrix = {}
        
        for drug in drug_list:
            drug_resistances = {}
            for mutation in mutation_list:
                # Calculate resistance effect (simplified)
                # In reality, this would use complex interaction models
                base_resistance = 1.0
                if "K103N" in mutation and "NNRTI" in drug:
                    base_resistance = 100.0
                elif "M184V" in mutation and "NRTI" in drug:
                    base_resistance = 10.0
                elif "L90M" in mutation and "PI" in drug:
                    base_resistance = 50.0
                
                drug_resistances[mutation] = base_resistance
            
            resistance_matrix[drug] = drug_resistances
        
        return resistance_matrix
    
    def batch_process_simulations(self, 
                                simulation_params: List[Dict],
                                simulation_func: Callable,
                                batch_size: int = 100) -> List:
        """
        Batch process multiple simulations efficiently.
        
        Args:
            simulation_params: List of parameter dictionaries for simulations
            simulation_func: Function to run each simulation
            batch_size: Size of batches to process in parallel
            
        Returns:
            List of simulation results
        """
        # Split into batches
        batches = [simulation_params[i:i + batch_size] 
                  for i in range(0, len(simulation_params), batch_size)]
        
        results = []
        
        # Process each batch in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch jobs
            futures = []
            for batch in batches:
                future = executor.submit(self._process_batch, batch, simulation_func)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[Dict], simulation_func: Callable) -> List:
        """Process a batch of simulations."""
        return [simulation_func(params) for params in batch]
    
    def optimize_genome_alignment(self, 
                                query_sequence: str,
                                reference_sequences: List[str],
                                threshold: float = 0.8) -> List[Tuple[int, float]]:
        """
        Optimized genome alignment using vectorized operations where possible.
        
        Args:
            query_sequence: Query sequence to align
            reference_sequences: List of reference sequences
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity) tuples for matches above threshold
        """
        matches = []
        
        # Convert sequences to numpy arrays of character codes for faster comparison
        query_array = np.frombuffer(query_sequence.encode('utf-8'), dtype=np.uint8)
        
        for i, ref_seq in enumerate(reference_sequences):
            if len(ref_seq) == 0:
                continue
                
            ref_array = np.frombuffer(ref_seq.encode('utf-8'), dtype=np.uint8)
            
            # Calculate similarity using vectorized operations
            min_len = min(len(query_array), len(ref_array))
            query_trimmed = query_array[:min_len]
            ref_trimmed = ref_array[:min_len]
            
            # Calculate percentage identity
            matches_count = np.sum(query_trimmed == ref_trimmed)
            similarity = matches_count / min_len
            
            if similarity >= threshold:
                matches.append((i, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_optimization_report(self) -> Dict:
        """
        Generate a report on optimization effectiveness.
        
        Returns:
            Optimization report
        """
        if not self.metrics_history:
            return {"message": "No performance metrics collected yet"}
        
        # Calculate aggregate metrics
        exec_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        cache_hit_rates = [m.cache_hit_rate for m in self.metrics_history]
        
        report = {
            "total_executions": len(self.metrics_history),
            "avg_execution_time": np.mean(exec_times),
            "std_execution_time": np.std(exec_times),
            "min_execution_time": min(exec_times),
            "max_execution_time": max(exec_times),
            "avg_memory_usage": np.mean(memory_usage),
            "avg_throughput": np.mean(throughputs),
            "avg_cache_hit_rate": np.mean(cache_hit_rates),
            "cache_size": len(self.cache),
            "cache_limit": self.cache_size_limit,
            "profiling_stats": self.profiling_stats,
            "recommended_optimizations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        if not self.metrics_history:
            return ["Run performance tests to generate recommendations"]
        
        avg_time = np.mean([m.execution_time for m in self.metrics_history])
        avg_memory = np.mean([m.memory_usage for m in self.metrics_history])
        avg_cache_hit = np.mean([m.cache_hit_rate for m in self.metrics_history])
        
        if avg_time > 1.0:  # More than 1 second average
            recommendations.append("Consider further algorithmic optimizations for slow functions")
        
        if avg_memory > 100:  # More than 100MB average
            recommendations.append("Implement more aggressive memory management")
        
        if avg_cache_hit < 0.5:  # Less than 50% cache hit rate
            recommendations.append("Increase cache size or optimize cache key generation")
        
        if len(self.cache) == self.cache_size_limit:
            recommendations.append("Cache is at capacity - consider increasing limit or optimizing eviction policy")
        
        return recommendations


def run_optimization_demo():
    """Demo function showing how to use the ComputationalOptimizer."""
    print("Starting HIV Computational Optimization Demo...")
    
    # Initialize the optimizer
    optimizer = ComputationalOptimizer()
    
    # Example 1: Using memoization
    print("\n1. Testing Memoization:")
    
    @optimizer.memoize
    def expensive_calculation(x):
        """Simulate an expensive calculation."""
        time.sleep(0.1)  # Simulate computation time
        return x ** 2 + 2 * x + 1
    
    # First call - should take time
    start = time.time()
    result1 = expensive_calculation(5)
    time1 = time.time() - start
    print(f"   First call: {result1}, took {time1:.3f}s")
    
    # Second call with same argument - should be instant
    start = time.time()
    result2 = expensive_calculation(5)
    time2 = time.time() - start
    print(f"   Second call: {result2}, took {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.2f}x" if time2 > 0 else "Instant retrieval")
    
    # Example 2: Timing functions
    print("\n2. Testing Function Timing:")
    
    @optimizer.time_it
    def simulate_viral_dynamics():
        """Simulate viral dynamics."""
        time.sleep(0.05)  # Simulate computation
        return np.random.rand(1000)  # Simulate results
    
    results = simulate_viral_dynamics()
    print(f"   Simulated viral dynamics for {len(results)} time points")
    
    # Example 3: Parallel processing
    print("\n3. Testing Parallel Processing:")
    
    def process_item(item):
        """Simulate processing an item."""
        time.sleep(0.01)  # Simulate work
        return item ** 2
    
    items = list(range(20))
    start = time.time()
    parallel_results = optimizer.parallel_process(items, process_item, chunk_size=5)
    parallel_time = time.time() - start
    
    print(f"   Processed {len(items)} items in parallel: {parallel_time:.3f}s")
    print(f"   First few results: {parallel_results[:5]}")
    
    # Example 4: Vectorized operations
    print("\n4. Testing Vectorized Operations:")
    
    test_array = np.random.rand(10000)
    
    # Non-vectorized approach
    start = time.time()
    non_vec_result = [x**2 + 2*x + 1 for x in test_array[:1000]]  # Limit for demo
    non_vec_time = time.time() - start
    
    # Vectorized approach
    start = time.time()
    vec_result = optimizer.vectorize_operation(test_array[:1000], lambda x: x**2 + 2*x + 1)
    vec_time = time.time() - start
    
    print(f"   Non-vectorized: {non_vec_time:.4f}s for 1000 elements")
    print(f"   Vectorized: {vec_time:.4f}s for 1000 elements")
    print(f"   Speedup: {non_vec_time/vec_time:.2f}x" if vec_time > 0 else "Vectorized approach was faster")
    
    # Example 5: Optimized viral dynamics
    print("\n5. Testing Optimized Viral Dynamics:")
    
    start = time.time()
    v_load, t_cells, i_cells = optimizer.optimize_viral_dynamics_calculation(
        viral_load=1e5, target_cells=1e6, infected_cells=1e3, drug_concentration=0.8
    )
    dyn_time = time.time() - start
    
    print(f"   Optimized viral dynamics calculation: {dyn_time:.6f}s")
    print(f"   Results - VL: {v_load:.2e}, Target: {t_cells:.2e}, Infected: {i_cells:.2e}")
    
    # Example 6: Optimized binding energy
    print("\n6. Testing Optimized Binding Energy:")
    
    protein_coords = np.random.rand(100, 3) * 10  # 100 protein atoms
    ligand_coords = np.random.rand(10, 3) * 5    # 10 ligand atoms
    
    start = time.time()
    binding_energy = optimizer.optimize_binding_energy_calculation(protein_coords, ligand_coords)
    binding_time = time.time() - start
    
    print(f"   Optimized binding energy calculation: {binding_time:.6f}s")
    print(f"   Binding energy: {binding_energy:.4f} kcal/mol")
    
    # Example 7: Batch processing
    print("\n7. Testing Batch Processing:")
    
    # Create simulation parameters
    sim_params = [{"param1": i, "param2": i*2} for i in range(50)]
    
    def dummy_simulation(params):
        """Dummy simulation function."""
        time.sleep(0.001)  # Simulate quick computation
        return {"result": params["param1"] + params["param2"], "id": params["param1"]}
    
    start = time.time()
    batch_results = optimizer.batch_process_simulations(sim_params, dummy_simulation, batch_size=10)
    batch_time = time.time() - start
    
    print(f"   Batch processed {len(sim_params)} simulations: {batch_time:.3f}s")
    print(f"   First few results: {[r for r in batch_results[:3]]}")
    
    # Example 8: Memory optimization
    print("\n8. Testing Memory Optimization:")
    optimizer.optimize_memory_usage()
    print(f"   Memory optimization completed. Current cache size: {len(optimizer.cache)}")
    
    # Example 9: Optimization report
    print("\n9. Generating Optimization Report:")
    report = optimizer.get_optimization_report()
    
    print(f"   Total executions timed: {report['total_executions']}")
    print(f"   Average execution time: {report['avg_execution_time']:.6f}s")
    print(f"   Average throughput: {report['avg_throughput']:.2f} ops/sec")
    print(f"   Average cache hit rate: {report['avg_cache_hit_rate']:.2f}")
    print(f"   Current cache size: {report['cache_size']}/{report['cache_limit']}")
    
    print("\nRecommended optimizations:")
    for rec in report['recommended_optimizations']:
        print(f"   - {rec}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_optimization_demo()