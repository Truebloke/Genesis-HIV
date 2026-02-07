"""
Macro-view module for Project Genesis-HIV: T-cell and viral load dynamics.

This module provides real-time visualization of T-cell dynamics and viral load changes.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Parameters for the viral dynamics simulation."""

    baseline_viral_load: float = 1e5  # copies/mL
    baseline_cd4: float = 500  # cells/μL
    infection_rate: float = 2e-8  # per virion per cell per day
    viral_clearance_rate: float = 23  # per day
    infected_cell_death_rate: float = 0.5  # per day
    cd4_replenishment_rate: float = 0.1  # per day
    treatment_efficacy: float = 0.9  # 0-1 scale
    resistance_development: float = 0.01  # per day
    cd4_depletion_per_infected_cell: float = (
        0.001  # CD4 cells depleted per infected cell per day
    )


class MacroViewEngine:
    """
    Engine for simulating and visualizing macro-level dynamics:
    - T-cell counts (CD4+ and CD8+)
    - Viral load dynamics
    - Treatment effects
    - Immune response
    """

    def __init__(self):
        """Initialize the MacroViewEngine."""
        self.time_points = [0]
        self.viral_load_history = [1e5]
        self.cd4_count_history = [500]
        self.cd8_count_history = [50]
        self.treatment_history = [0.0]  # Treatment efficacy over time
        self.resistance_history = [0.0]  # Resistance level over time

        # Simulation parameters
        self.params = SimulationParams()

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.time_points = [0]
        self.viral_load_history = [self.params.baseline_viral_load]
        self.cd4_count_history = [self.params.baseline_cd4]
        self.cd8_count_history = [50]
        self.treatment_history = [0.0]
        self.resistance_history = [0.0]

    def update_dynamics(self, time_step: float = 1.0):
        """
        Update the viral and immune dynamics for one time step.

        Args:
            time_step: Time step for the update (in days)
        """
        # Get current values
        current_time = self.time_points[-1] + time_step
        current_vl = self.viral_load_history[-1]
        current_cd4 = self.cd4_count_history[-1]
        current_cd8 = self.cd8_count_history[-1]
        current_treatment = (
            self.treatment_history[-1] if self.treatment_history else 0.0
        )
        current_resistance = (
            self.resistance_history[-1] if self.resistance_history else 0.0
        )

        # Calculate treatment effectiveness (reduced by resistance)
        effective_treatment = current_treatment * (1 - current_resistance)

        # Calculate number of infected cells (approximation)
        # Using a simplified model where infected cells are proportional to viral load
        infection_rate = self.params.infection_rate
        infected_cell_death_rate = self.params.infected_cell_death_rate
        current_infected_cells = (infection_rate * current_vl * current_cd4) / (
            infected_cell_death_rate + infection_rate * current_cd4
        )

        # Calculate viral replication rate (with treatment effect)
        base_replication = 20.0  # virions per infected cell per day (more realistic)
        net_replication = base_replication * (1 - effective_treatment)

        # Update viral load
        viral_production = net_replication * current_infected_cells
        viral_clearance = self.params.viral_clearance_rate * current_vl
        viral_change = (viral_production - viral_clearance) * time_step
        new_vl = max(50, current_vl + viral_change)  # Minimum detectable level

        # Update CD4 count
        cd4_depletion_rate = min(
            0.1, 0.01 * (new_vl / 1e5)
        )  # Higher VL causes faster depletion
        cd4_replenishment_rate = self.params.cd4_replenishment_rate
        cd4_change = (
            (cd4_replenishment_rate - cd4_depletion_rate) * current_cd4 * time_step
        )
        new_cd4 = max(10, current_cd4 + cd4_change)

        # Update CD8 count (reflects immune response to virus)
        cd8_activation_rate = min(
            0.05, 0.005 * (new_vl / 1e5)
        )  # Activation proportional to VL
        cd8_decay_rate = 0.01  # Natural decay rate
        cd8_change = (cd8_activation_rate - cd8_decay_rate) * current_cd8 * time_step
        new_cd8 = max(20, current_cd8 + cd8_change)

        # Update resistance (if under treatment)
        if effective_treatment > 0.1:  # Only if treatment is active
            resistance_increase = self.params.resistance_development * time_step
            new_resistance = min(1.0, current_resistance + resistance_increase)
        else:
            new_resistance = current_resistance

        # Update treatment (for demonstration, assuming constant treatment)
        new_treatment = current_treatment

        # Append to histories
        self.time_points.append(current_time)
        self.viral_load_history.append(new_vl)
        self.cd4_count_history.append(new_cd4)
        self.cd8_count_history.append(new_cd8)
        self.treatment_history.append(new_treatment)
        self.resistance_history.append(new_resistance)

    def run_simulation(self, duration_days: int = 365, time_step: float = 1.0):
        """
        Run the simulation for a specified duration.

        Args:
            duration_days: Duration of simulation in days
            time_step: Time step for updates in days
        """
        self.reset_simulation()

        steps = int(duration_days / time_step)
        for i in range(steps):
            self.update_dynamics(time_step)

    def create_dynamic_plot(self) -> go.Figure:
        """
        Create a dynamic plot of viral load and T-cell counts over time.

        Returns:
            Plotly figure with the dynamics plot
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Viral Load (log₁₀ copies/mL)", "T-cell Counts (cells/μL)"),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        )

        # Add viral load (log scale)
        fig.add_trace(
            go.Scatter(
                x=self.time_points,
                y=np.log10(self.viral_load_history),
                mode="lines",
                name="Viral Load",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        # Add CD4 count
        fig.add_trace(
            go.Scatter(
                x=self.time_points,
                y=self.cd4_count_history,
                mode="lines",
                name="CD4+ T cells",
                line=dict(color="blue", width=2),
            ),
            row=2,
            col=1,
        )

        # Add CD8 count
        fig.add_trace(
            go.Scatter(
                x=self.time_points,
                y=self.cd8_count_history,
                mode="lines",
                name="CD8+ T cells",
                line=dict(color="green", width=2),
            ),
            row=2,
            col=1,
        )

        # Add treatment periods if applicable
        if any(self.treatment_history):
            treatment_on = [i for i, t in enumerate(self.treatment_history) if t > 0.1]
            if treatment_on:
                fig.add_vrect(
                    x0=self.time_points[min(treatment_on)],
                    x1=self.time_points[max(treatment_on)],
                    fillcolor="yellow",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="Treatment Period",
                    annotation_position="top left",
                    row="all",
                    col=1,
                )

        # Update layout
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_yaxes(title_text="Log₁₀ Viral Load", row=1, col=1)
        fig.update_yaxes(title_text="CD4+ T cells (cells/μL)", row=2, col=1)
        fig.update_yaxes(
            title_text="CD8+ T cells (cells/μL)", secondary_y=True, row=2, col=1
        )

        fig.update_layout(
            height=700,
            title_text="HIV Dynamics: Viral Load and T-cell Counts Over Time",
            showlegend=True,
        )

        return fig

    def create_real_time_simulation(
        self, duration_days: int = 180, update_interval: float = 0.1
    ):
        """
        Create a real-time simulation that updates the plot as it runs.

        Args:
            duration_days: Duration of simulation in days
            update_interval: Time between updates in seconds
        """
        # Create placeholders for the chart and metrics
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Reset simulation
        self.reset_simulation()

        # Run simulation step by step
        time_step = 1.0  # day
        steps = int(duration_days / time_step)

        progress_bar = st.progress(0)

        for i in range(steps):
            # Update dynamics
            self.update_dynamics(time_step)

            # Create current chart
            current_fig = self.create_dynamic_plot()

            # Update chart
            chart_placeholder.plotly_chart(current_fig, width="stretch")

            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current VL", f"{self.viral_load_history[-1]:.2e}")
                with col2:
                    st.metric("Current CD4", f"{self.cd4_count_history[-1]:.0f}")
                with col3:
                    st.metric("Current CD8", f"{self.cd8_count_history[-1]:.0f}")
                with col4:
                    st.metric(
                        "Resistance Level", f"{self.resistance_history[-1]*100:.1f}%"
                    )

            # Update progress
            progress_bar.progress(min(1.0, i / steps))

            # Small delay to allow visualization
            time.sleep(update_interval)

        progress_bar.empty()
        st.success("Simulation completed!")


def show_macro_view():
    """Display the macro-view component in the dashboard."""
    st.title("📊 Macro-View: T-cell and Viral Load Dynamics")
    st.markdown("---")

    st.markdown("""
    This section provides real-time visualization of HIV dynamics at the population level:
    - **Viral Load**: Number of viral particles in blood over time
    - **CD4+ T cells**: Helper T cells that coordinate immune response
    - **CD8+ T cells**: Cytotoxic T cells that kill infected cells
    """)

    # Initialize engine if not already done
    if "macro_engine" not in st.session_state:
        st.session_state.macro_engine = MacroViewEngine()

    # Control panel
    st.subheader("Simulation Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        baseline_vl = st.number_input(
            "Baseline Viral Load",
            value=100000,
            format="%d",
            help="Initial viral load in copies/mL",
        )
        baseline_cd4 = st.number_input(
            "Baseline CD4 Count",
            value=500,
            format="%d",
            help="Initial CD4+ T cell count in cells/μL",
        )

    with col2:
        treatment_efficacy = st.slider(
            "Treatment Efficacy",
            0.0,
            1.0,
            0.9,
            help="Effectiveness of antiretroviral therapy",
        )
        resistance_rate = st.slider(
            "Resistance Development Rate",
            0.0,
            0.1,
            0.01,
            help="Daily rate of resistance development",
        )

    with col3:
        duration = st.slider(
            "Simulation Duration (days)",
            30,
            730,
            365,
            help="Total duration of simulation",
        )
        time_step = st.slider(
            "Time Step (days)",
            0.1,
            5.0,
            1.0,
            help="Time increment for each simulation step",
        )

    # Update parameters
    st.session_state.macro_engine.params.baseline_viral_load = baseline_vl
    st.session_state.macro_engine.params.baseline_cd4 = baseline_cd4
    st.session_state.macro_engine.params.treatment_efficacy = treatment_efficacy
    st.session_state.macro_engine.params.resistance_development = resistance_rate

    # Buttons for simulation
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Full Simulation"):
            with st.spinner(f"Running simulation for {duration} days..."):
                st.session_state.macro_engine.run_simulation(duration, time_step)

                # Create and display plot
                fig = st.session_state.macro_engine.create_dynamic_plot()
                st.plotly_chart(fig, width="stretch", key="macro_view_full_sim_chart")

    with col2:
        if st.button("Run Real-time Simulation"):
            st.session_state.macro_engine.reset_simulation()
            st.session_state.macro_engine.create_real_time_simulation(duration, 0.01)

    with col3:
        if st.button("Reset Simulation"):
            st.session_state.macro_engine.reset_simulation()
            st.success("Simulation reset!")

    # Display current plot if simulation has been run
    if len(st.session_state.macro_engine.time_points) > 1:
        st.subheader("Current Dynamics")
        fig = st.session_state.macro_engine.create_dynamic_plot()
        st.plotly_chart(fig, width="stretch", key="macro_view_current_dynamics_chart")

        # Show summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Peak Viral Load",
                f"{max(st.session_state.macro_engine.viral_load_history):.2e}",
            )
            st.metric(
                "Min CD4 Count",
                f"{min(st.session_state.macro_engine.cd4_count_history):.0f}",
            )

        with col2:
            st.metric(
                "Last Viral Load",
                f"{st.session_state.macro_engine.viral_load_history[-1]:.2e}",
            )
            st.metric(
                "Current CD4",
                f"{st.session_state.macro_engine.cd4_count_history[-1]:.0f}",
            )

        with col3:
            st.metric(
                "Days Simulated", f"{len(st.session_state.macro_engine.time_points)-1}"
            )
            st.metric(
                "Resistance Level",
                f"{st.session_state.macro_engine.resistance_history[-1]*100:.1f}%",
            )


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="HIV Macro-View Dynamics", page_icon="📊", layout="wide"
    )

    show_macro_view()
