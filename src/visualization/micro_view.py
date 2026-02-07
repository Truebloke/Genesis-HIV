"""
Micro-view module for Project Genesis-HIV: 3D animation of virion entry and CRISPR/Cas9.

This module provides 3D animations of:
- Virion entry into host cell
- CRISPR/Cas9 gene editing mechanism
- Molecular interactions at atomic level
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class AnimationFrame:
    """Represents a single frame in an animation."""

    time_point: float
    objects_positions: Dict[str, np.ndarray]  # Object name to 3D positions
    properties: Dict[str, Dict]  # Properties of objects (size, color, etc.)


class MicroViewEngine:
    """
    Engine for simulating and animating micro-level processes:
    - Virion entry into host cell
    - CRISPR/Cas9 gene editing
    - Molecular interactions
    """

    def __init__(self):
        """Initialize the MicroViewEngine."""
        self.animation_frames = []
        self.current_frame = 0
        self.animation_speed = 0.1  # seconds per frame

        # Animation parameters
        self.cell_radius = 5.0
        self.virion_radius = 0.5
        self.nucleus_radius = 2.0
        self.crispr_radius = 0.2

    def create_virion_entry_animation(
        self, duration: float = 10.0, steps: int = 100
    ) -> List[AnimationFrame]:
        """
        Create animation of virion entry into host cell.

        Args:
            duration: Duration of animation in seconds
            steps: Number of steps in animation

        Returns:
            List of animation frames
        """
        frames = []
        time_step = duration / steps

        for i in range(steps):
            t = i * time_step
            frame_time = t

            # Calculate positions
            # Virion moves toward cell
            approach_distance = self.cell_radius + self.virion_radius + 0.5
            entry_progress = min(1.0, t / (duration * 0.6))  # 60% of time for approach
            virion_x = -approach_distance * (1 - entry_progress)
            virion_y = 0.0
            virion_z = 0.0

            # Cell remains stationary
            cell_pos = np.array([0.0, 0.0, 0.0])

            # Nucleus remains inside cell
            nucleus_pos = np.array([0.5, 0.0, 0.0])  # Slightly offset from cell center

            # Create frame
            frame = AnimationFrame(
                time_point=frame_time,
                objects_positions={
                    "virion": np.array([virion_x, virion_y, virion_z]),
                    "cell": cell_pos,
                    "nucleus": nucleus_pos,
                },
                properties={
                    "virion": {"size": self.virion_radius, "color": "red"},
                    "cell": {"size": self.cell_radius, "color": "lightblue"},
                    "nucleus": {"size": self.nucleus_radius, "color": "blue"},
                },
            )

            # Add spike proteins to virion
            if "spikes" not in frame.properties:
                frame.properties["spikes"] = []

            # Add spikes around the virion surface
            num_spikes = 20
            for j in range(num_spikes):
                angle = 2 * np.pi * j / num_spikes
                spike_x = virion_x + self.virion_radius * 1.2 * np.cos(angle)
                spike_y = virion_y + self.virion_radius * 1.2 * np.sin(angle)
                spike_z = virion_z + self.virion_radius * 0.3 * np.random.randn()

                frame.objects_positions[f"spike_{j}"] = np.array(
                    [spike_x, spike_y, spike_z]
                )
                frame.properties[f"spike_{j}"] = {"size": 0.1, "color": "darkred"}

            frames.append(frame)

        self.animation_frames = frames
        return frames

    def create_crispr_mechanism_animation(
        self, duration: float = 15.0, steps: int = 150
    ) -> List[AnimationFrame]:
        """
        Create animation of CRISPR/Cas9 gene editing mechanism.

        Args:
            duration: Duration of animation in seconds
            steps: Number of steps in animation

        Returns:
            List of animation frames
        """
        frames = []
        time_step = duration / steps

        # Define DNA strand positions
        dna_length = 10.0
        dna_positions = np.linspace(-dna_length / 2, dna_length / 2, 20)

        for i in range(steps):
            t = i * time_step
            frame_time = t

            # DNA strand (double helix)
            dna_x = dna_positions
            dna_y = np.zeros_like(dna_positions)
            dna_z = np.zeros_like(dna_positions)

            # Add helical structure
            helix_offset = 0.5 * np.sin(2 * np.pi * dna_positions / dna_length)
            dna_z = helix_offset

            # CRISPR/Cas9 complex approaches DNA
            approach_progress = min(
                1.0, t / (duration * 0.4)
            )  # 40% of time for approach
            cas9_x = -6.0 + approach_progress * 4.0
            cas9_y = 1.0
            cas9_z = 0.0

            # Guide RNA
            grna_x = cas9_x - 0.3
            grna_y = cas9_y + 0.2
            grna_z = cas9_z

            # Create frame
            frame = AnimationFrame(
                time_point=frame_time,
                objects_positions={
                    "dna_strand1": np.column_stack([dna_x, dna_y, dna_z]),
                    "dna_strand2": np.column_stack([dna_x, dna_y + 0.5, dna_z]),
                    "cas9": np.array([cas9_x, cas9_y, cas9_z]),
                    "grna": np.array([grna_x, grna_y, grna_z]),
                },
                properties={
                    "dna_strand1": {"size": 0.3, "color": "green"},
                    "dna_strand2": {"size": 0.3, "color": "green"},
                    "cas9": {"size": 0.8, "color": "orange"},
                    "grna": {"size": 0.2, "color": "yellow"},
                },
            )

            # Add target recognition and cutting
            if t > duration * 0.4 and t < duration * 0.7:  # Recognition phase
                # Highlight target sequence
                target_idx = 10  # Middle of DNA
                if target_idx < len(dna_x):
                    frame.properties["target_highlight"] = {
                        "size": 0.6,
                        "color": "magenta",
                    }
                    frame.objects_positions["target_highlight"] = np.array(
                        [dna_x[target_idx], dna_y[target_idx], dna_z[target_idx]]
                    )

            elif t >= duration * 0.7:  # Cutting phase
                # Show DNA break
                cut_idx = 10
                if cut_idx < len(dna_x):
                    # Move DNA ends apart
                    left_end_x = dna_x[:cut_idx]
                    left_end_y = dna_y[:cut_idx]
                    left_end_z = dna_z[:cut_idx]

                    right_start_x = dna_x[cut_idx:]
                    right_start_y = dna_y[cut_idx:]
                    right_start_z = dna_z[cut_idx:]

                    # Add displacement to show break
                    right_displacement = (t - duration * 0.7) / (duration * 0.3) * 1.0
                    right_start_x += right_displacement

                    frame.objects_positions["dna_left"] = np.column_stack(
                        [left_end_x, left_end_y, left_end_z]
                    )
                    frame.objects_positions["dna_right"] = np.column_stack(
                        [right_start_x, right_start_y, right_start_z]
                    )
                    frame.properties["dna_left"] = {"size": 0.3, "color": "red"}
                    frame.properties["dna_right"] = {"size": 0.3, "color": "red"}

            frames.append(frame)

        self.animation_frames = frames
        return frames

    def create_molecular_interactions_animation(
        self, duration: float = 12.0, steps: int = 120
    ) -> List[AnimationFrame]:
        """
        Create animation of molecular interactions.

        Args:
            duration: Duration of animation in seconds
            steps: Number of steps in animation

        Returns:
            List of animation frames
        """
        frames = []
        time_step = duration / steps

        for i in range(steps):
            t = i * time_step
            frame_time = t

            # Create a protein structure
            protein_x = np.sin(np.linspace(0, 4 * np.pi, 30)) + 0.1 * np.random.randn(
                30
            )
            protein_y = np.cos(np.linspace(0, 4 * np.pi, 30)) + 0.1 * np.random.randn(
                30
            )
            protein_z = np.linspace(0, 6, 30) + 0.1 * np.random.randn(30)

            # Create a drug molecule approaching the protein
            approach_progress = min(1.0, t / (duration * 0.5))
            drug_x = 5.0 - approach_progress * 4.0
            drug_y = 0.0
            drug_z = 3.0

            # Binding site on protein
            binding_site_idx = 15
            binding_site_pos = np.array(
                [
                    protein_x[binding_site_idx],
                    protein_y[binding_site_idx],
                    protein_z[binding_site_idx],
                ]
            )

            # Create frame
            frame = AnimationFrame(
                time_point=frame_time,
                objects_positions={
                    "protein": np.column_stack([protein_x, protein_y, protein_z]),
                    "drug": np.array([drug_x, drug_y, drug_z]),
                    "binding_site": binding_site_pos,
                },
                properties={
                    "protein": {"size": 0.4, "color": "lightblue"},
                    "drug": {"size": 0.6, "color": "orange"},
                    "binding_site": {"size": 0.8, "color": "red"},
                },
            )

            # Add binding interaction when close enough
            if approach_progress > 0.8:
                dist_to_site = np.linalg.norm(
                    np.array([drug_x, drug_y, drug_z]) - binding_site_pos
                )
                if dist_to_site < 1.0:
                    # Show binding
                    frame.properties["binding_interaction"] = {
                        "size": 0.2,
                        "color": "purple",
                    }
                    midpoint = (
                        np.array([drug_x, drug_y, drug_z]) + binding_site_pos
                    ) / 2
                    frame.objects_positions["binding_interaction"] = midpoint

            frames.append(frame)

        self.animation_frames = frames
        return frames

    def _generate_sphere_mesh(self, center, radius, color, opacity=1.0, name="sphere"):
        """Generate a sphere mesh trace for improved realism."""
        # Use lower resolution (12x12) for smoother performance in animations
        phi = np.linspace(0, 2 * np.pi, 12)
        theta = np.linspace(0, np.pi, 12)
        phi, theta = np.meshgrid(phi, theta)

        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)

        return go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=name,
            showlegend=True,
            hoverinfo="name",
        )

    def generate_plotly_animation(
        self, animation_type: str, duration: float = 5.0, steps: int = 50
    ):
        """
        Generate a complete Plotly figure with animation frames for high performance.
        """
        # Generate raw frames data
        if animation_type == "entry":
            frames_data = self.create_virion_entry_animation(duration, steps)
        elif animation_type == "crispr":
            frames_data = self.create_crispr_mechanism_animation(duration, steps)
        elif animation_type == "interactions":
            frames_data = self.create_molecular_interactions_animation(duration, steps)
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")

        # Create the base figure (first frame)
        first_frame = frames_data[0]
        fig = go.Figure()

        # Add primary objects as surfaces
        for obj_name in ["virion", "cell", "nucleus"]:
            if obj_name in first_frame.objects_positions:
                opacity = 0.3 if obj_name == "cell" else 1.0
                fig.add_trace(
                    self._generate_sphere_mesh(
                        first_frame.objects_positions[obj_name],
                        first_frame.properties[obj_name]["size"],
                        first_frame.properties[obj_name]["color"],
                        opacity=opacity,
                        name=obj_name,
                    )
                )

        # Add other objects (spikes, etc.) as scatter
        for obj_name, pos in first_frame.objects_positions.items():
            if obj_name not in ["virion", "cell", "nucleus"]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[pos[0]] if pos.ndim == 1 else pos[:, 0],
                        y=[pos[1]] if pos.ndim == 1 else pos[:, 1],
                        z=[pos[2]] if pos.ndim == 1 else pos[:, 2],
                        mode="markers",
                        name=obj_name,
                        marker=dict(
                            size=first_frame.properties[obj_name]["size"] * 10,
                            color=first_frame.properties[obj_name]["color"],
                        ),
                    )
                )

        # Define frames
        animation_frames = []
        for i, frame in enumerate(frames_data):
            frame_traces = []
            # Surfaces
            for obj_name in ["virion", "cell", "nucleus"]:
                if obj_name in frame.objects_positions:
                    opacity = 0.3 if obj_name == "cell" else 1.0
                    frame_traces.append(
                        self._generate_sphere_mesh(
                            frame.objects_positions[obj_name],
                            frame.properties[obj_name]["size"],
                            frame.properties[obj_name]["color"],
                            opacity=opacity,
                            name=obj_name,
                        )
                    )

            # Scatters
            for obj_name, pos in frame.objects_positions.items():
                if obj_name not in ["virion", "cell", "nucleus"]:
                    frame_traces.append(
                        go.Scatter3d(
                            x=[pos[0]] if pos.ndim == 1 else pos[:, 0],
                            y=[pos[1]] if pos.ndim == 1 else pos[:, 1],
                            z=[pos[2]] if pos.ndim == 1 else pos[:, 2],
                            mode="markers",
                            marker=dict(
                                size=frame.properties[obj_name]["size"] * 10,
                                color=frame.properties[obj_name]["color"],
                            ),
                        )
                    )

            animation_frames.append(go.Frame(data=frame_traces, name=f"f{i}"))

        fig.frames = animation_frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Time:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": f"{frames_data[i].time_point:.2f}s",
                            "method": "animate",
                        }
                        for i, f in enumerate(fig.frames)
                    ],
                }
            ],
            scene=dict(
                xaxis=dict(title="X (nm)", range=[-10, 10]),
                yaxis=dict(title="Y (nm)", range=[-10, 10]),
                zaxis=dict(title="Z (nm)", range=[-10, 10]),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            title=f"{animation_type.capitalize()} Process Simulation",
            width=800,
            height=700,
        )

        return fig

    def generate_animation_frames(
        self, animation_type: str, duration: float = 10.0, steps: int = 100
    ):
        """
        Generate frames for a specific process.

        Args:
            animation_type: Type of animation ('entry', 'crispr', 'interactions')
            duration: Duration of animation in seconds
            steps: Number of steps in animation
        """
        if animation_type == "entry":
            self.animation_frames = self.create_virion_entry_animation(duration, steps)
        elif animation_type == "crispr":
            self.animation_frames = self.create_crispr_mechanism_animation(
                duration, steps
            )
        elif animation_type == "interactions":
            self.animation_frames = self.create_molecular_interactions_animation(
                duration, steps
            )
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")

        self.current_frame = 0


def show_micro_view():
    """Display the micro-view component in the dashboard."""
    st.title("🔬 Micro-View: 3D Animations of Molecular Processes")
    st.markdown("---")

    st.markdown("""
    This section provides 3D animations of key molecular processes:
    - **Virion Entry**: How HIV enters host cells
    - **CRISPR/Cas9 Mechanism**: Gene editing approach to HIV cure
    - **Molecular Interactions**: Drug-protein binding and other interactions
    """)

    # Initialize engine if not already done
    if "micro_engine" not in st.session_state:
        st.session_state.micro_engine = MicroViewEngine()

    # Animation selection
    animation_type = st.selectbox(
        "Select Animation to View",
        [
            "Virion Entry into Host Cell",
            "CRISPR/Cas9 Gene Editing",
            "Molecular Interactions",
        ],
    )

    # Animation controls
    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.slider("Animation Duration (seconds)", 5.0, 30.0, 10.0)

    with col2:
        steps = st.slider("Animation Steps", 50, 500, 100)

    with col3:
        speed = st.slider("Animation Speed", 0.01, 0.5, 0.1)

    # Update animation speed
    st.session_state.micro_engine.animation_speed = speed

    # Initialize session state for playback
    if "animation_fig" not in st.session_state:
        st.session_state.animation_fig = None

    # Animation button
    animation_mapping = {
        "Virion Entry into Host Cell": "entry",
        "CRISPR/Cas9 Gene Editing": "crispr",
        "Molecular Interactions": "interactions",
    }

    if st.button(f"Generate {animation_type} Animation"):
        with st.spinner("Rendering high-performance 3D animation..."):
            animation_key = animation_mapping[animation_type]
            st.session_state.animation_fig = (
                st.session_state.micro_engine.generate_plotly_animation(
                    animation_key, duration, steps
                )
            )

    # Show animation if exists
    if st.session_state.animation_fig is not None:
        st.plotly_chart(
            st.session_state.animation_fig,
            width="stretch",
            key="micro_view_native_animation",
        )

    # Educational content
    st.markdown("---")
    st.subheader("Educational Content")

    if animation_type == "Virion Entry into Host Cell":
        st.markdown("""
        ### HIV Entry Process
        
        HIV entry into host cells involves several key steps:
        
        1. **Attachment**: The viral envelope protein gp120 binds to the CD4 receptor on T-helper cells
        2. **Coreceptor Binding**: gp120 undergoes conformational changes allowing binding to CCR5 or CXCR4 coreceptors
        3. **Membrane Fusion**: gp41 mediates fusion of viral and cellular membranes
        4. **Viral Entry**: The viral capsid enters the cytoplasm
        5. **Uncoating**: The viral RNA genome is released
        6. **Reverse Transcription**: Viral RNA is converted to DNA by reverse transcriptase
        7. **Integration**: Viral DNA is integrated into the host genome by integrase
        
        Understanding these steps is crucial for developing entry inhibitors and other therapeutic interventions.
        """)

    elif animation_type == "CRISPR/Cas9 Gene Editing":
        st.markdown("""
        ### CRISPR/Cas9 for HIV Cure
        
        CRISPR/Cas9 technology offers potential for HIV cure by:
        
        1. **Target Recognition**: Guide RNA directs Cas9 to specific HIV sequences
        2. **DNA Binding**: Cas9 protein binds to target DNA sequence
        3. **Double-Strand Break**: Cas9 cuts both strands of DNA
        4. **Excision**: The integrated HIV provirus is removed
        5. **Repair**: Host DNA repair mechanisms restore the original sequence
        
        This approach aims to eliminate the latent reservoir by excising integrated proviral DNA.
        """)

    elif animation_type == "Molecular Interactions":
        st.markdown("""
        ### Molecular Interactions in HIV
        
        Key molecular interactions in HIV lifecycle:
        
        - **Protease Inhibitors**: Bind to viral protease preventing cleavage of Gag/Pol polyproteins
        - **Reverse Transcriptase Inhibitors**: Block conversion of viral RNA to DNA
        - **Integrase Inhibitors**: Prevent integration of viral DNA into host genome
        - **Entry Inhibitors**: Block viral attachment or fusion with host cell membrane
        
        Understanding these interactions helps in drug design and resistance prediction.
        """)


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="HIV Micro-View Animations", page_icon="🔬", layout="wide"
    )

    show_micro_view()
