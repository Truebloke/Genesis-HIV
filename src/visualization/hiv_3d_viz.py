"""
3D visualization module for Project Genesis-HIV.

This module provides 3D visualization capabilities for:
- HIV viral structure
- Protein structures
- Cellular interactions
- Drug binding interactions
- Genome integration sites
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go


def _create_sphere_surface(center, radius, color, opacity=1.0, name="sphere"):
    """Helper to create a 3D sphere surface."""
    phi = np.linspace(0, 2 * np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
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
    )


def create_3d_virus_structure(
    capsid_radius: float = 1.0, num_proteins: int = 150
) -> go.Figure:
    """
    Create a 3D visualization of HIV viral structure.

    Args:
        capsid_radius: Radius of the viral capsid
        num_proteins: Number of envelope proteins to display

    Returns:
        Plotly figure with 3D virus visualization
    """
    fig = go.Figure()

    # Create spherical capsid
    phi, theta = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x_capsid = capsid_radius * np.sin(theta) * np.cos(phi)
    y_capsid = capsid_radius * np.sin(theta) * np.sin(phi)
    z_capsid = capsid_radius * np.cos(theta)

    # Add capsid surface
    fig.add_trace(
        go.Surface(
            x=x_capsid,
            y=y_capsid,
            z=z_capsid,
            opacity=0.3,
            colorscale="Blues",
            name="Capsid",
            showscale=False,
        )
    )

    # Add envelope proteins randomly distributed on surface
    u = np.random.random(num_proteins)
    v = np.random.random(num_proteins)
    theta_proteins = np.arccos(2 * u - 1)
    phi_proteins = 2 * np.pi * v

    x_proteins = (capsid_radius + 0.1) * np.sin(theta_proteins) * np.cos(phi_proteins)
    y_proteins = (capsid_radius + 0.1) * np.sin(theta_proteins) * np.sin(phi_proteins)
    z_proteins = (capsid_radius + 0.1) * np.cos(theta_proteins)

    # Add envelope proteins as tiny spheres for realism
    for i in range(
        min(num_proteins, 20)
    ):  # Limit number of physical meshes for performance
        fig.add_trace(
            _create_sphere_surface(
                [x_proteins[i], y_proteins[i], z_proteins[i]],
                0.05,
                "red",
                name="Envelope Protein",
            )
        )

    # Add remaining proteins as high-quality points if count is high
    if num_proteins > 20:
        fig.add_trace(
            go.Scatter3d(
                x=x_proteins[20:],
                y=y_proteins[20:],
                z=z_proteins[20:],
                mode="markers",
                marker=dict(size=4, color="red", symbol="circle"),
                name="Envelope Proteins (gp120/gp41)",
                opacity=0.8,
            )
        )

    # Add core (contains RNA and enzymes)
    core_radius = capsid_radius * 0.6
    phi_core, theta_core = np.mgrid[0 : 2 * np.pi : 15j, 0 : np.pi : 8j]
    x_core = core_radius * np.sin(theta_core) * np.cos(phi_core)
    y_core = core_radius * np.sin(theta_core) * np.sin(phi_core)
    z_core = core_radius * np.cos(theta_core)

    fig.add_trace(
        go.Surface(
            x=x_core,
            y=y_core,
            z=z_core,
            opacity=0.5,
            colorscale="Reds",
            name="Core",
            showscale=False,
        )
    )

    fig.update_layout(
        title="3D Structure of HIV Virion",
        scene=dict(
            xaxis=dict(title="X (nm)", range=[-2, 2]),
            yaxis=dict(title="Y (nm)", range=[-2, 2]),
            zaxis=dict(title="Z (nm)", range=[-2, 2]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
        ),
        width=800,
        height=600,
    )

    return fig


def create_protein_structure_3d(
    protein_name: str = "HIV Protease", num_residues: int = 99
) -> go.Figure:
    """
    Create a 3D visualization of a HIV protein structure.

    Args:
        protein_name: Name of the protein to visualize
        num_residues: Number of amino acid residues

    Returns:
        Plotly figure with 3D protein visualization
    """
    fig = go.Figure()

    # Generate a random protein backbone structure
    t = np.linspace(0, 4 * np.pi, num_residues)
    x = np.sin(t) + 0.1 * np.random.randn(num_residues)
    y = np.cos(t) + 0.1 * np.random.randn(num_residues)
    z = t / 2 + 0.1 * np.random.randn(num_residues)

    # Color by secondary structure (simplified)
    colors = []
    for i in range(num_residues):
        if i % 10 < 3:  # Alpha helix
            colors.append("red")
        elif i % 10 < 6:  # Beta sheet
            colors.append("yellow")
        else:  # Loop
            colors.append("blue")

    # Add protein backbone
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            line=dict(width=6, color="gray"),
            marker=dict(size=4, color=colors, colorscale="Viridis"),
            name="Protein Backbone",
        )
    )

    # Add binding site (for protease)
    if "protease" in protein_name.lower():
        # Add substrate binding site
        binding_site_x = [
            x[len(x) // 2] + 0.5,
            x[len(x) // 2] + 0.7,
            x[len(x) // 2] + 0.9,
        ]
        binding_site_y = [
            y[len(y) // 2] + 0.5,
            y[len(y) // 2] + 0.7,
            y[len(y) // 2] + 0.9,
        ]
        binding_site_z = [
            z[len(z) // 2] + 0.5,
            z[len(z) // 2] + 0.7,
            z[len(z) // 2] + 0.9,
        ]

        fig.add_trace(
            go.Scatter3d(
                x=binding_site_x,
                y=binding_site_y,
                z=binding_site_z,
                mode="markers",
                marker=dict(size=8, color="magenta", symbol="diamond"),
                name="Binding Site",
            )
        )

    fig.update_layout(
        title=f"3D Structure of {protein_name}",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        width=800,
        height=600,
    )

    return fig


def create_cellular_environment_3d() -> go.Figure:
    """
    Create a 3D visualization of cellular environment with HIV infection process.

    Returns:
        Plotly figure with 3D cellular environment
    """
    fig = go.Figure()

    # Host Cell parameters
    cell_radius = 5.0
    nucleus_radius = 2.0

    # Host Cell (CD4+ T-cell)
    fig.add_trace(
        _create_sphere_surface(
            [0.0, 0.0, 0.0],
            cell_radius,
            "lightblue",
            opacity=0.3,
            name="Host Cell",
        )
    )

    # Add nucleus
    fig.add_trace(
        _create_sphere_surface(
            [cell_radius - 1.5, 0.0, 0.0],
            nucleus_radius,
            "blue",
            opacity=0.5,
            name="Cell Nucleus",
        )
    )

    # Add some viruses approaching the cell
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = 8 + 2 * np.random.random()

        # Virus position
        x_virus = dist * np.cos(angle)
        y_virus = dist * np.sin(angle)
        z_virus = 2 * np.random.random() - 1

        # Draw virus as sphere with spikes
        create_3d_virus_structure(capsid_radius=0.3, num_proteins=20)

        # Add virus to main figure as physical 3D mesh for realism
        fig.add_trace(
            _create_sphere_surface(
                [x_virus, y_virus, z_virus],
                0.3,
                "red",
                name="HIV Virion",
            )
        )

    # Add some integrated provirus in the nucleus
    num_provirus = 15
    angles = np.random.random(num_provirus) * 2 * np.pi
    radii = np.random.random(num_provirus) * nucleus_radius * 0.8

    x_provirus = (cell_radius - 1.5) + radii * np.cos(angles)
    y_provirus = radii * np.sin(angles)
    z_provirus = np.random.random(num_provirus) * 2 - 1

    fig.add_trace(
        go.Scatter3d(
            x=x_provirus,
            y=y_provirus,
            z=z_provirus,
            mode="markers",
            marker=dict(size=4, color="purple", symbol="diamond"),
            name="Integrated Provirus",
        )
    )

    fig.update_layout(
        title="3D HIV Infection Process",
        scene=dict(
            xaxis=dict(title="X (nm)", range=[-10, 10]),
            yaxis=dict(title="Y (nm)", range=[-10, 10]),
            zaxis=dict(title="Z (nm)", range=[-10, 10]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1000,
        height=800,
    )

    return fig


def create_drug_binding_3d(
    drug_name: str = "Raltegravir", protein_name: str = "Integrase"
) -> go.Figure:
    """
    Create a 3D visualization of drug binding to target protein.
    """
    fig = go.Figure()

    # Create a simplified protein structure
    t = np.linspace(0, 4 * np.pi, 100)
    x_prot = np.sin(t) + 0.2 * np.random.randn(100)
    y_prot = np.cos(t) + 0.2 * np.random.randn(100)
    z_prot = t / 3 + 0.2 * np.random.randn(100)

    fig.add_trace(
        go.Scatter3d(
            x=x_prot,
            y=y_prot,
            z=z_prot,
            mode="lines",
            line=dict(width=8, color="lightblue"),
            name="Protein Structure",
        )
    )

    # Add binding pocket
    pocket_center = np.array([x_prot[50], y_prot[50], z_prot[50]])
    pocket_size = 1.5

    # Create binding pocket as a transparent sphere
    fig.add_trace(
        _create_sphere_surface(
            pocket_center,
            pocket_size,
            "blue",
            opacity=0.2,
            name="Binding Pocket",
        )
    )

    # Add drug molecule at binding site
    drug_x = pocket_center[0] + 0.3
    drug_y = pocket_center[1] + 0.3
    drug_z = pocket_center[2] + 0.3

    # Represent drug as a cluster of physical 3D spheres
    drug_atoms_x = [drug_x, drug_x + 0.3, drug_x - 0.2, drug_x + 0.1]
    drug_atoms_y = [drug_y, drug_y + 0.1, drug_y + 0.3, drug_y - 0.2]
    drug_atoms_z = [drug_z, drug_z + 0.2, drug_z - 0.1, drug_z + 0.3]

    atom_colors = ["red", "green", "orange", "yellow"]
    for i in range(len(drug_atoms_x)):
        fig.add_trace(
            _create_sphere_surface(
                [drug_atoms_x[i], drug_atoms_y[i], drug_atoms_z[i]],
                0.15,
                atom_colors[i % len(atom_colors)],
                name=f"Atom {i+1}",
            )
        )

    # Add binding interaction lines
    for i in range(len(drug_atoms_x)):
        fig.add_trace(
            go.Scatter3d(
                x=[drug_atoms_x[i], pocket_center[0]],
                y=[drug_atoms_y[i], pocket_center[1]],
                z=[drug_atoms_z[i], pocket_center[2]],
                mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                showlegend=False,
                opacity=0.5,
            )
        )

    fig.update_layout(
        title=f"3D Binding: {drug_name} vs {protein_name}",
        scene=dict(
            xaxis=dict(
                title="X (Å)", range=[pocket_center[0] - 5, pocket_center[0] + 5]
            ),
            yaxis=dict(
                title="Y (Å)", range=[pocket_center[1] - 5, pocket_center[1] + 5]
            ),
            zaxis=dict(
                title="Z (Å)", range=[pocket_center[2] - 5, pocket_center[2] + 5]
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=900,
        height=700,
    )

    return fig

    return fig


def create_genome_integration_3d() -> go.Figure:
    """
    Create a 3D visualization of HIV genome integration into host DNA.

    Returns:
        Plotly figure with 3D genome integration visualization
    """
    fig = go.Figure()

    # Create host DNA double helix
    t = np.linspace(0, 4 * np.pi, 100)
    radius = 2

    # Helix 1
    x1 = radius * np.cos(t)
    y1 = radius * np.sin(t)
    z1 = t

    # Helix 2 (offset by π)
    x2 = radius * np.cos(t + np.pi)
    y2 = radius * np.sin(t + np.pi)
    z2 = t

    # Add host DNA
    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            mode="lines",
            line=dict(width=8, color="green"),
            name="Host DNA Strand 1",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            mode="lines",
            line=dict(width=8, color="green"),
            name="Host DNA Strand 2",
        )
    )

    # Add nucleosomes along DNA
    for i in range(0, len(t), 10):
        # Place nucleosome spheres
        fig.add_trace(
            go.Scatter3d(
                x=[x1[i]],
                y=[y1[i]],
                z=[z1[i]],
                mode="markers",
                marker=dict(size=6, color="brown", symbol="sphere"),
                name="Nucleosome" if i == 0 else None,  # Only label first one
                showlegend=i == 0,
            )
        )

    # Create HIV provirus (integrated viral DNA)
    provirus_length = 20  # Equivalent to ~9kb in visualization terms
    provirus_t = np.linspace(0, provirus_length, 50)
    provirus_x = np.full_like(provirus_t, 0)  # Straight line
    provirus_y = np.full_like(provirus_t, 0)
    provirus_z = provirus_t + 10  # Offset in z direction

    fig.add_trace(
        go.Scatter3d(
            x=provirus_x,
            y=provirus_y,
            z=provirus_z,
            mode="lines",
            line=dict(width=10, color="red"),
            name="Integrated HIV Provirus",
        )
    )

    # Add viral genes as colored boxes
    gene_starts = [2, 6, 10, 14, 18]
    gene_lengths = [1.5, 2, 2.5, 1.5, 1.8]
    gene_names = ["LTR", "GAG", "POL", "ENV", "TAT/REV"]
    gene_colors = ["purple", "blue", "orange", "yellow", "pink"]

    for i, (start, length, name, color) in enumerate(
        zip(gene_starts, gene_lengths, gene_names, gene_colors)
    ):
        gene_x = [0, 0]
        gene_y = [0, 0]
        gene_z = [start + 10, start + length + 10]

        fig.add_trace(
            go.Scatter3d(
                x=gene_x,
                y=gene_y,
                z=gene_z,
                mode="lines",
                line=dict(width=15, color=color),
                name=name,
            )
        )

    # Add integration site markers
    integration_sites = [5, 15]  # Two integration sites
    for site in integration_sites:
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[site + 10, site + 10],
                mode="markers",
                marker=dict(size=10, color="black", symbol="diamond"),
                name=f"Integration Site at {site:.1f}",
            )
        )

    fig.update_layout(
        title="3D View of HIV Genome Integration into Host DNA",
        scene=dict(
            xaxis_title="X (kb)",
            yaxis_title="Y (kb)",
            zaxis_title="Z (kb)",
            aspectmode="manual",
            aspectratio=dict(x=0.5, y=0.5, z=1),
        ),
        width=900,
        height=700,
    )

    return fig


def show_3d_visualizations():
    """Display all 3D visualizations in the dashboard."""
    st.title("🧬 3D HIV Visualization Suite")
    st.markdown("---")

    st.markdown("""
    This section provides interactive 3D visualizations of HIV structures and processes.
    Use your mouse to rotate, zoom, and pan through the structures.
    """)

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔍 Viral Structure",
            "🧬 Protein Structure",
            "🧫 Cellular Environment",
            "💊 Drug Binding",
            "🧬 Genome Integration",
        ]
    )

    with tab1:
        st.subheader("HIV Virion Structure")
        st.markdown("""
        Interactive 3D model of the HIV virion showing:
        - Capsid (blue semi-transparent sphere)
        - Envelope proteins (red dots on surface)
        - Core containing viral RNA and enzymes (red inner sphere)
        """)

        capsid_radius = st.slider("Capsid Radius", 0.5, 2.0, 1.0, key="capsid_radius")
        num_proteins = st.slider(
            "Number of Envelope Proteins", 50, 300, 150, key="num_proteins"
        )

        if st.button("Generate HIV Structure", key="hiv_structure"):
            with st.spinner("Creating 3D HIV structure..."):
                fig = create_3d_virus_structure(capsid_radius, num_proteins)
                st.plotly_chart(fig, width="stretch", key="3d_hiv_structure_chart")

    with tab2:
        st.subheader("HIV Protein Structures")
        st.markdown("""
        3D visualization of key HIV proteins:
        - Protease: Critical for viral maturation
        - Reverse Transcriptase: Converts RNA to DNA
        - Integrase: Integrates viral DNA into host genome
        - Envelope proteins: Mediate cell entry
        """)

        protein_options = [
            "HIV Protease",
            "Reverse Transcriptase",
            "Integrase",
            "Envelope Protein",
        ]
        selected_protein = st.selectbox("Select Protein to Visualize", protein_options)
        num_residues = st.slider("Number of Residues", 50, 500, 99, key="num_residues")

        if st.button("Generate Protein Structure", key="protein_structure"):
            with st.spinner(f"Creating 3D structure of {selected_protein}..."):
                fig = create_protein_structure_3d(selected_protein, num_residues)
                st.plotly_chart(fig, width="stretch", key="3d_protein_structure_chart")

    with tab3:
        st.subheader("Cellular Environment & Infection Process")
        st.markdown("""
        3D visualization of HIV infection process:
        - T cell with nucleus
        - Free virions approaching the cell
        - Integrated provirus in the nucleus
        """)

        if st.button("Visualize Infection Process", key="infection_process"):
            with st.spinner("Creating 3D cellular environment..."):
                fig = create_cellular_environment_3d()
                st.plotly_chart(fig, width="stretch", key="3d_infection_process_chart")

    with tab4:
        st.subheader("Drug Binding Interactions")
        st.markdown("""
        Visualization of how antiretroviral drugs bind to their targets:
        - Binding pockets in viral proteins
        - Drug molecules at binding sites
        - Interaction networks
        """)

        drug_options = [
            "Raltegravir (INSTI)",
            "Efavirenz (NNRTI)",
            "Darunavir (PI)",
            "Tenofovir (NRTI)",
        ]
        protein_options = [
            "Integrase",
            "Reverse Transcriptase",
            "Protease",
            "RT Active Site",
        ]

        col1, col2 = st.columns(2)
        with col1:
            selected_drug = st.selectbox("Select Drug", drug_options)
        with col2:
            selected_protein = st.selectbox("Select Target Protein", protein_options)

        if st.button("Visualize Drug Binding", key="drug_binding"):
            with st.spinner(
                f"Creating 3D view of {selected_drug} binding to {selected_protein}..."
            ):
                fig = create_drug_binding_3d(
                    selected_drug.split()[0],
                    selected_protein.replace(" Active Site", ""),
                )
                st.plotly_chart(fig, width="stretch", key="3d_drug_binding_chart")

    with tab5:
        st.subheader("Genome Integration Process")
        st.markdown("""
        3D visualization of HIV genome integration:
        - Host DNA double helix
        - Integrated HIV provirus
        - Viral genes within the provirus
        - Integration sites
        """)

        if st.button("Visualize Genome Integration", key="genome_integration"):
            with st.spinner("Creating 3D genome integration view..."):
                fig = create_genome_integration_3d()
                st.plotly_chart(fig, width="stretch", key="3d_genome_integration_chart")

    st.markdown("---")
    st.markdown("### 📌 Tips for Navigation")
    st.markdown("""
    - **Rotate**: Click and drag to rotate the 3D structure
    - **Zoom**: Scroll to zoom in/out
    - **Pan**: Hold Shift and drag to pan
    - **Reset View**: Double-click on the plot to reset the view
    - **Toggle Elements**: Click on legend items to show/hide elements
    """)


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="HIV 3D Visualization Suite", page_icon="🔬", layout="wide"
    )

    show_3d_visualizations()
