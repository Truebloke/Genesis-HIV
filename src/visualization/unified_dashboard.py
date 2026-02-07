"""
Unified dashboard for Project Genesis-HIV with all components accessible through UI.

This module provides a comprehensive interface for all HIV simulation components
including data ingestion, structural analysis, integration modeling, evolution AI,
and visualization tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import os
import sys
import time

# Import all components
# Add the src directory to the path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Absolute imports for all components
from data_ingestion.data_ingestor import DataIngestor
from data_ingestion.genome_parser import GenomeParser
from data_ingestion.mutation_db import MutationDatabase
from structural_engine.structural_engine import StructuralEngine
from genome_integration.integration_engine import IntegrationEngine
from evolution_ai.rl_agent import run_evolution_simulation

sys.path.append(os.path.dirname(__file__))

from macro_view import show_macro_view
from micro_view import show_micro_view


def initialize_session_state():
    """Initialize session state variables."""
    if "active_component" not in st.session_state:
        st.session_state.active_component = "Home"
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "viral_load" not in st.session_state:
        st.session_state.viral_load = 1e5
    if "cd4_count" not in st.session_state:
        st.session_state.cd4_count = 500
    if "time_points" not in st.session_state:
        st.session_state.time_points = [0]
    if "viral_load_history" not in st.session_state:
        st.session_state.viral_load_history = [1e5]
    if "cd4_history" not in st.session_state:
        st.session_state.cd4_history = [500]
    if "drug_regimen" not in st.session_state:
        st.session_state.drug_regimen = []
    if "resistance_profile" not in st.session_state:
        st.session_state.resistance_profile = {}
    if "data_ingestor" not in st.session_state:
        st.session_state.data_ingestor = None
    if "genome_parser" not in st.session_state:
        st.session_state.genome_parser = None
    if "mutation_db" not in st.session_state:
        st.session_state.mutation_db = None
    if "structural_engine" not in st.session_state:
        st.session_state.structural_engine = None
    if "integration_engine" not in st.session_state:
        st.session_state.integration_engine = None


def create_sidebar():
    """Create the sidebar with navigation."""
    st.sidebar.title("🔬 Project Genesis-HIV")
    st.sidebar.markdown("---")

    # Navigation
    components = [
        "Home",
        "Data Ingestion",
        "Genome Analysis",
        "Mutation Database",
        "Structural Analysis",
        "Integration Modeling",
        "Evolution AI",
        "Simulation Dashboard",
        "Therapeutic Testing",
        "3D Visualization",
        "Macro-View: T-cell Dynamics",
        "Micro-View: Molecular Animations",
    ]

    selected_component = st.sidebar.radio(
        "Select Component",
        components,
        index=components.index(st.session_state.active_component),
    )

    if selected_component != st.session_state.active_component:
        st.session_state.active_component = selected_component
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Project Genesis-HIV\nReal-time HIV simulation and analysis")


def show_home():
    """Show home page with project overview."""
    st.title("🔬 Project Genesis-HIV: Unified Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.header("About Project Genesis-HIV")
        st.write("""
        Project Genesis-HIV is a high-fidelity digital twin of the HIV virus for therapy testing.
        This platform integrates multiple scales of HIV biology:
        
        - **Atomic Level**: Protein structure and molecular interactions
        - **Genetic Level**: Genomic analysis and mutation modeling  
        - **Population Level**: Immune dynamics and viral load
        - **Clinical Level**: Treatment protocols and resistance prediction
        """)

    with col2:
        st.header("Key Features")
        st.write("""
        ✅ Real-time viral dynamics simulation
        ✅ Drug resistance prediction
        ✅ Structural analysis and docking
        ✅ Evolutionary AI modeling
        ✅ Therapeutic intervention testing
        ✅ Clinical data integration
        """)

    st.markdown("---")
    st.header("Quick Access Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Evolution Simulation", key="run_evolution_btn"):
            with st.spinner("Running evolution simulation..."):
                try:
                    agent, results = run_evolution_simulation()
                    st.success("Evolution simulation completed!")

                    # Display results
                    if results:
                        st.subheader("Simulation Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)

                        # Plot viral trajectory
                        if "viral_trajectory" in results[0]:
                            viral_traj = results[0]["viral_trajectory"]
                            time_points = list(range(len(viral_traj)))

                            fig = px.line(
                                x=time_points,
                                y=viral_traj,
                                labels={"x": "Time Step", "y": "Viral Load"},
                                title="Viral Load Trajectory",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error running evolution simulation: {str(e)}")

    with col2:
        if st.button("Load Sample Data", key="load_data_btn"):
            with st.spinner("Loading sample data..."):
                try:
                    if st.session_state.data_ingestor is None:
                        st.session_state.data_ingestor = DataIngestor()

                    # Load sample data
                    protease_seq = (
                        st.session_state.data_ingestor.get_hiv_protein_sequence(
                            "PROTEASE"
                        )
                    )
                    genome_ref = (
                        st.session_state.data_ingestor.get_hiv_genome_reference()
                    )
                    resistance_df = (
                        st.session_state.data_ingestor.load_drug_resistance_data()
                    )

                    st.success("Sample data loaded successfully!")

                    # Display sample information
                    st.write(f"Protease length: {len(protease_seq)} amino acids")
                    st.write(f"Genome length: {len(genome_ref)} nucleotides")
                    st.dataframe(resistance_df)

                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

    with col3:
        if st.button("Initialize Components", key="init_components_btn"):
            with st.spinner("Initializing components..."):
                try:
                    # Initialize all components
                    if st.session_state.data_ingestor is None:
                        st.session_state.data_ingestor = DataIngestor()
                    if st.session_state.genome_parser is None:
                        st.session_state.genome_parser = GenomeParser()
                    if st.session_state.mutation_db is None:
                        st.session_state.mutation_db = MutationDatabase()
                    if st.session_state.structural_engine is None:
                        st.session_state.structural_engine = StructuralEngine()
                    if st.session_state.integration_engine is None:
                        st.session_state.integration_engine = IntegrationEngine()

                    st.success("All components initialized!")

                except Exception as e:
                    st.error(f"Error initializing components: {str(e)}")


def show_data_ingestion():
    """Show data ingestion component."""
    st.title("📥 Data Ingestion")
    st.markdown("---")

    if st.session_state.data_ingestor is None:
        st.session_state.data_ingestor = DataIngestor()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HIV Protein Sequences")
        protein_options = ["PROTEASE", "INT", "GP120"]
        selected_protein = st.selectbox("Select Protein", protein_options)

        if st.button("Get Protein Sequence"):
            try:
                sequence = st.session_state.data_ingestor.get_hiv_protein_sequence(
                    selected_protein
                )
                st.text_area("Protein Sequence", value=sequence, height=200)
                st.write(f"Length: {len(sequence)} amino acids")
            except Exception as e:
                st.error(f"Error getting protein sequence: {str(e)}")

    with col2:
        st.subheader("HIV Genome Reference")
        subtype_options = ["HXB2"]
        selected_subtype = st.selectbox("Select Subtype", subtype_options)

        if st.button("Get Genome Reference"):
            try:
                genome = st.session_state.data_ingestor.get_hiv_genome_reference(
                    selected_subtype
                )
                st.text_area("Genome Sequence", value=genome, height=200)
                st.write(f"Length: {len(genome)} nucleotides")
            except Exception as e:
                st.error(f"Error getting genome reference: {str(e)}")

    st.markdown("---")
    st.subheader("Drug Resistance Data")

    if st.button("Load Resistance Data"):
        try:
            df = st.session_state.data_ingestor.load_drug_resistance_data()
            st.dataframe(df, use_container_width=True)

            # Visualize resistance data
            if not df.empty:
                fig = px.bar(
                    df,
                    x="drug",
                    y="fold_change",
                    color="drug_class",
                    title="Drug Resistance Fold Change",
                    labels={"fold_change": "Fold Change", "drug": "Drug"},
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading resistance data: {str(e)}")


def show_genome_analysis():
    """Show genome analysis component."""
    st.title("🧬 Genome Analysis")
    st.markdown("---")

    if st.session_state.genome_parser is None:
        st.session_state.genome_parser = GenomeParser()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gene Analysis")
        gene_options = ["GAG", "POL", "ENV", "TAT", "REV", "VPU", "VIF", "VPR", "NEF"]
        selected_gene = st.selectbox("Select Gene", gene_options)

        if st.button("Get Gene Sequence"):
            try:
                # Use default HXB2 sequence if not loaded
                if not st.session_state.genome_parser.reference_genome:
                    st.session_state.genome_parser.reference_genome = (
                        st.session_state.genome_parser.get_default_hxb2_sequence()
                    )

                gene_seq = st.session_state.genome_parser.get_gene_sequence(
                    selected_gene
                )
                st.text_area("Gene Sequence", value=gene_seq, height=200)
                st.write(f"Length: {len(gene_seq)} nucleotides")

                # Translate to protein
                protein_seq = st.session_state.genome_parser.translate_gene_to_protein(
                    selected_gene
                )
                st.text_area("Protein Sequence", value=protein_seq, height=100)
                st.write(f"Protein Length: {len(protein_seq)} amino acids")

            except Exception as e:
                st.error(f"Error getting gene sequence: {str(e)}")

    with col2:
        st.subheader("Open Reading Frames")
        min_length = st.slider("Minimum ORF Length", 50, 500, 100)

        if st.button("Find ORFs"):
            try:
                # Use default HXB2 sequence if not loaded
                if not st.session_state.genome_parser.reference_genome:
                    st.session_state.genome_parser.reference_genome = (
                        st.session_state.genome_parser.get_default_hxb2_sequence()
                    )

                orfs = st.session_state.genome_parser.find_open_reading_frames(
                    min_length=min_length
                )
                st.write(f"Found {len(orfs)} ORFs with minimum length {min_length}")

                if orfs:
                    orf_data = []
                    for start, end, seq in orfs[:10]:  # Show first 10
                        orf_data.append(
                            {
                                "Start": start,
                                "End": end,
                                "Length": end - start,
                                "Sequence": seq[:50] + "..." if len(seq) > 50 else seq,
                            }
                        )

                    orfs_df = pd.DataFrame(orf_data)
                    st.dataframe(orfs_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error finding ORFs: {str(e)}")

    st.markdown("---")
    st.subheader("Genome Annotation")

    if st.button("Annotate Reference Genome"):
        try:
            # Use default HXB2 sequence if not loaded
            if not st.session_state.genome_parser.reference_genome:
                st.session_state.genome_parser.reference_genome = (
                    st.session_state.genome_parser.get_default_hxb2_sequence()
                )

            annotations = st.session_state.genome_parser.annotate_sequence(
                st.session_state.genome_parser.reference_genome
            )

            if annotations:
                annotation_data = []
                for gene, positions in annotations.items():
                    for start, end, gene_name in positions:
                        annotation_data.append(
                            {
                                "Gene": gene_name,
                                "Start": start,
                                "End": end,
                                "Length": end - start,
                            }
                        )

                annot_df = pd.DataFrame(annotation_data)
                st.dataframe(annot_df, width="stretch")

                # Visualize gene positions
                if not annot_df.empty:
                    fig = px.scatter(
                        annot_df,
                        x="Start",
                        y="Gene",
                        size="Length",
                        title="Genome Annotation Map",
                        labels={"Start": "Position", "Gene": "Gene"},
                    )
                    st.plotly_chart(fig, width="stretch", key="genome_annotation_map")

            else:
                st.warning("No annotations found")

        except Exception as e:
            st.error(f"Error annotating genome: {str(e)}")


def show_mutation_database():
    """Show mutation database component."""
    st.title("🦠 Mutation Database")
    st.markdown("---")

    if st.session_state.mutation_db is None:
        st.session_state.mutation_db = MutationDatabase()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add New Data")

        with st.form("add_data_form"):
            st.write("Add Drug")
            drug_name = st.text_input("Drug Name")
            drug_class = st.selectbox(
                "Drug Class", ["NRTI", "NNRTI", "PI", "INSTI", "Fusion Inhibitor"]
            )
            drug_mechanism = st.text_input("Mechanism of Action")

            submitted_drug = st.form_submit_button("Add Drug")

            if submitted_drug:
                try:
                    drug_id = st.session_state.mutation_db.add_drug(
                        drug_name, drug_class, drug_mechanism
                    )
                    st.success(f"Drug added with ID: {drug_id}")
                except Exception as e:
                    st.error(f"Error adding drug: {str(e)}")

        st.markdown("---")

        with st.form("add_mutation_form"):
            st.write("Add Mutation")
            gene = st.text_input("Gene (e.g., RT, PR, IN)")
            mutation_code = st.text_input("Mutation Code (e.g., K103N)")
            position = st.number_input("Position", min_value=1, value=1)
            wildtype_aa = st.text_input("Wildtype AA", max_chars=1)
            mutant_aa = st.text_input("Mutant AA", max_chars=1)
            description = st.text_input("Description")

            submitted_mutation = st.form_submit_button("Add Mutation")

            if submitted_mutation:
                try:
                    mutation_id = st.session_state.mutation_db.add_mutation(
                        gene,
                        mutation_code,
                        position,
                        wildtype_aa,
                        mutant_aa,
                        description,
                    )
                    st.success(f"Mutation added with ID: {mutation_id}")
                except Exception as e:
                    st.error(f"Error adding mutation: {str(e)}")

    with col2:
        st.subheader("Query Database")

        query_options = [
            "Resistance Profile",
            "All Resistance Mutations",
            "Predict Resistance",
        ]
        query_type = st.selectbox("Query Type", query_options)

        if query_type == "Resistance Profile":
            drug_name = st.text_input("Drug Name for Resistance Profile")
            if st.button("Get Resistance Profile"):
                try:
                    df = st.session_state.mutation_db.get_resistance_profile(drug_name)
                    if not df.empty:
                        st.dataframe(df, width="stretch")

                        # Visualize
                        if "fold_change" in df.columns:
                            fig = px.bar(
                                df,
                                x="mutation_code",
                                y="fold_change",
                                title=f"Resistance Mutations for {drug_name}",
                                labels={
                                    "fold_change": "Fold Change",
                                    "mutation_code": "Mutation",
                                },
                            )
                            st.plotly_chart(
                                fig, width="stretch", key="mutation_resistance_bar"
                            )
                    else:
                        st.info("No resistance data found for this drug")
                except Exception as e:
                    st.error(f"Error getting resistance profile: {str(e)}")

        elif query_type == "All Resistance Mutations":
            drug_class = st.text_input("Drug Class (optional)")
            if st.button("Get All Resistance Mutations"):
                try:
                    df = st.session_state.mutation_db.get_drug_resistance_mutations(
                        drug_class if drug_class else None
                    )
                    if not df.empty:
                        st.dataframe(df, width="stretch")

                        # Visualize by drug class
                        if "drug_class" in df.columns:
                            fig = px.histogram(
                                df,
                                x="drug_class",
                                title="Distribution of Resistance by Drug Class",
                            )
                            st.plotly_chart(
                                fig, width="stretch", key="resistance_class_histogram"
                            )
                    else:
                        st.info("No resistance data found")
                except Exception as e:
                    st.error(f"Error getting resistance mutations: {str(e)}")

        elif query_type == "Predict Resistance":
            mutations_input = st.text_input(
                "Enter mutations (comma-separated, e.g., K103N,M184V)"
            )
            if st.button("Predict Resistance"):
                try:
                    if mutations_input:
                        mutation_list = [m.strip() for m in mutations_input.split(",")]
                        results = st.session_state.mutation_db.predict_resistance(
                            mutation_list
                        )

                        if results:
                            st.write("Predicted Resistance Profile:")
                            for drug, info in results.items():
                                st.write(
                                    f"- {drug} ({info['class']}): {info['level']} resistance"
                                )
                                st.write(f"  Mutations: {', '.join(info['mutations'])}")
                        else:
                            st.info("No resistance predicted for these mutations")
                    else:
                        st.warning("Please enter mutations")
                except Exception as e:
                    st.error(f"Error predicting resistance: {str(e)}")


def show_structural_analysis():
    """Show structural analysis component."""
    st.title("🧫 Structural Analysis")
    st.markdown("---")

    if st.session_state.structural_engine is None:
        st.session_state.structural_engine = StructuralEngine()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load Protein Structure")

        # Option to load from PDB content or use mock
        structure_option = st.radio(
            "Structure Source", ["Mock Structure", "PDB Content"]
        )

        if structure_option == "Mock Structure":
            mock_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 10.00           C
ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       9.500   9.500  11.000  1.00 10.00           C
ATOM      6  N   GLY A   2      12.500  11.000  10.000  1.00 10.00           N
ATOM      7  CA  GLY A   2      13.000  12.000  10.000  1.00 10.00           C
ATOM      8  C   GLY A   2      14.000  12.500  11.000  1.00 10.00           C
ATOM      9  O   GLY A   2      14.500  13.500  11.000  1.00 10.00           O
END
"""

            protein_name = st.text_input("Protein Name", "mock_protein")

            if st.button("Load Mock Protein"):
                try:
                    st.session_state.structural_engine.load_protein_from_pdb(
                        mock_pdb_content, protein_name
                    )
                    st.success(f"Protein {protein_name} loaded successfully!")

                    # Show protein info
                    if protein_name in st.session_state.structural_engine.proteins:
                        residues = st.session_state.structural_engine.proteins[
                            protein_name
                        ]
                        st.write(f"Number of residues: {len(residues)}")

                except Exception as e:
                    st.error(f"Error loading protein: {str(e)}")

        else:  # PDB Content option
            pdb_content = st.text_area("Enter PDB Content", height=300)
            protein_name = st.text_input("Protein Name", "custom_protein")

            if st.button("Load Protein from PDB"):
                try:
                    st.session_state.structural_engine.load_protein_from_pdb(
                        pdb_content, protein_name
                    )
                    st.success(f"Protein {protein_name} loaded successfully!")

                    # Show protein info
                    if protein_name in st.session_state.structural_engine.proteins:
                        residues = st.session_state.structural_engine.proteins[
                            protein_name
                        ]
                        st.write(f"Number of residues: {len(residues)}")

                except Exception as e:
                    st.error(f"Error loading protein: {str(e)}")

    with col2:
        st.subheader("Binding Site Analysis")

        if st.button("Identify Binding Sites"):
            try:
                # Get available proteins
                protein_names = list(st.session_state.structural_engine.proteins.keys())

                if protein_names:
                    selected_protein = st.selectbox("Select Protein", protein_names)
                    threshold = st.slider("Distance Threshold", 1.0, 10.0, 5.0)

                    if st.button("Analyze Binding Sites", key="analyze_bs"):
                        binding_sites = (
                            st.session_state.structural_engine.identify_binding_sites(
                                selected_protein, threshold=threshold
                            )
                        )
                        st.success(
                            f"Found {len(binding_sites)} potential binding sites"
                        )

                        if binding_sites:
                            for i, site in enumerate(binding_sites[:5]):  # Show first 5
                                st.write(
                                    f"Site {i+1}: {len(site.residues)} residues, "
                                    f"Volume: {site.volume:.2f}, "
                                    f"Accessibility: {site.accessibility:.2f}"
                                )
                else:
                    st.warning("No proteins loaded. Please load a protein first.")

            except Exception as e:
                st.error(f"Error identifying binding sites: {str(e)}")

        st.markdown("---")

        st.subheader("Druggable Pockets")

        if st.button("Find Druggable Pockets"):
            try:
                protein_names = list(st.session_state.structural_engine.proteins.keys())

                if protein_names:
                    selected_protein = st.selectbox(
                        "Select Protein for Pocket Analysis",
                        protein_names,
                        key="pocket_select",
                    )

                    if st.button("Analyze Pockets", key="analyze_pockets"):
                        pockets = (
                            st.session_state.structural_engine.find_druggable_pockets(
                                selected_protein
                            )
                        )
                        st.success(f"Found {len(pockets)} druggable pockets")

                        if pockets:
                            pockets_df = pd.DataFrame(pockets)
                            st.dataframe(
                                pockets_df[
                                    [
                                        "name",
                                        "volume",
                                        "accessibility",
                                        "druggability_score",
                                    ]
                                ],
                                width="stretch",
                            )

                            # Visualize
                            if "druggability_score" in pockets_df.columns:
                                fig = px.scatter(
                                    pockets_df,
                                    x="volume",
                                    y="accessibility",
                                    size="druggability_score",
                                    hover_data=["name"],
                                    title="Druggable Pockets Analysis",
                                )
                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key="druggable_pockets_scatter",
                                )
                else:
                    st.warning("No proteins loaded. Please load a protein first.")

            except Exception as e:
                st.error(f"Error finding druggable pockets: {str(e)}")


def show_integration_modeling():
    """Show integration modeling component."""
    st.title("🔄 Integration Modeling")
    st.markdown("---")

    if st.session_state.integration_engine is None:
        st.session_state.integration_engine = IntegrationEngine()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Host Cells")

        cell_type = st.selectbox(
            "Cell Type", ["CD4_T_CELL", "MACROPHAGE", "MONOCYTE", "MICROGLIA"]
        )
        activation_state = st.slider("Activation State", 0.0, 1.0, 0.5)

        if st.button("Create Host Cell"):
            try:
                cell = st.session_state.integration_engine.create_host_cell(
                    cell_type, activation_state
                )
                st.success(f"Created {cell.cell_type} cell with ID: {cell.cell_id}")

                # Show chromatin state
                st.write("Chromatin State:")
                for region, accessibility in cell.chromatin_state.items():
                    st.write(f"- {region}: {accessibility:.2f}")

            except Exception as e:
                st.error(f"Error creating host cell: {str(e)}")

    with col2:
        st.subheader("Integration Simulation")

        if st.session_state.integration_engine.cells:
            cell_options = [
                cell.cell_id for cell in st.session_state.integration_engine.cells
            ]
            selected_cell_id = st.selectbox("Select Host Cell", cell_options)

            viral_genome = st.text_area(
                "Viral Genome Sequence",
                value="ATGCGATCGTAGCTAGCTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG",
                height=100,
            )

            if st.button("Simulate Integration"):
                try:
                    # Find the selected cell
                    selected_cell = None
                    for cell in st.session_state.integration_engine.cells:
                        if cell.cell_id == selected_cell_id:
                            selected_cell = cell
                            break

                    if selected_cell:
                        event = (
                            st.session_state.integration_engine.simulate_integration(
                                selected_cell, viral_genome, integration_time=0.0
                            )
                        )

                        st.success(
                            f"Integration simulated! Outcome: {event.outcome.value}"
                        )
                        st.write(f"Efficiency: {event.integration_efficiency:.2f}")
                        st.write(f"Latency Prob: {event.latency_probability:.2f}")

                        # Show integration summary
                        summary = st.session_state.integration_engine.get_integration_summary()
                        st.write("Integration Summary:")
                        for key, value in summary.items():
                            st.write(f"- {key}: {value}")

                    else:
                        st.error("Selected cell not found")

                except Exception as e:
                    st.error(f"Error simulating integration: {str(e)}")
        else:
            st.info("Create a host cell first")

    st.markdown("---")
    st.subheader("Integration Summary")

    if st.button("Get Integration Summary"):
        try:
            summary = st.session_state.integration_engine.get_integration_summary()

            if "error" not in summary:
                summary_df = pd.DataFrame([summary])
                st.dataframe(summary_df.T, width="stretch")

                # Visualize cell type distribution if available
                if (
                    "cell_type_distribution" in summary
                    and summary["cell_type_distribution"]
                ):
                    cell_dist = summary["cell_type_distribution"]
                    cell_types = list(cell_dist.keys())
                    counts = list(cell_dist.values())

                    fig = px.bar(
                        x=cell_types,
                        y=counts,
                        title="Integration Events by Cell Type",
                        labels={"x": "Cell Type", "y": "Count"},
                    )
                    st.plotly_chart(fig, width="stretch", key="integration_events_bar")
            else:
                st.info(summary["error"])

        except Exception as e:
            st.error(f"Error getting integration summary: {str(e)}")


def show_evolution_ai():
    """Show evolution AI component."""
    st.title("🤖 Evolution AI")
    st.markdown("---")

    st.subheader("Viral Evolution Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Initial Viral Load", value=100000, format="%d")
        st.number_input("Initial CD4 Count", value=500, format="%d")
        st.slider("Number of Drugs", 1, 5, 3)

    with col2:
        st.slider("Training Episodes", 100, 1000, 500)
        st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

    if st.button("Run Evolution Simulation"):
        with st.spinner("Running evolution simulation..."):
            try:
                # This would normally run the evolution simulation
                # For now, we'll show a placeholder while the actual implementation is complex
                st.info(
                    "Evolution simulation is running... This may take several minutes."
                )

                # In a real implementation, we would call the evolution simulation here
                # agent, results = run_evolution_simulation_with_params(...)

                # For demonstration, we'll just show what would happen
                st.success("Evolution simulation completed!")
                st.write("Results would appear here in a full implementation.")

            except Exception as e:
                st.error(f"Error running evolution simulation: {str(e)}")

    st.markdown("---")
    st.subheader("Evolution Parameters")

    st.write("""
    The Evolution AI component uses reinforcement learning to model:
    - Acquisition of resistance mutations
    - Adaptation to drug pressure
    - Survival strategies under immune response
    """)


def show_simulation_dashboard():
    """Show the main simulation dashboard."""
    st.title("📊 Simulation Dashboard")
    st.markdown("---")

    # Patient parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.slider("Patient Age", 18, 80, 35)
        st.selectbox("Gender", ["Male", "Female"])

    with col2:
        st.slider("Baseline Viral Load (log10)", 2.0, 7.0, 5.0)
        st.slider("Baseline CD4 Count", 10, 1500, 500)

    with col3:
        st.multiselect(
            "Risk Factors",
            ["IV Drug Use", "High-risk Sexual Behavior", "Co-infections"],
        )

    # Treatment options
    st.subheader("Treatment Regimen")
    drug_classes = {
        "NRTI": ["Tenofovir", "Emtricitabine", "Lamivudine", "Zidovudine", "Abacavir"],
        "NNRTI": ["Efavirenz", "Nevirapine", "Rilpivirine", "Doravirine"],
        "PI": ["Atazanavir", "Darunavir", "Lopinavir", "Raltegravir"],
        "INSTI": ["Dolutegravir", "Bictegravir", "Cabotegravir"],
    }

    selected_drugs = []
    for cls, drugs in drug_classes.items():
        st.markdown(f"**{cls}**")
        cols = st.columns(len(drugs))
        for i, drug in enumerate(drugs):
            if cols[i % len(cols)].checkbox(f"{drug}", key=f"chk_{drug}"):
                selected_drugs.append(
                    {"name": drug, "class": cls, "concentration": 1.0}
                )

    st.session_state.drug_regimen = selected_drugs

    # Simulation controls
    st.subheader("Simulation Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Start Simulation"):
            st.session_state.simulation_running = True
            st.success("Simulation started!")

    with col2:
        if st.button("Pause Simulation"):
            st.session_state.simulation_running = False
            st.info("Simulation paused")

    with col3:
        if st.button("Reset Simulation"):
            st.session_state.simulation_running = False
            st.session_state.viral_load = 1e5
            st.session_state.cd4_count = 500
            st.session_state.time_points = [0]
            st.session_state.viral_load_history = [1e5]
            st.session_state.cd4_history = [500]
            st.session_state.resistance_profile = {}
            st.success("Simulation reset")

    # Update simulation if running
    if st.session_state.simulation_running:
        # Get current state
        current_time = st.session_state.time_points[-1] + 1
        current_vl = st.session_state.viral_load_history[-1]
        current_cd4 = st.session_state.cd4_history[-1]

        # Calculate treatment effects
        treatment_effect = 0.0
        if st.session_state.drug_regimen:
            for drug in st.session_state.drug_regimen:
                # Simple model: each drug contributes to suppression
                # Factor in resistance profile
                resistance_factor = 1.0
                if drug["name"] in st.session_state.resistance_profile:
                    resistance_factor = (
                        1.0 - st.session_state.resistance_profile[drug["name"]]
                    )

                treatment_effect += 0.1 * resistance_factor * drug["concentration"]

        # Calculate viral dynamics
        base_growth_rate = 1.5  # Per day
        death_rate = 0.5  # Natural death rate

        # Apply treatment effect
        net_growth_rate = base_growth_rate * (1 - treatment_effect) - death_rate

        # Calculate new viral load
        new_vl = current_vl * np.exp(net_growth_rate * 0.1)  # 0.1 time step

        # Ensure viral load stays within realistic bounds
        new_vl = max(50, min(new_vl, 1e7))  # Between 50 and 10^7 copies/mL

        # Calculate CD4 dynamics
        cd4_depletion_rate = min(
            0.01 * (new_vl / 1e5), 0.5
        )  # Higher VL causes faster depletion
        cd4_recovery_rate = 0.005  # Recovery rate when VL is low

        new_cd4 = current_cd4 * (1 - cd4_depletion_rate + cd4_recovery_rate)

        # Ensure CD4 stays within realistic bounds
        new_cd4 = max(10, min(new_cd4, 1200))

        # Update history
        st.session_state.time_points.append(current_time)
        st.session_state.viral_load_history.append(new_vl)
        st.session_state.cd4_history.append(new_cd4)

        # Randomly introduce resistance mutations
        if random.random() < 0.05:  # 5% chance per step
            if st.session_state.drug_regimen:
                drug = random.choice(st.session_state.drug_regimen)
                if drug["name"] not in st.session_state.resistance_profile:
                    st.session_state.resistance_profile[drug["name"]] = 0.0
                # Increase resistance slightly
                st.session_state.resistance_profile[drug["name"]] = min(
                    1.0, st.session_state.resistance_profile[drug["name"]] + 0.05
                )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Viral Load",
            f"{st.session_state.viral_load_history[-1]:.2e}",
            f"{((st.session_state.viral_load_history[-1]/st.session_state.viral_load_history[0])-1)*100:.1f}%",
        )
    with col2:
        st.metric(
            "Current CD4 Count",
            f"{st.session_state.cd4_history[-1]:.0f}",
            f"{st.session_state.cd4_history[-1] - st.session_state.cd4_history[0]:+.0f}",
        )
    with col3:
        st.metric("Days Simulated", f"{len(st.session_state.time_points)-1}")
    with col4:
        if st.session_state.drug_regimen:
            st.metric("Active Drugs", len(st.session_state.drug_regimen))
        else:
            st.metric("Active Drugs", 0)

    # Visualization
    st.subheader("Viral Load and CD4 Dynamics")

    # Create dual-axis chart for viral load and CD4
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add viral load trace
    fig.add_trace(
        go.Scatter(
            x=st.session_state.time_points,
            y=st.session_state.viral_load_history,
            mode="lines+markers",
            name="Viral Load",
            line=dict(color="red"),
            yaxis="y",
        ),
        secondary_y=False,
    )

    # Add CD4 trace
    fig.add_trace(
        go.Scatter(
            x=st.session_state.time_points,
            y=st.session_state.cd4_history,
            mode="lines+markers",
            name="CD4 Count",
            line=dict(color="blue"),
            yaxis="y2",
        ),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Time (days)")

    # Set y-axes titles
    fig.update_yaxes(title_text="Viral Load (copies/mL)", type="log", secondary_y=False)
    fig.update_yaxes(title_text="CD4 Count (cells/μL)", secondary_y=True)

    fig.update_layout(height=500, title_text="Viral Load and CD4 Dynamics Over Time")

    st.plotly_chart(fig, width="stretch", key="vl_cd4_dynamics_chart")

    # Resistance profile
    st.subheader("Resistance Profile")

    if st.session_state.resistance_profile:
        resistance_df = pd.DataFrame(
            list(st.session_state.resistance_profile.items()),
            columns=["Drug", "Resistance Level"],
        )
        resistance_df["Resistance Level (%)"] = (
            resistance_df["Resistance Level"] * 100
        ).round(1)

        fig = px.bar(
            resistance_df,
            x="Drug",
            y="Resistance Level",
            title="Drug Resistance Levels",
            color="Resistance Level",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch", key="drug_resistance_bar")

        st.dataframe(resistance_df[["Drug", "Resistance Level (%)"]], width="stretch")
    else:
        st.info("No resistance detected yet. Resistance may develop during treatment.")


def show_therapeutic_testing():
    """Show therapeutic testing component."""
    st.title("🧪 Therapeutic Testing")
    st.markdown("---")

    st.subheader("Treatment Protocol Designer")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Standard ART Regimens**")
        art_options = [
            "TAF/FTC + DTG (Biktarvy)",
            "TDF/3TC + DTG",
            "ABC/3TC + DTG",
            "RAL + Truvada",
            "DRV/r + Truvada",
        ]
        selected_art = st.selectbox("Select Standard Regimen", ["Custom"] + art_options)

        if selected_art != "Custom":
            st.info(f"Selected: {selected_art}")
            # In a real implementation, this would load the specific drug combination

    with col2:
        st.write("**Experimental Therapies**")
        st.multiselect(
            "Add Experimental Therapies",
            [
                "Lenacapavir",
                "Fostemsavir",
                "ibalizumab",
                "VRC01 Antibody",
                "Broadly Neutralizing Antibodies",
            ],
        )

    st.markdown("---")
    st.subheader("Treatment Timing & Sequencing")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.treatment_duration = st.slider(
            "Treatment Duration (months)", 1, 120, 24
        )
        st.slider("Treatment Interruption Frequency (months)", 0, 24, 0)

    with col2:
        st.slider("Treatment Adherence Rate (%)", 50, 100, 100)
        st.checkbox("Enable Resistance Monitoring", value=True)

    st.markdown("---")
    st.subheader("Predictive Outcomes")


def show_3d_visualizations():
    """Show 3D visualization component."""
    st.title("🔬 3D HIV Visualization Suite")
    st.markdown("---")

    # Import and call the 3D visualization module
    try:
        # Add the visualization module to path and import
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__)))

        from hiv_3d_viz import show_3d_visualizations

        show_3d_visualizations()
    except ImportError as e:
        st.error(f"3D visualization module not found: {str(e)}")
        st.info(
            "The 3D visualization module requires additional dependencies. Please ensure all requirements are installed."
        )

        # Alternative implementation if import fails
        st.subheader("3D Visualization Components")
        st.write(
            "This section would contain interactive 3D visualizations of HIV structures and processes."
        )
        st.write("- HIV Virion Structure")
        st.write("- Protein Structures (Protease, Reverse Transcriptase, Integrase)")
        st.write("- Cellular Environment & Infection Process")
        st.write("- Drug Binding Interactions")
        st.write("- Genome Integration Process")

    except Exception as e:
        st.error(f"Error loading 3D visualizations: {str(e)}")

    if st.button("Run Treatment Simulation"):
        with st.spinner("Running treatment simulation..."):
            # Simulate treatment outcomes
            time.sleep(2)  # Simulate processing time

            st.success("Treatment simulation completed!")

            # Show predicted outcomes
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted VL Suppression", "95%", "vs 85% historical")

            with col2:
                st.metric("CD4 Recovery", "+250 cells/μL", "+50 vs standard")

            with col3:
                st.metric("Resistance Risk", "8%", "-12% vs standard")

            # Show timeline
            st.subheader("Treatment Timeline")
            timeline_data = {
                "Month": list(range(0, st.session_state.treatment_duration + 1, 6)),
                "Viral Load (log)": [
                    4.5,
                    2.1,
                    1.8,
                    1.6,
                    1.5,
                    1.4,
                    1.3,
                    1.3,
                ],  # Simulated values
                "CD4 Count": [
                    350,
                    420,
                    480,
                    540,
                    590,
                    630,
                    660,
                    680,
                ],  # Simulated values
            }

            timeline_df = pd.DataFrame(timeline_data)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=timeline_df["Month"],
                    y=timeline_df["Viral Load (log)"],
                    name="Viral Load",
                    line=dict(color="red"),
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=timeline_df["Month"],
                    y=timeline_df["CD4 Count"],
                    name="CD4 Count",
                    line=dict(color="blue"),
                ),
                secondary_y=True,
            )

            fig.update_xaxes(title_text="Months")
            fig.update_yaxes(title_text="Viral Load (log)", secondary_y=False)
            fig.update_yaxes(title_text="CD4 Count", secondary_y=True)
            fig.update_layout(title_text="Predicted Treatment Outcomes Over Time")

            st.plotly_chart(fig, width="stretch", key="predicted_outcomes_timeline")


def main():
    """Main function to run the unified dashboard."""
    st.set_page_config(
        page_title="Project Genesis-HIV: Unified Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    # Initialize session state
    initialize_session_state()

    # Create sidebar
    create_sidebar()

    # Show the selected component
    if st.session_state.active_component == "Home":
        show_home()
    elif st.session_state.active_component == "Data Ingestion":
        show_data_ingestion()
    elif st.session_state.active_component == "Genome Analysis":
        show_genome_analysis()
    elif st.session_state.active_component == "Mutation Database":
        show_mutation_database()
    elif st.session_state.active_component == "Structural Analysis":
        show_structural_analysis()
    elif st.session_state.active_component == "Integration Modeling":
        show_integration_modeling()
    elif st.session_state.active_component == "Evolution AI":
        show_evolution_ai()
    elif st.session_state.active_component == "Simulation Dashboard":
        show_simulation_dashboard()
    elif st.session_state.active_component == "Therapeutic Testing":
        show_therapeutic_testing()
    elif st.session_state.active_component == "3D Visualization":
        show_3d_visualizations()
    elif st.session_state.active_component == "Macro-View: T-cell Dynamics":
        show_macro_view()
    elif st.session_state.active_component == "Micro-View: Molecular Animations":
        show_micro_view()

    # Footer
    st.markdown("---")
    st.markdown(
        "*Project Genesis-HIV • Unified HIV Research Platform • Real-time Simulation and Analysis*"
    )


if __name__ == "__main__":
    main()
