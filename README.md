# Project Genesis-HIV: Advanced HIV Simulation Platform

## Overview
Project Genesis-HIV is a high-fidelity digital twin of the HIV virus for therapy testing and cure research. This platform integrates multiple scales of HIV biology to enable realistic simulation of viral dynamics, immune responses, and therapeutic interventions.

The project aims to accelerate HIV cure research by providing a comprehensive in silico environment where researchers can test therapeutic strategies, model viral evolution, and predict treatment outcomes with high biological fidelity.

## Key Features

### Multi-Scale Modeling
- **Atomic Level**: Protein structure and molecular interactions with OpenMM integration
- **Genetic Level**: Genomic analysis, mutation modeling, and resistance prediction
- **Cellular Level**: Immune dynamics, viral replication, and cell-to-cell transmission
- **Population Level**: Viral load dynamics and treatment responses
- **Clinical Level**: Treatment protocols and patient outcomes

### Core Components
- **Data Ingestion**: Integration with PDB, LANL HIV DB, and Stanford Drug Resistance DB
- **Structural Analysis**: 3D protein modeling and molecular docking
- **Genome Integration**: Modeling of viral integration into host genome
- **Evolution AI**: Reinforcement learning for viral evolution and resistance development
- **Therapeutic Testing**: Evaluation of single and combination interventions
- **Real-time Visualization**: Interactive dashboards with 3D animations

### Biological Realism
- Realistic mutation rates based on RT fidelity data
- Detailed immune system modeling (CD4+, CD8+ T cells, B cells, cytokines)
- Accurate PK/PD models for antiretroviral drugs
- Latent reservoir modeling with reactivation dynamics
- Tissue-specific compartments and drug penetration

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd project_genesis_hiv
```

2. Create and activate virtual environment:
```bash
python -m venv hiv_env
source hiv_env/bin/activate  # On Windows: hiv_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard
The main interface is accessed through the unified dashboard:

```bash
streamlit run src/visualization/unified_dashboard.py
```

Then open your browser to `http://localhost:8501`

### Dashboard Components

#### Home
Overview of the project and quick access to main functions

#### Data Ingestion
- Load HIV protein sequences from PDB
- Access HIV genome references
- Query drug resistance data

#### Genome Analysis
- Gene sequence analysis
- Open reading frame detection
- Genome annotation

#### Mutation Database
- Add and query mutations
- Predict drug resistance
- Manage drug information

#### Structural Analysis
- Load and analyze protein structures
- Identify binding sites
- Find druggable pockets

#### Integration Modeling
- Model viral integration into host genome
- Simulate reservoir establishment
- Model latent infection dynamics

#### Evolution AI
- Run viral evolution simulations
- Model resistance development
- Test therapeutic interventions

#### Simulation Dashboard
- Monitor viral load and CD4 counts
- Control treatment regimens
- Visualize dynamics over time

#### Therapeutic Testing
- Design treatment protocols
- Test combination therapies
- Evaluate intervention effectiveness

#### 3D Visualization
- Interactive 3D models of HIV structures
- Drug binding interactions
- Genome integration visualization

#### Macro-View: T-cell Dynamics
- Real-time visualization of T-cell and viral load dynamics
- Population-level modeling
- Treatment response tracking

#### Micro-View: Molecular Animations
- 3D animations of virion entry into cells
- CRISPR/Cas9 gene editing mechanisms
- Molecular interaction dynamics

## Research Applications

### For HIV Cure Research
- Test shock-and-kill approaches (latency reversing agents)
- Evaluate gene therapy strategies (CCR5 editing, CAR-T cells)
- Model combination interventions for synergistic effects
- Predict treatment-free remission probability
- Assess resistance risks with new approaches

### For Clinical Trial Design
- Simulate patient populations with realistic heterogeneity
- Predict treatment outcomes for different regimens
- Optimize dosing regimens and timing
- Assess biomarker responses to interventions

### For Basic Research
- Study viral evolution dynamics under different pressures
- Analyze immune-virus interactions in detail
- Investigate reservoir establishment and maintenance mechanisms
- Test novel therapeutic hypotheses in silico

## Architecture

### Technology Stack
- **Backend**: Python 3.10+ (NumPy, SciPy, Pandas)
- **Bioinformatics**: Biopython, RDKit, OpenMM
- **AI/ML**: PyTorch, Stable Baselines3, Gymnasium
- **UI/Visualization**: Streamlit, Plotly, PyVista, NGLView
- **Data**: SQLAlchemy, Requests (for PDB/LANL/Stanford DB)

### Directory Structure
```
project_genesis_hiv/
├── src/
│   ├── data_ingestion/      # Data loading and parsing
│   ├── structural_engine/   # Protein structure analysis
│   ├── genome_integration/  # Integration modeling
│   ├── evolution_ai/        # AI and evolution modeling
│   ├── visualization/       # Dashboard and visualization
│   └── ...
├── tests/                   # Unit and integration tests
├── data/                    # Sample data files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Validation

The model has been validated against:
- Clinical trial data from major studies
- In vitro experimental results
- Observational cohort data
- Published resistance patterns
- Known viral dynamics parameters

## Contributing

We welcome contributions to enhance the platform further. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Project Genesis-HIV in your research, please cite:

```
Project Genesis-HIV: Advanced HIV Simulation Platform for Cure Research
Authors: [To be updated with actual authorship]
Year: 2026
```

## Support

For support, please contact the development team or submit an issue on the GitHub repository.

## Acknowledgments

This project integrates data and concepts from numerous research institutions and databases:
- RCSB Protein Data Bank
- Los Alamos HIV Sequence Database
- Stanford HIV Drug Resistance Database
- ClinicalTrials.gov
- Research teams worldwide studying HIV pathogenesis and cure strategies