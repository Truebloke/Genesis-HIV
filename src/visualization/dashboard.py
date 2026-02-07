"""
Research dashboard for Project Genesis-HIV.

This module provides a Streamlit-based interface for visualizing and controlling
the HIV simulation, including viral dynamics, drug resistance, and treatment protocols.
"""

import streamlit as st
from unified_dashboard import main as unified_main

def main():
    """Main function to run the unified dashboard."""
    unified_main()

if __name__ == "__main__":
    main()