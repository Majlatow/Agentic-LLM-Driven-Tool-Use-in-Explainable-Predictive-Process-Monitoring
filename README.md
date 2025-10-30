# Agentic LLM-Driven Explainable Predictive Process Monitoring

This repository contains the proof-of-concept implementation of an **agentic large language model (LLM)-driven architecture** for **explainable predictive process monitoring (PPM)**.  
The system autonomously decomposes analytical queries into executable subtasks, orchestrates process mining and XAI tools, and generates transparent, human-interpretable outputs through structured reasoning and dependency-aware orchestration.

## Framework Overview

The **XPPM-Agent** architecture is composed of multiple interacting LLM-based agents:
- **Orchestrator Agent:** Coordinates query interpretation, dependency management, and result synthesis.  
- **Planner Agent:** Performs hierarchical task decomposition and constructs dependency-aware directed acyclic graphs (DAGs).  
- **Tool Matcher Agent:** Identifies and selects suitable computational tools for each subtask.  
- **Filler Agent:** Prepares tool input parameters and validates consistency across subtasks.  

The system integrates a **GBM model** for event-level processing time prediction, **SHAP** for feature attribution-based explanations, and a **Quantile Regression Forest (QRF)** surrogate for uncertainty quantification using conformal prediction intervals.

## Features
- Agentic orchestration of analytical pipelines for PPM and XAI tasks.  
- Autonomous task decomposition with dependency tracking.  
- SHAP-based model explainability and QRF-based uncertainty quantification.  
- Modular integration of internal and external analytical tools.  
- Transparent, JSON-traceable reasoning and reproducible workflows.  

## Installation

Clone the repository, add datasets (after request and permission) and install the dependencies listed in `requirements.txt`
