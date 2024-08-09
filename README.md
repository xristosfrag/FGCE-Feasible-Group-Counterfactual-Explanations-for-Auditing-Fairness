# FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness
## Abstract

This paper introduces the first graph-based framework for generating group counterfactual explanations
to audit model fairness, a crucial aspect of trustworthy machine learning. Counterfactual explanations are instrumental in understanding and mitigating unfairness by revealing how inputs should change to achieve a desired outcome. Our framework, named Feasible Group Counterfactual Explanations (FGCEs), captures real-world feasibility constraints and constructs subgroups with similar counterfactuals, setting it apart from existing methods. It also addresses key trade-offs in counterfactual generation, including the balance between the number of counterfactuals, their associated costs, and the breadth of coverage achieved. To evaluate these trade-offs and assess fairness, we propose metrics tailored to group counterfactual generation. 
Our experimental results on benchmark datasets demonstrate that our approach effectively manages feasibility constraints and trade-offs, and the potential of our proposed metrics in identifying and quantifying fairness issues.

## Installation

To install the required dependencies for running the FGCE framework, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness.git

2. **Navigate to the project directory and then create a virtual environment (optional but recommended)**:
   ```bash
    pip install virtualenv
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate  # On Windows, use `<virtual-environment-name>\Scripts\activate`

3. **Install the required packages**:
   ```bash
    pip install -r requirements.txt