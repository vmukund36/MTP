# Mechanistic Interpretability of Safety Refusal Circuits in Llama-3

**Author:** Mukund Venkatasubramanian
**Institution:** University of Birmingham and IIT Madras
**Thesis Title:** Mechanistic Interpretability of Safety Refusal Circuits in Large Language Models: A Format-Dependent Analysis of Llama-3
**Model:** `meta-llama/Llama-3.1-8b-Instruct`

---

## 1. Project Overview

This repository hosts the experimental codebase and datasets for the Master's Thesis titled "Mechanistic Interpretability of Safety Refusal Circuits in Large Language Models." This research investigates the causal mechanisms governing safety refusals in instruction-tuned Large Language Models (LLMs). By integrating behavioral statistical analysis with mechanistic reverse-engineering techniques, this project identifies a sparse, format-dependent neural circuit responsible for enforcing safety alignment in Llama-3-8B.

### Key Contributions
* **Behavioral Format Sensitivity:** Establishes via non-parametric statistical testing ($p < 10^{-20}$) that safety refusal is context-dependent, exhibiting robust activation in completion-based formats (Fill-In-The-Blank, Multiple Choice) while failing in analysis-based formats (Question Answering).
* **Refusal Circuit Topology:** Maps a sparse sub-network of attention heads responsible for refusal, identifying Layer 29 as the critical decision bottleneck.
* **Component Specification:** Isolates specific "Guardrail Heads" (e.g., L29.H1) that write refusal tokens and "Compliance Heads" (e.g., L29.H3) that oppose safety mechanisms.

---

## 2. Methodology

The experimental pipeline is divided into two stages:

### Stage I: Behavioral Analysis
This stage quantifies the impact of prompt syntax on model safety using a controlled dataset.
* **Dataset Source:** We utilize the **AdvBench** dataset (Harmful Behaviors subset), widely considered the standard for evaluating refusal capabilities in aligned models.
* **Synthetic Control:** We select 60 distinct harmful instructions from AdvBench and wrap each in three syntactically distinct prompt formats to isolate the effect of syntax from semantic content:
    * **QA (Analysis):** "Is this harmful?"
    * **FIB (Completion):** "Assistant: I [cannot/will]"
    * **MCQ (Discrimination):** "Option A (Comply) vs Option B (Refuse)"
* **Metric:** Logit Difference (`Logit(Safe) - Logit(Unsafe)`).
* **Statistical Test:** Friedman Test followed by post-hoc Wilcoxon Signed-Rank tests.

### Stage II: Circuit Discovery
This stage employs mechanistic interpretability to localize the neural components responsible for the robust refusal observed in the FIB format.
* **Direct Logit Attribution (DLA):** Computes the direct contribution of each attention head to the final logit difference, identifying heads that write the refusal token.
* **Activation Patching (Causal Tracing):** Identifies heads that carry necessary safety information by transplanting activations from a "Clean" (Refusing) context to a "Corrupted" (Jailbroken) context and measuring the restoration of the refusal signal.

---

## 3. Installation and Requirements

This project is optimized for execution on a single NVIDIA A100 (80GB VRAM) GPU.

### Prerequisites
* Python 3.10+
* CUDA 11.8+
* PyTorch 2.1.0+

### Setup Instructions
1.  Clone the repository:
    ```bash
    git clone [https://github.com/vmukund36/MTP.git](https://github.com/vmukund36/MTP.git)
    cd MTP
    ```

2.  Create and activate a virtual environment:
    ```bash
    conda create -n mech_interp python=3.10
    conda activate mech_interp
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Dependencies
The core dependencies are listed in `requirements.txt`:
* `torch`
* `transformer_lens`
* `datasets`
* `pandas`
* `numpy`
* `scipy`
* `matplotlib`
* `seaborn`

---

## 4. Repository Structure

* `main.py`: The primary experimental script. It performs model loading, data synthesis, statistical testing, and circuit discovery (DLA and Patching) in a single execution flow.
* `requirements.txt`: List of Python libraries required for reproduction.
* `README.md`: Project documentation.
* `outputs/`: Directory containing generated data and visualizations.
    * `thesis_stats_behavioral.csv`: Raw logit difference scores for all samples across formats.
    * `thesis_circuit_dla.csv`: Attribution scores for all 1024 attention heads.
    * `thesis_circuit_patching.csv`: Activation patching restoration scores for all 1024 attention heads.
    * `safety_circuit_analysis.png`: Heatmaps visualizing the identified refusal circuit.

---

## 5. Usage

To run the full experimental pipeline, execute the main script:

```bash
python main.py
