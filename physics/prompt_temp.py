from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class ChemistryPrompt:
    def __init__(self):
        pass
    def chemisry_question_generate(self, items: str, question: str) -> str:
        prompt = f"""
        Create a new reasonable, scientifically grounded, and solvable chemistry problem from the original problem by applying some of the following transformations (focus primarily on the transformations of "{items}"):

        1. Alter chemical parameters: Change concentrations, temperatures, pressures, or chemical species. **CRITICAL: The new values must be physically realistic (e.g., temperatures must be above absolute zero, concentrations should be within solubility limits) and chemical species must follow standard valency and bonding rules.**
        2. Introduce practical laboratory or industrial constraints: Add concepts like limiting reagents, actual percent yield, impurities, or non-ideal gas behavior (Van der Waals constants). Ensure the scenario mirrors real-world chemical practice.
        3. Reverse the problem logic (Inverse Problems): Instead of asking for the final outcome, provide the observed result (e.g., a specific pH, equilibrium constant, or product mass) and ask the user to deduce the initial conditions or the identity of an unknown reagent.
        4. Increase structural or mechanistic complexity: For organic/inorganic chemistry, introduce concepts like stereochemistry (R/S, E/Z), regioselectivity (Markovnikov/anti-Markovnikov), or multi-step synthetic pathways. **The proposed reaction mechanisms must be scientifically plausible.**
        5. Integrate multiple chemical domains: Combine thermodynamics with kinetics, or electrochemistry with stoichiometry. **Ensure all provided physical constants (e.g., ΔH, ΔG, R, Kw, Faraday's constant) are consistent with each other and the chosen chemical system.**

        ### STRICT SCIENTIFIC CONSTRAINTS:
        - **Chemical Feasibility:** Do not invent non-existent compounds or impossible oxidation states.
        - **Physical Laws:** All generated problems must obey the Laws of Thermodynamics, Conservation of Mass, and Charge Balance.
        - **Environmental Consistency:** For aqueous reactions, ensure the temperature and pressure allow water to remain in the appropriate phase unless phase change is the focus.
        - **Numerical Integrity:** Ensure that the problem is mathematically solvable and that the provided data does not lead to a logical contradiction (e.g., a negative Kelvin temperature or a yield greater than 100%).

        Here is the original problem:
        {question}
        
        Write another chemistry problem inspired by this one. 
        Focus on shifting the analytical approach while maintaining absolute scientific accuracy. 
        Start directly with the problem statement and DO NOT include any introductory or concluding phrases.
        After the problem is generated, finish your response right away.
        """
        return prompt
    def build_classification_prompt(self, question: str) -> str:
        prompt = f"""
        You are a classification assistant specialized in chemistry. Your task is to classify the given text into one primary category and one secondary category strictly according to the following taxonomy. 

        Do not output any extra explanation. Return ONLY a valid JSON object with the keys "primary_category" and "secondary_category".

        ### CRITICAL FORMATTING INSTRUCTION:
        You must use the exact English names from the taxonomy below, but you must STRICTLY EXCLUDE all numerical prefixes (e.g., "1.", "2.3"). 
        - Correct Output Example: {{"primary_category": "Organic Chemistry", "secondary_category": "Organic Synthesis"}}
        - Incorrect Output Example: {{"primary_category": "2. Organic Chemistry", "secondary_category": "2.3"}}
        - Incorrect Output Example: {{"primary_category": "Organic Chemistry", "secondary_category": "2.3 Organic Synthesis"}}

        Taxonomy:
        1. Inorganic Chemistry
        - 1.1 Elemental and Isotope Chemistry
        - 1.2 Coordination Chemistry
        - 1.3 Inorganic Synthesis and Separation Chemistry
        - 1.4 Solid-State and Physical Inorganic Chemistry
        - 1.5 Bioinorganic Chemistry

        2. Organic Chemistry
        - 2.1 Organoelemental and Organometallic Chemistry
        - 2.2 Natural Products Chemistry
        - 2.3 Organic Synthesis
        - 2.4 Physical Organic and Organic Photochemistry
        - 2.5 Bioorganic Chemistry

        3. Analytical Chemistry
        - 3.1 Chemical and Electrochemical Analysis
        - 3.2 Spectroscopy, Spectrometry, and Mass Spectrometry
        - 3.3 Chromatography and Separation Analysis
        - 3.4 Thermal, Radiochemical, and Phase Analysis
        - 3.5 Chemometrics

        4. Physical Chemistry and Chemical Physics
        - 4.1 Chemical Thermodynamics and Thermochemistry
        - 4.2 Chemical Kinetics and Catalysis
        - 4.3 Quantum Chemistry, Structural Chemistry, and Computational Chemistry
        - 4.4 Colloid and Interface Chemistry
        - 4.5 Electrochemistry, Magnetochemistry, and Photochemistry

        5. Polymer Chemistry and Physics
        - 5.1 Polymer Synthesis
        - 5.2 Physical Chemistry of Polymers (including Polymer Physics)
        - 5.3 Functional, Natural, and Inorganic Polymers

        6. Nuclear Chemistry
        - 6.1 Radiochemistry and Environmental Radiochemistry
        - 6.2 Nuclear Reaction Chemistry (Fission, Fusion, Transmutation)

        7. Applied and Interdisciplinary Chemistry
        - 7.1 Applied Chemistry and Chemical Engineering
        - 7.2 Chemical Biology
        - 7.3 Materials Chemistry (Nanochemistry, Carbon Chemistry, Soft Chemistry)
        - 7.4 History of Chemistry

        Classify the following text based on the taxonomy above.
        Text: {question}
        """
        return prompt


@PROMPT_REGISTRY.register()
class PhysicsPrompt:
    def __init__(self):
        pass
    def build_physics_prompt(self, items: str, question: str) -> str:
        prompt = f"""
        Create a new, highly valuable, and rigorous physics/chemistry problem from the original problem by applying some of the following transformations (focus specifically on the transformations of "{items}"):

        1. Modify Physical Assumptions: Transition the problem from ideal/theoretical conditions to realistic ones (e.g., introduce friction, non-ideal fluid behavior, quantum decoherence, or relativistic effects).
        2. Cross-Domain Integration: Combine the core physical concept with another discipline or subfield (e.g., linking quantum mechanics with thermodynamics, or astrophysics with particle physics).
        3. Experimental Contextualization: Shift the focus from pure theory to experimental design, asking about observable signatures, measurement challenges, noise mitigation, or data interpretation in a laboratory setting.
        4. Quantitative/Qualitative Pivot: Transform a conceptual "how/why" question into a rigorous quantitative calculation by introducing specific variables, boundary conditions, and physical constants; or vice versa, requiring a mathematical derivation to prove a concept.
        5. System Scaling and Complexity: Expand the problem from a single-particle/two-body system to a many-body system, or introduce time-dependent perturbations and external fields that necessitate case-by-case analysis.

        [Rigorous Quality Control & Physical Consistency - STRICTLY ENFORCED]
        - Complete Solvability: The problem MUST be entirely self-contained. You must explicitly provide all necessary physical constants, material properties, initial states, and boundary conditions required to reach the final answer. Do not leave any variables undefined unless they are meant to cancel out in the derivation.
        - No Contradictions: Ensure that none of the provided numerical values, constraints, or assumptions contradict one another.
        - Real-World Plausibility: The scenario MUST adhere to the fundamental laws of physics and chemistry (e.g., conservation of energy/momentum, thermodynamic limits, realistic material limitations). Do not generate physically impossible scenarios (like a macroscopic object traveling faster than light, or an engine exceeding Carnot efficiency).

        Here is the original problem:
        {question}

        Write another problem inspired by this one. 
        Not only change the physical scenario, but also try to create a new problem that requires a deeper or entirely different analytical approach to solve. Ensure the new problem is highly challenging but perfectly logically sound.
        Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
        After the problem is generated, finish your response right away.
        """
        return prompt
    
    def build_physics_classification_prompt(self, question: str) -> str:
        prompt = f"""
        You are a classification assistant specialized in Physics. Your task is to classify the given text into one primary category and one secondary category according to the following taxonomy. Do not output any extra explanation. Return only a JSON object with the keys "primary_category" and "secondary_category".

        Taxonomy:
        1. Classical Mechanics and Fluid Dynamics
        - 1.1 Kinematics, Dynamics, and Statics (Basic Newtonian mechanics)
        - 1.2 Solid Mechanics and Materials (Elasticity, fracture, deformation)
        - 1.3 Fluid Mechanics and Aerodynamics (Hydrodynamics, gases, turbulence)
        - 1.4 Vibrations, Oscillations, and Acoustics (Mechanical waves, sound)

        2. Thermodynamics and Statistical Mechanics
        - 2.1 Laws of Thermodynamics and Heat Transfer (Macroscopic thermal physics)
        - 2.2 Kinetic Theory of Gases (Ideal and non-ideal gases)
        - 2.3 Statistical Mechanics (Microstates, partition functions, phase transitions)

        3. Electromagnetism and Electronics
        - 3.1 Electrostatics and Magnetostatics (Electric/magnetic fields, Coulomb's/Biot-Savart law)
        - 3.2 Electrodynamics and Maxwell's Equations (Electromagnetic waves, radiation)
        - 3.3 Electronic Circuits and Devices (AC/DC circuits, basic electronics)

        4. Optics and Light
        - 4.1 Geometrical/Ray Optics (Lenses, mirrors, refraction, reflection)
        - 4.2 Physical/Wave Optics (Interference, diffraction, polarization)
        - 4.3 Quantum Optics, Lasers, and Nonlinear Optics

        5. Quantum Mechanics and Atomic/Molecular Physics
        - 5.1 Foundations of Quantum Mechanics (Schrödinger equation, wavefunctions, uncertainty)
        - 5.2 Atomic and Molecular Structure (Spectra, electron configurations)
        - 5.3 Quantum Information and Computation (Qubits, entanglement, quantum cryptography)

        6. Condensed Matter and Plasma Physics
        - 6.1 Solid State Physics and Crystallography (Lattices, phonons, band theory)
        - 6.2 Superconductivity, Magnetism, and Semiconductors
        - 6.3 Plasma Physics (Ionized gases, magnetohydrodynamics)

        7. Relativity, Particle Physics, and Astrophysics
        - 7.1 Special and General Relativity (Spacetime, gravitational waves, black holes)
        - 7.2 Nuclear Physics (Radioactivity, fission, fusion, nuclear structure)
        - 7.3 High Energy and Particle Physics (Standard Model, quarks, QCD, accelerators)
        - 7.4 Astrophysics and Cosmology (Stellar evolution, dark matter, universe expansion)

        8. Mathematical, Computational, and Applied Physics
        - 8.1 Mathematical Methods in Physics (Tensors, complex analysis in physics)
        - 8.2 Computational Physics and Simulations (Numerical methods, Monte Carlo)
        - 8.3 Experimental Physics and Metrology (Lab instruments, measurement errors, data analysis)

        Classify the following text into one primary category and one secondary category based on the taxonomy above. The text is:
        {question}
        """
        return prompt
