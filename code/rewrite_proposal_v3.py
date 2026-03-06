import re

filepath = "/home/pranav/Desktop/MTP-Final/SafetyAwareRoutingProposal.tex"
with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update abstract
text = text.replace("Formulated as a multi-objective optimization problem, the system balances delivery speed, traffic congestion, and collision risk. It also considers the incentive structures that influence rider behavior in the gig economy.",
"Operating as a true multi-objective Pareto optimization challenge, the system balances probabilistic collision risk against expected delivery time. Crucially, it replaces arbitrary hard deadlines with Soft Time Windows (STW) yielding exponential financial lateness penalties, and introduces hard constraints for cumulative rider behavioral stress and road-hierarchy protections to prevent shifting hazards to vulnerable neighborhoods.")

# 2. Update intro
text = text.replace("This model treats the delivery ecosystem as a multi-objective optimization problem that penalizes dangerous routes.",
"This model establishes a multi-objective Pareto optimization framework prioritizing deterministic pathfinding. Recognizing the inherent unit mismatches in naive summation models, we formulate safety mathematically as route survival probability rather than additive penalties. Furthermore, we expand the scope of risk data beyond historical crashes to incorporate Surrogate Safety Measures (SSM) and dynamic weather-state multipliers.")

# 3. Update formulations heavily
start_formulation = text.find(r"\section{Safety-Aware VRPTW Formulation}")
end_formulation = text.find(r"\section{Data Acquisition Pipeline}")

new_formulation = r"""\section{Safety-Aware VRPTW Formulation}

To elevate the framework to rigorous academic standards, we structure the SA-VRPTW utilizing Soft Time Windows (VRPTW-STW) and probabilistic risk modeling. We expressly avoid linear scalarization of mismatched units (time vs. crash indices), instead defining a true multi-objective Pareto optimization front.

\subsection{Network and Sets}

The urban delivery area is a directed graph $G = (V, E)$. The vertices $V = \{0\} \cup C$ include the central micro-fulfillment depot (node $0$) and customer locations ($C = \{1, 2, \dots, N\}$). Given the reality of q-commerce batching, a homogeneous fleet of riders $K$ is deployed.

\subsection{Parameters and Out-of-the-Box Multipliers}

Each customer $i \in C$ possesses a localized demand $q_i$. Crucially, vehicle capacity $Q$ is constrained severely (typically $Q \in \{1, 3\}$), matching the continuous-dispatch reality of q-commerce rather than traditional large-scale parcel delivery. 

Deliveries face highly elastic Soft Time Windows $[e_i, l_i]$. Arriving after $l_i$ does not mathematically invalidate the route but triggers an exponential financial penalty function $P_L(\tau_i^k)$, representing platform SLA compensation and rider behavioral anxiety.

Every road segment $(i,j)$ features:
\begin{itemize}
    \item $t_{ij} \in \mathbb{R}_{>0}$: The dynamic travel time.
    \item $r_{ij} \in [0,1]$: The probability of a localized crash. To counter under-reporting bias in police databases, this is a composite measure fusing historical iRAD data with Surrogate Safety Measures (SSM) such as lane width and infrastructure lighting.
    \item $W \in [1.0, 5.0]$: A dynamic weather-state multiplier. During severe monsoon conditions, baseline $r_{ij}$ probabilities are heavily amplified.
    \item $h_{ij} \in \{0, 1\}$: A binary Road Hierarchy index (1 for residential/quiet zones, 0 for arterials), utilized to prevent algorithms from offloading heavy systematic risk into vulnerable pedestrian neighborhoods.
\end{itemize}

\subsection{Pareto Objective Functions}

Instead of a generic weighted sum, the algorithm seeks to identify the Pareto frontier mapping two distinct, unit-consistent objectives:

\textbf{Objective 1 (Efficiency \& Soft Penalties):} Minimize total routing travel time combined with the exponential financial penalties incurred by SLA violations:
\begin{equation}
\min F_1 = \sum_{k \in K} \sum_{(i,j) \in E} t_{ij} x_{ij}^k + \sum_{k \in K} \sum_{i \in C} P_L(\tau_i^k)
\end{equation}

\textbf{Objective 2 (Probabilistic Route Survival):} Rather than summing arbitrary risk indices, we strictly maximize the absolute probability of accident-free route completion across the active fleet:
\begin{equation}
\max F_2 = \prod_{k \in K} \prod_{(i,j) \in E} \left( 1 - (W \cdot r_{ij}) x_{ij}^k \right)
\end{equation}

By solving this multi-objective framework utilizing NSGA-II or $\epsilon$-constraint methodologies, decision-makers are presented with an exact trade-off curve distinguishing fractional delivery delays from measurable fleet mortality.

\subsection{Behavioral and Hierarchy Constraints}

The foundational routing constraints (Flow Conservation, Capacity, and Sub-tour Elimination) remain standard. However, we introduce two highly novel cyber-physical constraints:

\noindent\textbf{Behavioral Rider Fatigue Threshold:}
Drawing upon behavioral logistics and Prospect Theory, riders suffer cognitive degradation when consistently exposed to highly stressful arterial corridors. We enforce a maximum cumulative stress budget $S_{max}$ for any single rider workflow:
\begin{equation}
\sum_{(i,j) \in E} f(c_{ij}, r_{ij}) x_{ij}^k \leq S_{max}, \quad \forall k \in K
\end{equation}
where $f(\cdot)$ is a non-linear function exponentially penalizing back-to-back traversal of dangerous, gridlocked intersections. 

\noindent\textbf{Vulnerable Neighborhood Protection:}
To prevent the algorithm from indiscriminately routing delivery traffic through school zones and quiet alleys to avoid major intersections, we strictly limit the operational usage of residential infrastructure $h_{ij}$:
\begin{equation}
\sum_{k \in K} \sum_{(i,j) \in E} h_{ij} x_{ij}^k \leq H_{cap}
\end{equation}
This ensures the multi-objective optimization does not inadvertently generate new pedestrian safety externalities.

"""

text = text[:start_formulation] + new_formulation + text[end_formulation:]

# 4. Modify ALNS to match
text = text.replace("During intensive iterations, \"destroy\" operators strategically strip specific customer deliveries off the current plan",
"This necessitates the design of highly customized architectural operators. For instance, our implementation deploys a custom 'High-Risk Route Destroy' operator. During intensive iterations, this operator strategically un-maps trajectories predominantly crossing high $r_{ij}$ edges or heavily penalizing vulnerable $h_{ij}$ hierarchies. Intelligent 'repair' operators subsequently")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)

