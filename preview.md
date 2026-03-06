Safety-aware vehicle routing for quick commerce delivery
Introduction

Quick commerce platforms deliver groceries and daily essentials within very short time limits, typically around ten minutes. Platforms such as Blinkit, Zepto, and Swiggy Instamart operate dense fleets of two-wheeler delivery partners dispatched from local dark stores. The routing algorithm used by the platform prioritizes short travel time to meet delivery promises.

However, the relentless pursuit of speed introduces profound externalities. The mathematical core of these operations is the Vehicle Routing Problem with Time Windows (VRPTW), established by foundational works such as Solomon (1987) and Desrochers et al. (1992). Recent research has evolved this into a multi-objective optimization challenge that integrates safety-awareness and real-time risk quantification (Hoseinzadeh et al., 2020). Advanced frameworks, such as the RADR framework (2026), utilize spatiotemporal learning to proactively avoid high-risk road segments.

Furthermore, the interaction between algorithmic routing and rider incentives introduces systemic safety risks. Studies by Sinchaisri et al. (2023) and Salmon et al. (2023) highlight how platform incentives and "income-targeting" behaviors can induce risk-taking, necessitating behavioral decision elements within the routing framework. Delivery partners may choose shorter but riskier routes or increase speed when deadlines approach.

For this reason, the delivery system is modeled as a Safety-Aware Vehicle Routing Problem with Time Windows (VRPTW) extended with behavioral decision elements. The objective is to integrate delivery speed, congestion, and road safety risk within one optimization framework.

The model treats quick commerce routing as a multi-objective problem that balances three quantities:

delivery time

congestion exposure

collision risk

System structure

The delivery ecosystem has two interacting decision layers.

Platform routing algorithm

Delivery partner route choice

The platform computes an optimal route using traffic and distance information. The delivery partner may accept or deviate from this route depending on perceived incentives and risk.

                 Customer Orders
                       │
                       ▼
                Dark Store Dispatch
                       │
                       ▼
             Platform Routing Algorithm
       (time, congestion, safety evaluation)
                       │
             Suggested Delivery Route
                       │
                       ▼
             Delivery Partner Decision
       (speed, incentives, local knowledge)
                       │
                       ▼
                 Final Route Taken
                       │
                       ▼
                   Delivery

The routing model therefore incorporates both infrastructure conditions and rider behavior.

Mathematical formulation

The urban road network is modeled as a directed graph.

G=(V,E)
G=(V,E)

where

V
V represents locations (dark store and customers)

E
E represents road segments.

Node set

V={0}∪C
V={0}∪C

0
0 denotes the dark store and 
C
C denotes customer locations.

Edge set

E={(i,j)∣i,j∈V,i≠j}
E={(i,j)∣i,j∈V,i

=j}
Parameters

For each road segment 
(i,j)
(i,j)

dij
d
ij
	​


distance of the road segment

tij
t
ij
	​


expected travel time

cij
c
ij
	​


congestion index

rij
r
ij
	​


collision risk probability derived from accident records

For each customer node 
i
i

[ei,li]
[e
i
	​

,l
i
	​

]

delivery time window

qi
q
i
	​


order size

Q
Q

vehicle capacity

Decision variables

Route selection

xijk={1	if rider k travels from i to j
0	otherwise
x
ij
k
	​

={
1
0
	​

if rider k travels from i to j
otherwise
	​


Arrival time

τik
τ
i
k
	​


Vehicle load

yik
y
i
k
	​

Constraints

Each customer is served once

∑k∑jxijk=1
k
∑
	​

j
∑
	​

x
ij
k
	​

=1

Flow conservation

∑ixijk−∑ixjik=0
i
∑
	​

x
ij
k
	​

−
i
∑
	​

x
ji
k
	​

=0

Vehicle capacity

yjk≥yik+qj−Q(1−xijk)
y
j
k
	​

≥y
i
k
	​

+q
j
	​

−Q(1−x
ij
k
	​

)

Time feasibility

τik+tij≤τjk
τ
i
k
	​

+t
ij
	​

≤τ
j
k
	​


Time window

ei≤τik≤li
e
i
	​

≤τ
i
k
	​

≤l
i
	​

Objective function

The routing objective minimizes delivery time while limiting congestion exposure and safety risk.

min⁡λ1∑tijxij+λ2∑rijxij+λ3∑cijxij
minλ
1
	​

∑t
ij
	​

x
ij
	​

+λ
2
	​

∑r
ij
	​

x
ij
	​

+λ
3
	​

∑c
ij
	​

x
ij
	​


where

tij
t
ij
	​

 represents travel time

rij
r
ij
	​

 represents crash probability

cij
c
ij
	​

 represents congestion level

The weights 
λ1,λ2,λ3
λ
1
	​

,λ
2
	​

,λ
3
	​

 define the trade-off between delivery speed and safety.

Implementation using real data

The model can be implemented using publicly available datasets.

Road network

OpenStreetMap road data provides

road geometry

intersection nodes

road length and type

This dataset is converted into the routing graph.

Accident risk data

The Integrated Road Accident Database (iRAD) developed by the Ministry of Road Transport and Highways provides

accident coordinates

collision severity

time of accident

Accident records are mapped to road segments to compute

rij
r
ij
	​


collision probability.

Traffic and congestion data

Traffic information can be obtained from

Google Maps traffic API

Mapbox traffic API

SUMO traffic simulation datasets

These provide

tij
t
ij
	​


expected travel time and

cij
c
ij
	​


congestion index.

Delivery location data

Customer nodes can be generated from

residential building coordinates

synthetic demand generation

anonymized delivery datasets

Data processing workflow
Input datasets
  - OpenStreetMap road network
  - iRAD accident database
  - Traffic API data

Step 1
Construct road graph G(V,E)

Step 2
Map accident coordinates to road edges
Compute crash probability r_ij

Step 3
Estimate travel time t_ij and congestion c_ij

Step 4
Generate delivery nodes and time windows

Step 5
Construct VRP instance

Step 6
Solve routing optimization
Example pseudocode
Load road_network from OpenStreetMap
Convert network to graph G(V,E)

Load accident_dataset
For each accident:
    map accident to nearest road edge
    increment accident count

For each edge:
    compute risk r_ij

Load traffic_data
estimate travel_time t_ij
estimate congestion c_ij

Generate delivery_nodes
assign time_windows

Construct VRP_instance(G, parameters)

Solve VRP using optimization algorithm
Return optimal routes
Solving techniques

The routing model belongs to the class of NP-hard combinatorial optimization problems. Several approaches can be used.

Mixed Integer Linear Programming

Exact optimization for small instances.

Meta-heuristics

Genetic Algorithm

Ant Colony Optimization

Suitable for medium-scale routing.

Deep Reinforcement Learning

Used for dynamic routing where traffic conditions change in real time.

For the thesis implementation, a meta-heuristic solver combined with traffic simulation is sufficient for evaluating safety-aware routing strategies.