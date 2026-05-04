# Speaker Script & Panel Q&A
## SA-VRPTW MTP Thesis Presentation
**Pranav Bhadane | IIT Kharagpur | 2026**

---

## HOW TO USE THIS DOCUMENT
- Each slide has a **script** (what to say, in plain conversational language)
  and **panel Q&A** (likely questions with honest, confident answers).
- Speak slowly, pause at the end of each bullet.
- The formulation slides (6–7) are where panels probe hardest —
  the Q&A section there is the most important.
- Total target time: ~20 minutes for 15 content slides (leave 5 min for Q&A).

---

## SLIDE 1 — Title
**Time: 30 seconds**

> "Good morning. My name is Pranav Bhadane, and today I will present my
> Master's thesis titled 'Safety-Aware Vehicle Routing for Quick-Commerce
> Delivery in Indian Cities.' This work was done under the supervision of
> Professor Subhajit Sidhanta in the Department of Industrial and Systems
> Engineering."

---

## SLIDE 2 — Introduction
**Time: 2 minutes**

> "Let me start with the setting. You have all seen apps like Blinkit and
> Zepto — they promise to deliver groceries to your door in ten minutes.
> To do this, they operate from small neighbourhood warehouses called dark
> stores, and they send two-wheeler riders through city streets to deliver
> orders.
>
> Now here is the problem. India had over four lakh road accidents and
> nearly one lakh seventy thousand fatalities in 2022 alone — this is from
> the Ministry of Road Transport and Highways official report. These riders
> are unprotected two-wheelers in very heavy traffic. They are under
> constant pressure to deliver fast.
>
> The standard vehicle routing software used in industry just tries to
> minimise travel time. It treats safety as something to worry about later,
> if at all. There is no mechanism to say 'this route is faster but more
> dangerous — find me a safer option.' That is the gap this thesis fills."

**Key point to land:** *Fastest ≠ safest. Industry software has no explicit safety knob.*

---

## SLIDE 3 — Literature Review
**Time: 1.5 minutes**

> "Existing literature on vehicle routing goes back to Solomon's 1987 work,
> which established the benchmark instances still used today. The soft
> time-window variant allows late arrivals at a penalty — that is what we
> use, but with a convex penalty instead of a linear one, because in
> q-commerce being ten seconds late costs very differently from being five
> minutes late.
>
> On the safety side, researchers like Hoseinzadeh have added crash risk to
> routing, but they fold it into the objective as a weighted term. The
> problem with that is you can make the risk disappear just by changing the
> weight — there is no hard guarantee. Salmon, Ehrgott, and others showed
> that epsilon-constraint methods are better for this.
>
> For the heuristic solvers, Prins gave us the giant-tour with Bellman-Ford
> split for genetic algorithms, and Ropke and Pisinger gave us Adaptive
> Large Neighbourhood Search — both of which we implement."

---

## SLIDE 4 — Research Gaps
**Time: 1 minute**

> "When I reviewed the literature, five gaps were clear.
> Nobody had combined soft time windows, crash safety constraints, and the
> specific q-commerce operating model — ten-minute promises, one to two
> items per rider, micro-batching — into one formulation.
> Risk calibration was typically a black box — you just pick a number from
> thin air.
> Solver comparisons often used different evaluation code across methods,
> which means you are not comparing the same objective.
> Residential streets — narrow lanes that cut through neighbourhoods — had
> no explicit protection mechanism.
> And nobody had benchmarked on real Indian dark-store coordinates;
> everyone used synthetic random depots.
> This thesis addresses all five."

---

## SLIDE 5 — Contributions
**Time: 1.5 minutes**

> "Let me summarise what we contribute. First, the formulation itself:
> crash exposure is a hard constraint, not a weighted term, so the safety
> guarantee is explicit. Second, the calibration: every number traces back
> to a public government document — no black box. Third, all three solvers
> share the exact same evaluation code, so the comparison is fair.
> Fourth — and this was a surprise — the genetic algorithm beats ALNS
> at large problem sizes and is three hundred and twenty-five times faster.
> Fifth, we use real Blinkit dark-store coordinates extracted from a
> publicly committed KML file. And sixth, the entire pipeline is
> reproducible — code, config, and data are all on GitHub under MIT licence."

---

## SLIDE 6 — Problem Formulation
**Time: 3 minutes** *(most important slide — take your time)*

> "Now let me explain the formulation in simple terms.
>
> We are trying to assign customers to riders and decide in what order each
> rider visits them. The objective — what we are minimising — is called F1.
> It has two parts: a penalty for arriving early before the customer's order
> is even ready, and a penalty for arriving late after the ten-minute
> promise. The late penalty is exponential — the longer you are late, the
> much worse it gets. This matches real SLA contracts better than a linear
> penalty.
>
> But here is the key design choice: crash risk is NOT in this objective.
> It is in a separate constraint. For every rider route, the total log-
> survival risk must stay below a threshold R-bar. In probability terms,
> this means the route survival probability must be at least e to the minus
> R-bar. The planner picks R-bar — they are choosing the safety level
> directly, not hoping a weight coefficient does the right thing.
>
> We also have two residential-street constraints. One caps how many
> residential-edge traversals any single rider can make per route — set to
> 100. The other caps the fleet total. These prevent the algorithm from
> pushing all the risky arterial traffic onto quiet neighbourhood lanes.
>
> On the right, you can see the super-graph quantities. For each pair of
> locations, we pre-compute the shortest-time path on the actual street
> graph, then record three numbers: T — travel time, R — log-survival risk,
> H — count of residential edges used. The optimiser works with these
> pre-computed numbers rather than the full street graph."

---

## SLIDE 7 — Data Pipeline & BASM
**Time: 2 minutes**

> "Let me walk through how we build an instance from raw data.
>
> We start by downloading the street network for a city from OpenStreetMap
> using the OSMnx library. We then inflate all the travel times using the
> Bureau of Public Roads formula — this gives us congested travel times
> rather than free-flow, which is more realistic for Indian peak hours.
>
> Then we attach crash probabilities to each road segment. This is the BASM
> — the Bengaluru Accident Surrogate Model. The formula clips a product of
> three terms: lambda, the base probability for that road class computed
> from the MoRTH 2022 government report; sigma, a severity multiplier that
> preserves the finding from Mohan et al. that arterial roads are more
> dangerous than local streets; and pi, a local proxy term shown in the
> equation on the slide.
>
> Pi combines three things available in OpenStreetMap: edge betweenness
> — a measure of how much through-traffic an edge carries — signal density,
> and crossing density. Betweenness gets 40 percent weight because it is
> the strongest correlate of traffic volume.
>
> Importantly, this chain uses only public data. We do not need access to
> any restricted government database.
>
> For depots, we take real Blinkit store locations from a publicly committed
> KML file, cluster them using k-means, and snap the centroids to the
> nearest road intersection. Customers are then sampled with residential
> density weighting — heavier sampling near residential streets."

---

## SLIDE 8 — Solver Architecture
**Time: 1 minute**

> "One of the core design principles of this project is that all three
> solvers — the exact MILP, the genetic algorithm, and the ALNS — consume
> the exact same input and produce output evaluated by the exact same code.
>
> On the left is the Instance: a typed Python dataclass containing depots,
> customers, pre-computed super-arcs, and all the problem parameters.
> All three solvers take this as input.
>
> On the right is the shared evaluation contract: two Python functions that
> every solver must pass through. The first computes F1. The second checks
> all constraints strictly. If the solution fails, the solver raises an
> error — no silent violations.
>
> This matters because in most VRP papers, different solvers compute
> the objective differently, so you cannot actually compare them.
> We eliminate that bias by design."

---

## SLIDE 9 — Algorithms
**Time: 1.5 minutes**

> "For the genetic algorithm: we represent each solution as a permutation
> of all customers across all depots. The Bellman-Ford decoder then finds
> the optimal way to split this long sequence into individual rider routes.
> It does this by solving a shortest-path problem — it tries every possible
> split point and picks the one that minimises total F1 plus any penalty.
> Critically, any split that violates a hard constraint gets a one-billion
> cost sentinel, so the decoder only ever picks a violating split if there
> is literally no other option.
>
> For ALNS: we start with a greedy solution, then repeatedly destroy part
> of it — remove some customers — and repair it by reinserting them in the
> best position. We have four different ways to destroy and three ways to
> repair. An adaptive roulette wheel learns which combinations work better
> and gives them higher probability. Simulated annealing acceptance lets us
> escape local optima."

---

## SLIDE 10 — Code Snapshot
**Time: 1 minute**

> "This is the actual core of the Bellman-Ford split. You can see the
> nested loop: for each possible start index i and end index j, we compute
> the cost of making customers i through j into a single route.
> If the route violates the residential cap, the R-bar risk budget, or the
> duration limit, we add one-billion to the cost. This is the hard-
> constraint sentinel: the optimiser will never choose such a route unless
> it has absolutely no alternative.
>
> The key takeaway is that this code is shared verbatim by the GA, the
> ALNS, and the MILP evaluator. This is what makes our solver comparison
> valid."

---

## SLIDE 11 — Experimental Setup
**Time: 1 minute**

> "We ran experiments across four primary cities — Bengaluru, Delhi,
> Gurugram, and Mumbai. Pune has partial data and is noted separately.
> Hyderabad is excluded because the Blinkit KML file had zero placemarks
> there after filtering.
>
> We swept customer counts from 20 to 200, with up to 10 random seeds per
> cell. The run was done on a remote server and produced 428 feasible result
> rows with zero constraint violations across all of them."

---

## SLIDE 12 — Results: F1 vs N
**Time: 1 minute**

> "These four plots show F1 — our service quality cost — against instance
> size for each city, with the shaded band showing the min-to-max range
> across seeds.
>
> Two things are immediately clear. First, at N equals 20 and 50, GA and
> ALNS are about the same — you cannot tell them apart. Second, at N equals
> 100 and 200, the blue GA line is consistently below the red ALNS line —
> GA finds better solutions.
>
> F1 grows roughly proportionally with N, which makes sense: more customers
> means more deliveries that may arrive late."

---

## SLIDE 13 — Results: Wall-Clock & Speedup
**Time: 1 minute**

> "The wall-clock time difference is dramatic. The GA takes 26 seconds at
> N equals 200. The ALNS takes over 8400 seconds — that is more than two
> hours — for the same instances. The speedup ratio reaches over 300 times
> in Bengaluru and Delhi.
>
> This is because the GA's Bellman-Ford decoder is O of n-squared and runs
> very efficiently in Python. ALNS has 5000 iterations, each of which
> requires re-evaluating multiple insertion candidates across all routes.
> At large N, those iterations become expensive."

---

## SLIDE 14 — Results: City Analysis
**Time: 1 minute**

> "The heatmap on the left shows mean F1 for each city and N combination.
> Darker colour means higher lateness cost — harder instances.
> You can see Bengaluru is lightest — lowest F1 — because it has 113
> Blinkit stores giving us dense depot coverage. Gurugram and Mumbai are
> darker at large N because they have fewer stores relative to how spread
> out demand is.
>
> The bar chart confirms that GA always outperforms ALNS at N equals 200
> in every city where we have data."

---

## SLIDE 15 — Limitations & Future Work
**Time: 1 minute**

> "I want to be honest about what this work does not do.
>
> The calibration is a prior, not a measurement. We have no access to
> city-level crash microdata, so our risk estimates are grounded in
> government aggregates and OSM structure, not observed crash locations.
>
> The super-graph design fixes which physical streets are driven on at
> instance-generation time. The optimiser then only chooses customer
> orderings. A fully safety-aware system would also choose which streets
> to use — that is a materially harder problem and a clear future direction.
>
> The Pareto sweep, which would show the full trade-off curve between
> safety and efficiency at varying R-bar values, is defined and ready to
> run but was not completed in the current time budget."

---

## SLIDE 16 — Conclusion
**Time: 45 seconds**

> "To summarise: we have a clean formulation where safety is an explicit
> constraint, a calibrated crash-risk model built entirely from public data,
> three solvers that all evaluate the same way, and an empirical finding
> that the GA is both faster and better than ALNS at large problem sizes.
> The whole thing is public and reproducible."

---

---

# PANEL Q&A — EXPECTED QUESTIONS

## On Formulation

**Q: Why not just add crash risk as a weighted term in the objective?**

> "Weighted-sum formulations collapse incompatible units — minutes of delay
> and expected crash probability — into one number. The weight is chosen by
> the modeller, not by the operator. If you change the weight slightly,
> the solution changes completely, and the operator cannot reason about
> what safety level they are getting. With our epsilon-constraint, the
> operator picks a number — say, I want each route to have at least 95
> percent survival probability — and the solver either finds that or
> reports infeasibility. The guarantee is transparent."

**Q: What is the physical meaning of R-bar?**

> "R-bar is the maximum cumulative log-survival risk allowed per route.
> If R-bar equals 0.05, then e to the minus 0.05 equals approximately
> 95.1 percent survival probability for that route. If R-bar equals 0.40,
> survival probability is about 67 percent. So by tightening R-bar the
> operator is directly choosing a minimum survival guarantee per trip."

**Q: Why use exponential soft-time-window penalty instead of linear?**

> "In a ten-minute delivery promise, being two minutes late has a very
> different operational consequence from being ten minutes late. Linear
> penalties treat these the same per unit. The exponential grows faster
> as lateness increases, which reflects the escalating cost: rider
> re-routing, customer churn, SLA refunds, and platform reputation damage
> all compound as delay grows."

**Q: What is the super-graph and why fix paths at generation time?**

> "The super-graph is a pre-processed complete graph over just the depots
> and customers. For each pair, we run Dijkstra once on the full street
> network and record the three aggregated quantities: travel time, log-
> survival risk, and residential edge count. The MILP and heuristics then
> work with just these aggregates rather than the millions of street edges.
> This reduces the problem size by orders of magnitude.
>
> The limitation is that the path between any two locations is fixed.
> We optimise orderings, not streets. A future extension would jointly
> optimise both, but that changes the variable structure fundamentally."

**Q: Why is H-cap-route set to 100? How was that value chosen?**

> "One hundred is a reasonable operational upper bound for Indian urban
> deliveries within a 35-minute route. Residential edges in our OSM graphs
> are typically 50 to 150 metres long, so 100 edges represents roughly
> 5 to 15 kilometres of residential street traversal per route — enough
> for practical routing while preventing systematic cut-through traffic.
> The value is a configuration parameter and can be changed; we report
> results under the default."

**Q: How do you handle the multi-depot structure?**

> "Each rider is assigned to exactly one home depot and can only leave from
> and return to that depot. We enforce three constraints: the rider leaves
> the home depot at most once, returns if and only if it left, and cannot
> touch any other depot. This is stated explicitly in Chapter 3 with four
> equations."

---

## On Calibration (BASM)

**Q: How reliable is the BASM calibration without crash microdata?**

> "It is a calibrated prior, and I am honest about that in the thesis.
> The absolute per-edge probabilities are estimates, not measurements.
> What the calibration does correctly is: preserve the relative ordering
> — arterials are riskier than collectors, which are riskier than local
> streets — and scale each class by public national aggregate rates.
> We cross-validate the resulting modelled annual events against
> city-level published totals and require them to be within 25 percent.
> For a routing system that needs a risk signal, this is far better than
> a synthetic placeholder."

**Q: What is the pi proxy term and why those specific weights?**

> "Pi redistributes risk within a road class based on local graph
> structure. The three components are edge betweenness — a proxy for
> through-traffic volume — signal density, and crossing density.
> Betweenness gets the highest weight, 0.40, because it is the strongest
> structural correlate of traffic load. Signal and crossing density share
> the remaining 0.30 each because they capture intersection complexity
> equally. The weights are transparent in the config file and can be
> changed by a practitioner with better local data."

**Q: Why not use iRAD or state-level crash databases?**

> "Access to the Integrated Road Accident Database microdata requires
> institutional approval that was not available within this project's
> timeframe. The contribution of BASM is precisely that it produces a
> defensible calibration without requiring restricted data, making the
> framework usable by any researcher with public access. If microdata
> becomes available, BASM can be upgraded by replacing the lambda values
> — no code changes needed."

---

## On Solvers

**Q: Why does GA outperform ALNS at large N? Is that expected?**

> "It was not what I expected initially. The reason is that the
> Bellman-Ford split decoder makes the GA extremely effective for this
> particular problem structure. At each fitness evaluation, BF finds the
> optimal partition of the chromosome into routes — it pre-solves the
> route-assignment sub-problem exactly. ALNS, by contrast, uses destroy
> and repair operators that must simultaneously re-route and re-order
> customers, and with 5000 iterations fixed regardless of N, the per-
> iteration budget becomes insufficient as N grows.
>
> This finding suggests that for SA-VRPTW specifically, the Prins-style
> giant-tour representation is a strong fit because the BF decoder
> efficiently handles the hard constraints."

**Q: Why 200 generations for GA? How sensitive is that?**

> "200 generations was chosen based on pilot runs where convergence was
> observed within 150 to 180 generations for the tested instance sizes.
> It is a configuration parameter. Increasing it improves solution quality
> at the cost of wall time; with our BF decoder running in 26 seconds at
> N equals 200, there is room to double the generations if needed."

**Q: ALNS ran for over two hours at N=200 — is that practical?**

> "Not for real-time dispatch, no. The ALNS as implemented is configured
> for research-quality benchmarking rather than operational use. For a
> production system at N equals 200, the GA at 26 seconds is the clear
> choice. ALNS could be made faster by reducing the iteration count or
> using early termination on convergence."

**Q: Did you compare against any baseline from the literature?**

> "The default-epsilon benchmark — with no constraints active — gives us
> the unconstrained F1 for each solver. The MILP at small N provides an
> optimality gap reference. However, there is no directly comparable
> published result for Indian q-commerce instances with this formulation
> because the problem setting itself is novel. The contribution is the
> framework and the empirical finding about GA vs ALNS, not a state-of-
> the-art comparison on a shared benchmark."

---

## On Experiments

**Q: Why only 2 seeds for some cities? Is the data sufficient?**

> "The experiment was time-constrained by the remote server's capacity.
> Bengaluru, which has the most dark stores and is the primary city,
> has up to 10 seeds. For cities with fewer Blinkit stores — Gurugram
> and Mumbai — the smaller geographic extent means fewer distinct
> customer realisations, so 2 seeds already show consistent patterns.
> The directional finding — GA outperforms ALNS at large N — is
> consistent across every city with more than one seed."

**Q: Why is Pune excluded from the main analysis?**

> "Pune only has N equals 20 data with one seed per solver in the current
> run. That is insufficient to draw conclusions about scalability or
> solver comparison. We report it for completeness but correctly do not
> include it in the main figures."

**Q: How does the pipeline handle the case where no feasible solution exists?**

> "The validator raises an InfeasibleError, which is caught by the grid
> runner and recorded in the manifest with an error field. The manifest
> is then filtered to keep only feasible rows. The customer-feasibility
> filter in the instance generator was added specifically to prevent
> the common failure mode where a customer's round-trip distance alone
> exceeds the time budget — guaranteeing individual feasibility before
> the solver even starts."

---

## General / Broad Questions

**Q: What is the practical implication for a company like Blinkit?**

> "Today, Blinkit's dispatch algorithm optimises for delivery time. Our
> framework adds two explicit dials: an R-bar slider that says 'guarantee
> this minimum route survival probability,' and an H-bar slider that says
> 'do not penetrate residential streets more than this fleet-wide limit.'
> A planner can tune these to reflect the company's risk tolerance and
> community commitments, and immediately see the F1 cost of doing so.
> That is operationally actionable information that no current system
> provides."

**Q: Is the code publicly available?**

> "Yes. The full codebase, Hydra configuration files, calibration notes,
> and the committed Blinkit KML snapshot are all on GitHub at
> github.com/Coderopp/MTP-II under the MIT licence."

**Q: What would you do differently if you had more time?**

> "Three things. First, run the full Pareto sweep — the code is ready,
> I just need the compute time. Second, attempt to get access to
> city-level crash microdata to upgrade BASM from a calibrated prior
> to a validated measurement. Third, lift the super-graph path-fixing
> assumption and allow the optimiser to choose physical streets — that
> is the most important modelling improvement for future work."
