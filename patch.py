def run_scalability_experiment(G_bengaluru):
    print("--- [EXPERIMENT 1] Rigorous Algorithmic Scalability ---")
    results = []
    
    for n in [10, 15, 20]: # Capping at 20 since MILP explicitly freezes past it
        print(f"    Benchmarking N={n} ...")
        T_mat, R_mat = generate_routing_matrices(G_bengaluru, n)
        
        # MILP VRPTW
        time_milp, val = solve_milp(n, T_mat, R_mat, 0.5, 0.5)
        results.append({"Size N": n, "Time (s)": time_milp, "Algorithm": "Exact VRPTW (MILP)"})
        
        time_ga, val = solve_metaheuristic_ga(n, T_mat, R_mat, 0.5, 0.5)
        results.append({"Size N": n, "Time (s)": time_ga, "Algorithm": "Metaheuristic (GA/ALNS)"})
        
        # Note: phase 3 will append the real DRL results here natively

    df = pd.DataFrame(results).dropna()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Size N", y="Time (s)", hue="Algorithm", marker="o", linewidth=2.5)
    plt.yscale("log")
    plt.title("Algorithmic Scalability (Run on Real Node Topologies)", pad=15, fontweight='bold')
    plt.ylabel("Execution Time (Seconds) - Log Scale")
    plt.xlabel("Number of Customer Orders (N)")
    plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='MILP Capacity Limit')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/1_scalability_plot.png", dpi=300)
    plt.close()
