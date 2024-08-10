import numpy as np
import dimod
from collections import defaultdict
from itertools import product

from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.system import LeapHybridCQMSampler
import dwave.inspector

from pyomo.environ import *

from scipy import stats

def generate_qubo_with_completion_time_variable(num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, h):
    P = {}
    Pp = {}
    for j in J:
        P[(j, num_machines)] = 0
        Pp[(j, num_machines)] = 0
        m = num_machines
        while (j, m) in mp:
                mpp = mp[(j, m)]
                if mpp != 0:
                    P[(j, mpp)] = P[(j, m)] + p[(j,mpp)]
                    Pp[(j, mpp)] = Pp[(j, m)] + p[(j,m)]
                m = mpp
    
    sumE = {i: sum(value for key, value in E.items() if key[0] == i) for i in set(key[0] for key in E)}
    
    sum_E_per_j = {j: sum(E[j, m] for m in range(1, num_machines + 1)) for j in range(1, num_jobs + 1)}


    ## Create Variables for the problem
    # Generate all combinations of variables
    variables = list(product(range(1, num_jobs + 1), range(1, num_machines + 1), range(1, num_time_slots + 1)))
    
    # Initialize an empty QUBO dictionary
    Q = {}
    
    # Populate QUBO dictionary with random coefficients
    for (j1, m1, t1) in variables:
        for (j2, m2, t2) in variables:
            Q[((j1,m1,t1), (j2,m2,t2))] = 0
    Q[(), ()] = 0


    ## Objective function: W E/T Cost
    # Define the objective function
    constant_value = 0
    for j in J:
        constant_value = - d[j]*w[j]*h[0]
        Q[(), ()] += constant_value
        for t in range(d[j], num_time_slots + 1):
            var_name = (j, num_machines, t)
            # Linear term (w[j] + sumE[j]) * t * x_jmt
            Q[var_name, var_name] += ((w[j] + sumE[j]) * t)*h[0]
            
        for tp in range(1, num_time_slots + 1):
            for m in range(1, num_machines):
                var_name = (j, m, tp)
                # Quadratic term -E[j,m] * (tp + Pp[j,m]) * x_jmt
                Q[var_name, var_name] += (-E[j, m] * (tp + Pp[j, m]))*h[0]


    ## Constraint 1: No − overlap
    for m in range(1, num_machines):
        for j in J:
            for t in T:
                for jp in J:
                    if j != jp:
                        for tp in range(t, t + p[jp, m] - 1 + 1):
                            if tp <= num_time_slots:
                                var1 = (j, m, t)
                                var2 = (jp, m, tp)
                                Q[(var1, var2)] += 1*h[1]


    ## Constraint 2: Precedence
    for j in J:
        for m in range(1, num_machines):
            if (j, m) in mp:
                for t in T:
                    for tp in range(1, t + p[j, m] - 1 + 1):
                        if tp <= num_time_slots:
                            if mp[j, m] != 0:
                                var1 = (j, mp[j, m], t)
                                var2 = (j, m, tp)
                                Q[(var1, var2)] += 1*h[2]
    
    for j in J:
        m = num_machines
        for t in range(d[j], num_time_slots+1):
            for tp in range(1, t- 1 + 1):
                if mp[j, m] != 0:
                    var1 = (j, mp[j, m], t)
                    var2 = (j, m, tp)
                    Q[(var1, var2)] += 1*h[2]


    ## Constraint 3: Operation once
    for j in J:
        for m in range(1, num_machines):
            if p[j, m] > 0:
                constant_value = + h[3]
                Q[(), ()] += constant_value
                # Add quadratic terms
                for t in range(p[j, m], num_time_slots+1):
                    var1 = (j, m, t)
                    Q[(var1, var1)] -= 1*h[3]  # y_{j,m,t}^2
                    for tp in range(p[j, m], num_time_slots+1):
                        if t != tp:
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 2*h[3]  # y_{j,m,t} y_{j,m,tp}
                for t in range(1, p[j, m]):
                    for tp in range(1, p[j, m]):
                        var1 = (j, m, t)
                        var2 = (j, m, tp)
                        Q[(var1, var2)] += 1*h[3]  # y_{j,m,t} y_{j,m,tp}
    

    ## Constraint 4: Tardy − once
    for j in J:
        constant_value = + h[4]
        Q[(), ()] += constant_value
        tmp = 0
        for t in range(d[j], num_time_slots + 1):
            var = (j, num_machines, t)
            tmp += 1
            Q[(var, var)] += 1*h[4]
            for tp in range(d[j], num_time_slots + 1):
                if tp != t:
                    Q[(var, (j, num_machines, tp))] += 2*h[4]
            Q[(var, var)] -= 2*h[4]


    qubo = dict(Q)
    return qubo


def generate_qubo_with_start_time_variable(num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, h):
    P = {}
    Pp = {}
    for j in J:
        P[(j, num_machines)] = 0
        Pp[(j, num_machines)] = 0
        m = num_machines
        while (j, m) in mp:
                mpp = mp[(j, m)]
                if mpp != 0:
                    P[(j, mpp)] = P[(j, m)] + p[(j,mpp)]
                    Pp[(j, mpp)] = Pp[(j, m)] + p[(j,m)]
                m = mpp
    sumE = {i: sum(value for key, value in E.items() if key[0] == i) for i in set(key[0] for key in E)}
    sum_E_per_j = {j: sum(E[j, m] for m in range(1, num_machines + 1)) for j in range(1, num_jobs + 1)}
    
    variables = list(product(J, M, T))
    Q = {}
    
    for (j1, m1, t1) in variables:
        for (j2, m2, t2) in variables:
            Q[((j1,m1,t1), (j2,m2,t2))] = 0
    Q[(), ()] = 0

    ## Objective function: W E/T Cost
    # Define the objective function
    constant_value = 0
    for j in J:
        constant_value = - d[j]*w[j]*h[0]
        Q[(), ()] += constant_value
        for t in range(d[j], num_time_slots+1):
            var_name = (j, num_machines, t)
            Q[var_name, var_name] += ((w[j] + sumE[j]) * t)*h[0]
    
        for m in range(1, num_machines):
            for tp in T:
                var_name = (j, m, tp)
                Q[var_name, var_name] += (-E[j, m] * (tp + P[j, m]))*h[0]

    ## Constraint 1: No − overlap
    for m in range(1, num_machines):
        for j in J:
            for t in T:
                for jp in J:
                    if j != jp:
                        for tp in range(t, min(t + p[j, m], num_time_slots+1)):
                            var1 = (j, m, t)
                            var2 = (jp, m, tp)
                            Q[(var1, var2)] += 1*h[1]


    ## Constraint 2: Precedence
    for j in J:
        # for m in range(1, num_machines):
        for m in M:
            if (j, m) in mp:
                if mp[j, m] != 0:
                    for t in T:
                        for tp in range(0, min(t + p[j, mp[j, m]], num_time_slots+1)):
                            var1 = (j, mp[j, m], t)
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 1*h[2]


    ## Constraint 3: Operation once
    for j in J:
        for m in range(1, num_machines):
            if p[j, m] > 0:
                constant_value = + h[3]
                Q[(), ()] += constant_value
                # Add quadratic terms
                for t in T:
                    var1 = (j, m, t)
                    Q[(var1, var1)] -= 1*h[3]  # y_{j,m,t}^2
                    for tp in T:
                        if t!= tp:
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 2*h[3]  # y_{j,m,t} y_{j,m,tp}
    

    ## Constraint 4: Tardy − once
    for j in J:
        constant_value = + h[4]
        Q[(), ()] += constant_value
        tmp = 0
        for t in range(d[j], num_time_slots+1):
            var = (j, num_machines, t)
            tmp += 1
            Q[(var, var)] += 1*h[4]
            for tp in range(d[j], num_time_slots+1):
                if tp != t:
                    Q[(var, (j, num_machines, tp))] += 2*h[4]
            Q[(var, var)] -= 2*h[4]


    qubo = dict(Q)
    return qubo


def solve_qubo_by_DWave_software(Q, solver, num_reads):
    if solver == 'SA':
        status = solve_qubo_by_DWave_software_SA(Q, num_reads)
    elif solver == 'QA':
        status = solve_qubo_by_DWave_software_QA(Q, num_reads)
    else:
        status = solve_qubo_by_DWave_software_LeapHybrid(Q)
    return status


def solve_qubo_by_DWave_software_SA(Q, num_reads):
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    # Create a linear beta schedule with the desired number of sweeps
    beta_range = [0.1, 10.0]

    response = sampler.sample(bqm, num_reads=num_reads, beta_range=beta_range)
    return response

def solve_qubo_by_DWave_software_QA(qubo, num_reads):
    
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(qubo, num_reads=num_reads)
    return response


def solve_qubo_by_DWave_software_LeapHybrid(qubo):

    sampler = LeapHybridSampler(solver={'category': 'hybrid'})
    response = sampler.sample_qubo(qubo)
    return response


def objective_function(model):
    return sum(model.w[j] * model.delays[j] for j in model.J) + sum(model.es[j,m] * (model.delivery_time[j] - model.start_time[j,m]) for m in model.M for j in model.J)

def eq2_rule(model, j):
    return model.delays[j] >= model.start_time[j,model.f[j]] + model.t[j,model.f[j]] - model.d[j]

def eq3_rule(model, i, j, m, mp):
    if i == j and m !=mp:
        if model.finish_times[(i,m)] <= model.finish_times[(j,mp)]:
            return model.start_time[i,m] + model.t[i,m] <= model.start_time[j,mp]
    if m == mp and i!=j:
        if model.finish_times[(i,m)] <= model.finish_times[(j,mp)]:
            return model.start_time[i,m] + model.t[i,m] <= model.start_time[j,mp]
    return Constraint.Skip

def eq4_rule(model, j):
    return model.delivery_time[j] >= model.d[j]

def eq5_rule(model, j):
    return model.delivery_time[j] >= model.start_time[j,model.f[j]] + model.t[j,model.f[j]]


def solve_LP_after_QA(finish_times, num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, f):
    finish_times = {k: v for k, v in finish_times.items() if k[1] != num_machines}
    p = {k: v for k, v in p.items() if k[1] != num_machines}
    E = {k: v for k, v in E.items() if k[1] != num_machines} 
    
    model = ConcreteModel()
    model.J = Set(initialize=range(1, num_jobs+1))  # Set of jobs
    model.M = Set(initialize=range(1, num_machines))  # Set of machines
    
    model.w = Param(model.J, initialize=w)
    model.t = Param(model.J, model.M, initialize=p)
    model.d = Param(model.J, initialize=d)
    model.es = Param(model.J, model.M, initialize=E)
    model.f = Param(model.J, initialize=f)
    model.finish_times = Param(model.J, model.M, initialize=finish_times, default=0)
    
    model.start_time = Var(model.J, model.M, within=NonNegativeReals)
    model.delays = Var(model.J, within=NonNegativeReals)
    model.delivery_time = Var(model.J, within=NonNegativeReals)
    
    model.obj = Objective(rule=objective_function, sense=minimize)
    model.eq2 = Constraint(model.J, rule=eq2_rule)
    model.eq3 = Constraint(model.J, model.J, model.M, model.M, rule=eq3_rule)
    model.eq4 = Constraint(model.J, rule=eq4_rule)
    model.eq5 = Constraint(model.J, rule=eq5_rule)
    
    # model.pprint()
    
    solver = SolverFactory('cplex', solver_io='python')
    # solver = SolverFactory('glpk')
    results = solver.solve(model, tee=False)

    return model, results


def calculate_hypothesis_tests(data_for_hypo, alpha = 0.05):
    # Step 2: Identify the first two minimum values of the makespan before LP colum
    df_sorted = data_for_hypo.sort_values(by=['makespan before LP', 'WET cost after LP'])
    # print(df_sorted)
    df = df_sorted # ['makespan before LP'].nsmallest(2)
    # print(df)
    grouped = df.groupby('makespan before LP')
    is_not_done = True
    i = [0]
    ii = 1
    while is_not_done:
        print("hereeee")
        group_names = list(grouped.groups.keys())
        group_lowest = []
        for index in i:
            group = grouped.get_group(group_names[index])['WET cost after LP'].tolist()
            group_lowest.extend(group)
        if ii < len(group_names):
            group_next_to = grouped.get_group(group_names[ii])['WET cost after LP'].tolist()
        else:
            is_not_done = False
            return False, group_names[1]
    
        # Calculate for group_next_to
        n_next_to = len(group_next_to)
        mean_next_to = np.mean(group_next_to)
        std_dev_next_to = np.std(group_next_to, ddof=1)  # Sample standard deviation
        
        # Calculate for group_lowest
        n_lowest = len(group_lowest)
        mean_lowest = np.mean(group_lowest)
        std_dev_lowest = np.std(group_lowest, ddof=1)  # Sample standard deviation
        
        s_p = np.sqrt(((n_next_to - 1) * std_dev_next_to**2 + (n_lowest - 1) * std_dev_lowest**2) / (n_next_to + n_lowest - 2))
        
        t_statistic = (mean_next_to - mean_lowest) / (s_p * np.sqrt(1 / n_next_to + 1 / n_lowest))
        
        
        df = n_next_to + n_lowest - 2
        # One-sided critical t-value
        t_critical_one_sided = stats.t.ppf(1 - alpha, df)
    
        if t_statistic >= t_critical_one_sided:
            # print("Reject the null hypothesis (H_0): Mean_next_to > Mean_lowest.")
            is_not_done = False
            print("okay")
            return True, group_names[0]
        else:
            i.append(ii)
            ii = ii + 1










