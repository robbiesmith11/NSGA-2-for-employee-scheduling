import cProfile
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from numba import jit
from ortools.sat.python import cp_model


def GUI():
    """
    constraints: (using ORtools)
        -maxium shifts per employee a day <= 1
        -every employee meets min and max hours
        -schedule meets employees availability
        -schedule meets group demand per day

    Optimise: (using NSGA-2)
        -minimise hours to potentially their min
        -maximise skilled individuals on shift per day

    future improvements:
        
        -run time slowed down by mutation function (numba, vectorisation)
        -paraleisation of create initial population
        -references
        
    References:
        - https://www.youtube.com/watch?v=SL-u_7hIqjA - logic of the main steps i used for my nsga-2 and technique of elitism by combining start and end population and sorting for best half
        - https://ai.stackexchange.com/questions/23637/what-are-most-commons-methods-to-measure-improvement-rate-in-a-meta-heuristic - learning rate functions and termination
        - https://stackoverflow.com/questions/30062429/how-to-get-every-first-element-in-2-dimensional-list - used in visualise fronts function
        - https://youtu.be/dPdB7zyGttg?si=K34MGvCvLun_l9iq - stop streamlit annoyingly rerunning every time user selects new value from selectbox by using session state
    """
    # should stay at 1000 as we only have the demand for that amount (future improvement ML prediction for needed demand)
    workers = 1000
    # can customise to add additional hour lengths of shifts
    shifts = [0, 2, 6, 8, 12]
    population_size = st.number_input("population size", min_value=1)
    generation_size = st.number_input("generation size", min_value=1)
    mutation_rate = st.number_input("mutation rate", min_value=0.0, max_value=1.0)
    tournement_size = st.number_input("tournement size", min_value=2, max_value=5)
    # hour_weight = st.number_input("hours weight", min_value = 1, max_value = 5)
    # skill_weight = st.number_input("skill weight", min_value = 1, max_value = 5)

    # threshold to terminate algorithm and return population
    termination_threshold = st.number_input(
        "termination threshold", min_value=0.0, max_value=1.0
    )
    # threshold on what is classed as local optima (input value higher than termination threshold)
    break_local_optima_threshold = st.number_input(
        "break local optima threshold", min_value=0.0, max_value=5.0
    )

    if "finished_state" not in st.session_state:
        st.session_state.finished_state = False

    load = st.button("optimise")

    if load and not st.session_state.finished_state:
        st.session_state.population, st.session_state.df = main(
            workers,
            population_size,
            generation_size,
            mutation_rate,
            tournement_size,
            termination_threshold,
            break_local_optima_threshold,
            shifts
        )
    
        st.session_state.finished_state = True

    if st.session_state.finished_state:
        visualise_fronts(st.session_state.population)
        st.session_state.individual = pick_individual(st.session_state.population)
        if "individual" in st.session_state and st.session_state.finished_state:
            new_rota(st.session_state.individual, st.session_state.df)
            rerun = st.button("Rerun optimisation")
            if rerun: 
                st.session_state.finished_state = False
    
def main(
    workers,
    population_size,
    generation_size,
    mutation_rate,
    tournement_size,
    termination_threshold,
    break_local_optima_threshold,
    shifts,
):
    # import datasets
    df_avail = pd.read_csv(r"C:\Users\smith\Downloads\New_dataset.csv")
    df_demand = pd.read_csv(r"C:\Users\smith\Downloads\Scaled_Demand.csv")
    # cleaning data in columns before being used
    df_avail = df_avail.replace({"A": 1, "NW": 0, np.nan: 0})
    df_avail = df_avail.drop("Start time", axis=1)

    df_num_workers = df_avail.iloc[:workers]

    # creating a data strcuture to store relevant population data.
    # i started with python classes but because numpy runs faster using vectorisation and works with numba i opted to change

    type_population = np.dtype(
        [
            (
                "schedule",
                np.int32,
                (workers, 7),
            ),  # 7 representing days of the week for the schedule
            (
                "fitness",
                np.float64,
                (2),
            ),  # representing amount of objectives for the fitness
            ("front", np.int32),  # ranking of indiviudal
            (
                "crowding_distance",
                np.float64,
            ),  # 2nd use of ranking to decide on better inidivdual within front by variety of individual
            (
                "dominates",
                np.int32,
                (population_size * 2),
            ),  # used object type so that an np array of varying sizes can be added as each individual will have different amounts
            ("domination_count", np.int32),  # count of how many it is dominated by
        ]
    )

    demands = group_demand(df_demand)
    population = np.zeros(population_size, dtype=type_population)
    # create initial populaiton:
    initial_population = create_population(
        population_size, df_num_workers, workers, shifts, population, demands
    )
    # calcualte nsga-2 properties:
    non_dominated_sorting(initial_population)
    crowding_distance(initial_population)
    # visualise_fronts(initial_population)

    final_population = generations(
        generation_size,
        initial_population,
        tournement_size,
        mutation_rate,
        type_population,
        df_avail,
        df_demand,
        shifts,
        df_num_workers,
        termination_threshold,
        break_local_optima_threshold,
    )

    return final_population, df_num_workers


def generations(
    generation_size,
    population,
    tournement_size,
    mutation_rate,
    type_population,
    df,
    df_demand,
    shifts,
    df_num_workers,
    termination_threshold,
    break_local_optima_threshold,
):

    fitnesses = []

    # Loop through number of generations performing selection-crossover-mutation on population then calculating new fitness/performance values
    # then combining previous and current population to sort for best half as form of elitism by preserving some of the best while also having diversity
    # due to the crowding distance factor, population is then reset to zeroes before appending the best population_size half schedules to it

    for i in range(generation_size):
        # calculate improvement of generation
        improvement = calculate_improvement_rate(fitnesses, population)
        # finish loop if improvement is less than threshold
        if improvement < termination_threshold:
            return population
        # run the break local optima function if is below the break local optima threshold
        elif improvement < break_local_optima_threshold:
            print("local optima")
            mutated = break_local_optima(
                population, 
                df, 
                df_demand, 
                type_population, 
                shifts
            )
            mutated, population = reset_attributes(mutated, population, df_num_workers)
            population = create_next_population(mutated, population, type_population)
            print("Generation", i + 1)
        # run as normal if its not stuck in local optima or below termination threshold
        else:
            mutated = default_run(
                population,
                df,
                df_demand,
                type_population,
                shifts,
                tournement_size,
                mutation_rate,
            )
            mutated, population = reset_attributes(mutated, population, df_num_workers)
            population = create_next_population(mutated, population, type_population)
            print("Generation", i + 1)
    print(population)
    return population


def create_next_population(mutated, population, type_population):
    
    # combine the population and calculate new fronts and crowd distance in relation to the combined population
    combined_population = np.concatenate((population, mutated))
    non_dominated_sorting(combined_population)
    crowding_distance(combined_population)
    # get the fronts
    fronts = combined_population["front"]
    # get the distances
    distances = combined_population["crowding_distance"]
    # sort the population based on the fronts and reverse of the distances as higher = better
    sorted_population = combined_population[np.lexsort((-distances, fronts))]
    # use the best half of the combined population to form the next generation
    new_population = sorted_population[: len(population)]
    # reset population and use current schedules for the next population
    population = np.zeros(len(population), dtype=type_population)
    population = np.copy(new_population)
    return population


def break_local_optima(population, df, df_demand, type_population, shifts):
    children = crossover(population, 3, df, df_demand, type_population)
    mutated = mutation(children, 0.15, shifts, df, df_demand, type_population)
    return mutated


def default_run(
    population, df, df_demand, type_population, shifts, tournement_size, mutation_rate
):
    children = crossover(population, tournement_size, df, df_demand, type_population)
    mutated = mutation(children, mutation_rate, shifts, df, df_demand, type_population)
    return mutated


def calculate_improvement_rate(fitnesses, population):
    # calculate the rate of improvement from last 2 generations if it is lower than the threshold then
    # terminate algorithm so can not waste resources continuing the run time when improvement is marginal

    # average fitness of population
    current = np.mean([individual["fitness"] for individual in population])
    # add fitness to list of fitnesses to track improvement(python placeholder, find way to use numpy)
    fitnesses.append(current)
    # check if this is the first generation
    if len(fitnesses) > 1:
        # becuase the newest fitness was just added the previous will be index -2
        previous = fitnesses[-2]
        improvement = euclidean_distance(current, previous)
        return improvement
    else:
        improvement = np.inf
        return improvement


def mutation(population, mutation_rate, shifts, df, df_demand, type_population):

    # for mutation i think its important to randomise shift lengths to introduce more variety and also swap days that they have optionally off
    # but are still avaialble to work. this would allow more diversity within the population to breed and potentially creating future better offspring

    days = len(population[0]["schedule"][0])
    workers = len(population[0]["schedule"])
    mutated = np.zeros(len(population), dtype=type_population)
    attempts = 50
    index = 0
    for individual in range(len(population)):
        # if a number less than mutation rate is picked then mutate the chromosome
        if random.random() < mutation_rate:
            not_feasibility = True
            while not_feasibility:
                mutation_start = np.random.randint(0, workers - 2)
                mutation_end = np.random.randint(mutation_start + 1, workers)

                chromosome = np.copy(population[individual]["schedule"])
                section = np.copy(chromosome[mutation_start:mutation_end])
                shift_swap_mutation(section, df)
                shift_length_mutation(section, shifts)
                # print("tried")
                chromosome[mutation_start:mutation_end] = section
                chromosome_feasibility = feasibility_check(chromosome, df, df_demand)
                if chromosome_feasibility and index < len(mutated):
                    # if it is valid increment index and add the schedule to the children
                    mutated[index]["schedule"] = chromosome
                    not_feasibility = False
                    print("mutated")
                    index += 1
                else:
                    attempts -= 1
                    if attempts == 0:
                        mutated[index]["schedule"] = np.copy(
                            population[individual]["schedule"]
                        )
                        break
        else:
            mutated[index]["schedule"] = np.copy(population[individual]["schedule"])
            index += 1

    return mutated


def shift_swap_mutation(section, df):
    # Mutation to find days employee is available but it isnt working but is available and swap that day with a day they are already working
    # this will allow the algorithm to explore different combinations of skill workers per day

    # loop through each workers schedule
    for w in range(len(section)):
        # days the worker can work but has been given as off
        availability = df.loc[w, "Sunday":"Saturday"].values == 1
        free_days_indices = np.where(section[w] == 0)[0]
        free_days = free_days_indices[availability[free_days_indices]]
        # if there are any days then that can be swapped then..
        if len(free_days) > 0:
            # choose one of the free days
            free_day_to_swap = random.choice(free_days)
            # all days in schedule employee is currently working
            working_days = np.where(section[w] != 0)[0]
            if len(working_days) > 0:
                # random choice of working days to swap
                work_day_to_swap = random.choice(working_days)
                # swap elements in the schedule
                section[w, free_day_to_swap], section[w, work_day_to_swap] = (
                    section[w, work_day_to_swap],
                    section[w, free_day_to_swap],
                )

    return section


@jit(nopython=True)
def shift_length_mutation(section, shifts):
    # from the "shifts" array select a random shift from it to change n amount for each n days of each schedule

    for w in range(len(section)):
        # list of days the employee is working
        working_days = np.where(section[w] != 0)[0]
        if len(working_days) > 0:
            # randomly chose number of days to mutate with a range up to how many they are working
            num_days = np.random.randint(1, len(working_days))

            mutate_days = working_days[
                np.random.randint(0, len(working_days), size=num_days)
            ]
            for day in mutate_days:
                section[w][day] = shifts[np.random.randint(0, len(shifts))]

    return section


def crossover(population, size, df, df_demand, type_population):

    # Creating a new population type data structure "children" and filling it with the offspring created
    # by selecting parents through tournement selection of the current population and randomly selecting a crossover start/end point to replace parent1[start:end] with parent2[start:end]
    # maintaing worker availability feasibility while also allowing for alot of possible childs. A feasibility check is performed afterwards to ensure
    # it meets the previous hours and group demand constraints

    days = len(population[0]["schedule"][0])
    workers = len(population[0]["schedule"])
    children = np.zeros(len(population), dtype=type_population)
    attempts = 100
    index = 0

    # loop to ensure children is filled with right amount of childs
    while index < len(children):
        # selecting via tournement selection with a tournement size
        parent_one = selection(population, size)
        parent_two = selection(population, size)
        # loop to ensure parent1 doesnt equal parent2 , i ran into problems where initial populations werent diverse so decieded to base it on
        # fitness instead of id
        while np.array_equal(parent_one["fitness"], parent_two["fitness"]):
            parent_two = selection(population, size)

        # creating a start and end point for crossover
        # start being workers -2 allows there always to be a valid start and end
        crossover_start = np.random.randint(0, workers - 2)
        # end point will always be above the start to ensure it valid
        crossover_end = np.random.randint(crossover_start + 1, workers)

        # creating copies of the parents so the orignals wont change. i ran into memory leak problems using deepcopy and found this to work instead
        child_one = np.copy(parent_one["schedule"])
        child_two = np.copy(parent_two["schedule"])

        # performing crossover
        child_one[crossover_start:crossover_end] = parent_two["schedule"][
            crossover_start:crossover_end
        ]
        child_two[crossover_start:crossover_end] = parent_one["schedule"][
            crossover_start:crossover_end
        ]
        # checking feasibility, returns a boolean value
        child_one_feasible = feasibility_check(child_one, df, df_demand)
        child_two_feasible = feasibility_check(child_two, df, df_demand)

        # check if both children are valid seperately as one may be valid the other may not be
        if child_one_feasible and index < len(children):
            # if it is valid increment index and add the schedule to the children
            children[index]["schedule"] = child_one
            index += 1
        if child_two_feasible and index < len(children):
            # if it is valid increment index and add the schedule to the children
            children[index]["schedule"] = child_two
            index += 1

        # fail safe to break while loop incase cant find len(children) amount of feasible solutions
        else:
            attempts -= 1
            if attempts == 0:
                break
    return children


def feasibility_check(child, df, df_demand):

    # to check feasibility of an individual against constraints of min<= x <=max and also agaisnt the group demand for each day

    # check hours and group feasibility
    hours_feasbility = False
    demand_feasbility = False
    # creat arrays of the min and max hours of employees from df
    min_hours = df["Min_Hours"].values
    max_hours = df["Max_Hours"].values

    # create array of total hours of each employe from both children
    child_total_hours = np.sum(child, axis=1)

    # check the totals against min and max
    # each element of the child feasbilties are True or false np.all is used to check if all of them satisfy true
    child_feasiblity = np.all(
        (child_total_hours >= min_hours) & (child_total_hours <= max_hours)
    )

    # turn the hours to true as if its in constraint
    if child_feasiblity:
        hours_feasbility = True

    # demand check:
    days = len(child[0])
    groups = df["Group code"].nunique()
    group_mapping = {"a": 0, "b": 1, "c": 2, "d": 3}

    # stores the current group count scheduled for each day
    demand_current_matrix = np.zeros((days, groups))
    # stores whats needed for each day
    demand_needed_matrix = df_demand[
        ["Group_a", "Group_b", "Group_c", "Group_d"]
    ].to_numpy()

    # uses same logic as the total_skill function but adjusted without skill calculation
    for worker in range(len(child)):
        worker_group = df.loc[worker, "Group code"]
        for index, hours in enumerate(child[worker]):
            if hours != 0:
                demand_current_matrix[index, group_mapping[worker_group]] += 1
    # each element of the group feasbilties are True or false np.all is used to check if all of them satisfy true
    group_feasibility = np.all(demand_current_matrix >= demand_needed_matrix)
    # turn the demand to true if group meets the constraitns
    if group_feasibility:
        demand_feasbility = True

    return hours_feasbility and demand_feasbility


# Tournement Selection
def selection(population, size):

    # selction to choose 1 parent each time its called by performing tournement selection on a "size" amount of individuals
    # the individuals will compete with eachother and the highest front will win and the crowding distance will be the tie breaker to determine the
    # tournement winner (parent)

    # random choice between all the indces of population, and the size being the tournement size
    bracket = np.random.choice(len(population), size=size)
    # get list of the values to compare between the individuals
    fronts = population["front"][bracket]
    distances = population["crowding_distance"][bracket]
    # sort by front then distance as a deciding factor for the better individual
    sorted_bracket = bracket[np.lexsort((-distances, fronts))]
    # returns the winning parent
    return population[sorted_bracket[0]]


def calculate_fitness(individual, df):

    days = len(individual["schedule"][0])
    groups = df["Group code"].nunique()

    # fitness of minimise hours
    hours_goal = df["Min_Hours"].sum()  # sum of possible optimal minimium hours
    total_hours = individual["schedule"].sum()  # sum of current scheduled hours
    o1 = abs(
        total_hours - hours_goal
    )  # using the difference divided by 1 as better solutions will have lower hours difference

    # fitness of maximising skill per shift
    skills_goal_value = df[
        "Special Skill"
    ].sum()  # give the total number of skilled individuals overall
    skills_goal_matrix = np.full(
        (days, groups), skills_goal_value
    )  # 7 days and 4 groups.

    group_mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    group_np = df["Group code"].map(group_mapping).values
    skill_np = df["Special Skill"].values

    current_skill = total_skill(individual, group_np, skill_np, groups, days)
    o2 = np.sum(skills_goal_matrix) - np.sum(current_skill)

    return [o1, o2]


def create_population(
    population_size, df_num_workers, workers, shift_hours, population, demands
):

    for individual in range(population_size):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = random.randint(1, 100)
        length_shifts = len(
            shift_hours
        )  # used to iterate through shift_hours without calling len
        days = 7  # 7 days in a week which can be hardcoded as this won't change.

        shifts = {}
        for w in range(workers):
            for d in range(days):
                for s in range(length_shifts):
                    # creates a boolean dictionary for each combination of employees as a unique key
                    shifts[(w, d, s)] = model.NewBoolVar(f"shift_w{w}_d{d}_s{s}")
        # Number of shifts has to be max 1:
        for w in range(workers):
            for d in range(days):
                # add constraint to model
                model.Add(sum(shifts[(w, d, s)] for s in range(length_shifts)) <= 1)
        # Don't schedule if they aren't available:
        for w in range(workers):
            for d in range(days):
                # checks if the employee availability is off for the day
                if (
                    df_num_workers.iloc[w, d + 3] == 0
                ):  # d+3 is the column in the csv where availability starts
                    # starts at 1 to avoid 0 in the list which would be 0hrs
                    for s in range(1, length_shifts):
                        # add constraint that the solution is infeasible when employee cant work on that day
                        model.Add(shifts[(w, d, s)] == 0)
        # Hours constraint
        for w in range(workers):
            total_hours = 0
            for d in range(days):
                for s in range(1, length_shifts):
                    # sum of the total hours of the week by multiplying the boolean value 1/0 if its feasible or not by the hours
                    total_hours = shift_hours[s] * shifts[(w, d, s)] + total_hours
            # add constraint to meet the min and max hours
            model.Add(total_hours >= df_num_workers.loc[w, "Min_Hours"])
            model.Add(total_hours <= df_num_workers.loc[w, "Max_Hours"])
        # group constraint
        for d in range(days):
            for (
                group,
                demand,
            ) in demands.items():  # loops through each group in the demands dictionary
                # count for how many of each each group per day are in
                group_count = []
                for w in range(workers):
                    # if the current employee has the group that the loops on then add it to count
                    if df_num_workers.loc[w, "Group code"] == group:
                        group_count.append(w)
                total_group = 0
                # loop through each of the current groups total employees
                for w in group_count:
                    for s in range(1, length_shifts):
                        total_group = total_group + shifts[(w, d, s)]
                # add constraint of the current group scheduled has to be equal or more than the demand needed for the day
                model.Add(total_group >= demand[d])

        # creating individual from solver
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Feasible", individual)
            for w in range(workers):
                for d in range(days):
                    for s in range(1, length_shifts):
                        # if the boolean value is true then the solution is feasible so append it to population
                        if solver.Value(shifts[(w, d, s)]):
                            population[individual]["schedule"][w][d] = shift_hours[s]
                            break
        else:
            print("unfeasible")
        population[individual]["fitness"] = calculate_fitness(
            population[individual], df_num_workers
        )
    return population


@jit(nopython=True)
def non_dominated_sorting(population):

    # loop through population twice tobe able to compare each individual with eachother
    for i in range(len(population)):
        dominates_list = []
        for j in range(len(population)):
            # check if the current iteration is not itself
            if i != j:
                current = population[i]
                next = population[j]
                # check if current dominates next
                if dominates(current["fitness"], next["fitness"]):
                    # if it does dominate add next to list of indice that it dominates
                    dominates_list.append(j)
                    # increment nexts "dominated by" count as its been dominated by current
                    next["domination_count"] += 1
        population[i]["dominates"][: len(dominates_list)] = dominates_list
        population[i]["dominates"][len(dominates_list) :] = -1
    # using np.where to find where each of the first fronts are by them having a dominated by count of 0
    current_front_indexes = np.where(population["domination_count"] == 0)[0]
    # increments after each iteration
    front = 1
    while len(current_front_indexes) > 0:

        # loop through the indexes of the current front and apply the front value that its on
        for index in current_front_indexes:
            population[index]["front"] = front
            # changing domination count to -1 to mark it as processed
            population[index]["domination_count"] = -1
            # loop through the list of individuals the current dominates and if theyre not on current front then decrease domination count so they can be considered for next iteration of current front
            for dominated in population[index]["dominates"]:
                if population[dominated]["domination_count"] > 0:
                    population[dominated]["domination_count"] -= 1
        front += 1
        # calcualting the same variables again to update the current_front
        current_front_indexes = np.where(population["domination_count"] == 0)[0]

    return population


@jit(nopython=True)
def dominates(current, next):
    return np.all(current <= next) and np.any(current < next)


@jit(nopython=True)
def crowding_distance(population):

    objective_length = len(population[0]["fitness"])
    # loop through the number of objective values
    for o in range(objective_length):
        # get each fitness so that they can be sorted to find the best and worst values
        fitnesses = np.array([individual["fitness"][o] for individual in population])
        # sorts fitnesses returns a list of the indices of the fitnesses in sorted order
        sorted_by_fitness = np.argsort(fitnesses)

        # choose best and worst fitness to preserve variety within fronts
        population[sorted_by_fitness[0]]["crowding_distance"] = np.inf
        population[sorted_by_fitness[-1]]["crowding_distance"] = np.inf

        # if the max and min are the same then skip the calculation to the avoid the divsion by zero error which is more likely to happen when stuck in local optimas
        if (
            population[sorted_by_fitness[-1]]["fitness"][o]
            == population[sorted_by_fitness[0]]["fitness"][o]
        ):
            continue

        for i in range(len(population)):
            # check if current isnt equal to infite alread before calcuating. if it is then continue to next iteration
            if population[sorted_by_fitness[i]]["crowding_distance"] == np.inf:
                continue
            population[sorted_by_fitness[i]]["crowding_distance"] = population[
                sorted_by_fitness[i]
            ]["crowding_distance"] + (
                population[sorted_by_fitness[i + 1]]["fitness"][o]
                - population[sorted_by_fitness[i - 1]]["fitness"][o]
            ) / (
                population[sorted_by_fitness[-1]]["fitness"][o]
                - population[sorted_by_fitness[0]]["fitness"][o]
            )


@jit(nopython=True)
def total_skill(individual, group_np, skill_np, groups, days):
    # loop to check if employee is working on day and if they are then what is there group and skill

    # using a numpy matrix as to run faster and can further make code efficent by applying vectorisation if needed
    skill_count = np.zeros(
        (days, groups)
    )  # 7 representing the days and 4 representing the number of the groups this can be hardcoded as this wont change

    # calcualte deviation from skill goal to skilled workers to currently assigned skilled workers per days
    for worker in range(len(individual["schedule"])):
        worker_group = group_np[worker]  # group code is the 2nd column in df
        worker_skill = skill_np[worker]  # skill is the 3rd column in df

        # for each day of the schedule for the employee check if they have a shift (hours !=0 )and they have a skill before incrementing the index
        for index, hours in enumerate(individual["schedule"][worker]):
            if hours != 0 and worker_skill == 1:
                skill_count[index, worker_group] += 1

    return skill_count


def visualise_fronts(population):
    
    #get each fitness and front from the populations individuals
    fitnesses = []
    fronts = []
    for individual in population:
        if individual["front"] == 1:
            fitnesses.append(individual["fitness"])
            fronts.append(individual["front"])
    #using reference from stack overflow to get the x and y values so they can be plotted
    x = [i[0] for i in fitnesses]
    y = [i[1] for i in fitnesses]
 
    #plot fronts
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=fronts)
    plt.title("Pareto Fronts (lower = better)")
    plt.xlabel("Objective 1: Minimize Hours")
    plt.ylabel("Objective 2: Maximize Skilled Individuals" )
    plt.grid(True)
    st.pyplot(plt)

def pick_individual(population):

    #get all the fitnesses to display to user 
    fitnesses = []
    for individual in population:
        #has to be in string format to display
        fitnesses.append(str(individual["fitness"]))
    #select box for user to pick which one to use
    selected_fitness = st.selectbox("Pick preferred rota based on tradeoffs", options=fitnesses, index=0)
    for individual in population:
        #convert to string format to check
        if str(individual["fitness"]) == selected_fitness:
            return individual


def new_rota(individual, df):
    new_df = df.copy()
    for w in range(len(individual["schedule"])):
        new_df.loc[w, "Sunday":"Saturday"] = individual["schedule"][w]
    st.write(new_df)

def reset_attributes(mutated, population, df_num_workers):
    # calculate new fitness for the mutated population while fitness the other attributes
    for individual in range(len(mutated)):
        mutated[individual]["fitness"] = calculate_fitness(
            mutated[individual], df_num_workers
        )
        mutated[individual]["dominates"][:] = -1
        population[individual]["dominates"][:] = -1
        population[individual]["domination_count"] = 0
        population[individual]["front"] = 0
        population[individual]["crowding_distance"] = 0.0
    return mutated, population


def group_demand(df_demand):

    # demand of each ggroup put into a list format
    group_a = df_demand["Group_a"].values
    group_b = df_demand["Group_b"].values
    group_c = df_demand["Group_c"].values
    group_d = df_demand["Group_d"].values
    # dictionary is used as the values are easier to read from as the values in the employee df are the key values of the dictionary
    return {"a": group_a, "b": group_b, "c": group_c, "d": group_d}


def euclidean_distance(pointx, pointy):
    return np.sqrt(np.sum((pointx - pointy) ** 2))


if __name__ == "__main__":
    GUI()
