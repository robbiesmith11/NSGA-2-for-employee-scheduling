import pandas as pd 
import numpy as np
import random
from numba import jit
import cProfile
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
"""
constraints: using ORtools
-maxium shifts per employee a day <= 1
-every employee meets min and max hours
-schedule meets employees availability
-schedule meets group demand per day

Optimise: using NSGA-2
-minimise hours to potentially their min
-maximise skilled individuals in per day
-???

steps left: 

-improvements exponetial fitness
-numba
-learning rate
-(optional) - meta models to search for approx fitness values to speed up, visualisation
-coperative selection?
"""

def main():
    
    #import datasets
    df_avail = pd.read_csv(r"C:\Users\smith\Downloads\New_dataset.csv")
    df_demand = pd.read_csv(r"C:\Users\smith\Downloads\Scaled_Demand.csv")
    #cleaning data in columns before being used
    df_avail = df_avail.replace({"A": 1, "NW": 0 , np.nan: 0})
    df_avail = df_avail.drop("Start time", axis=1)

    #adjustable parameters
    workers = 1000
    population_size = 300
    generation_size = 50
    mutation_rate = 0.03
    tournement_size = 3
    elite_size = 3

    #can customise to add additional hour lengths of shifts
    shifts = [0,2,6,8,12]
    
    df_num_workers = df_avail.iloc[:workers]

    #creating a data strcuture to store relevant population data.
    #i started with python classes but because numpy runs faster using vectorisation and works with numba i opted to change

    type_population = np.dtype([
    ("schedule", np.int32, (workers, 7)),  #7 representing days of the week for the schedule
    ("fitness", np.float64, (2)), #representing amount of objectives for the fitness
    ("front", np.int32),  # ranking of indiviudal
    ("crowding_distance", np.float64), #2nd use of ranking to decide on better inidivdual within front by variety of individual
    ("dominates", object), #used object type so that an np array of varying sizes can be added as each individual will have different amounts
    ("domination_count", np.int32)  # count of how many it is dominated by 
    ])

    demands = group_demand(df_demand)
    population = np.zeros(population_size, dtype=type_population)
    #create initial populaiton:
    initial_population = create_population(population_size, df_num_workers, workers, shifts, population, demands)
    #calcualte nsga-2 properties:
    non_dominated_sorting(initial_population)
    crowding_distance(initial_population)
    
    final_population = generations(generation_size, initial_population, tournement_size, mutation_rate, elite_size, type_population, df_avail, df_demand, shifts, df_num_workers)

def generations(generation_size, population, tournement_size, mutation_rate, elite_size, type_population, df, df_demand, shifts, df_num_workers):

    for i in range(generation_size):

        children = crossover(population, tournement_size, df, df_demand, type_population) #parents are chosen in crossover
        mutated = mutation(children, mutation_rate, shifts, df, df_demand)

        #calculate new fitness for the mutated population
        for individual in range(len(mutated)):
            mutated[individual]["fitness"] = calculate_fitness(mutated[individual], df_num_workers)
            population[individual]["dominates"] = np.array([])

        #add population and mutated together then calculate the , front and sorting
        #new_population = np.concatenate((population, mutated), axis=0)
        #new_population["domination_count"] = 0
        #new_population["front"] = 0
        #new_population["crowding_distance"] = 0.0
        
        non_dominated_sorting(mutated)
        crowding_distance(mutated)

        fronts = mutated["front"]
        distances = mutated["crowding_distance"]
        sorted_population = mutated[np.lexsort((-distances, fronts))]
        
        #sort for best len(population) then population = sorted_new_population
        new_population = sorted_population[:len(mutated)]
        top_five = new_population[:5]
        print("generation", i)
        print(top_five["fitness"])
        print(top_five["front"])
        print(top_five["crowding_distance"])

        population = np.zeros(len(population), dtype=type_population)
        population["schedule"] = new_population["schedule"]
        
    return population

def break_local_optima(population):

    """if top 5 individuals are identical in fitness scores then increase mutation size and reduce tournement size
    """

def mutation(population, mutation_rate, shifts, df, df_demand):

    """As crossover changes schedules through days and hours worked on swapped days. I think its important to also include diversity
    in the different types of shifts. by introducing shift length mutation the population will be introduced to a varied amount of shift lengths
    from the inital creation"""

    for individual in population:
        #if a number less than mutation rate is picked then mutate the chromosome
        if random.random() < mutation_rate:
            not_feasible = True
            #loop through each worker to apply mutation to a day of their schedule
            for w in range(len(individual["schedule"])):
                #while loop to make sure solution is feasible 
                while not_feasible:
                    #copy of individual in the while loop so if the solution is not feasible it resets back to its orignal stopping a loop of infeasible schedules
                    mutated = individual.copy()
                    schedule = mutated["schedule"][w]
                    random_day = random.randrange(len(schedule))
                    random_shift = random.choice(shifts)
                    #loop to make sure not applying hours to a day they have off "0"
                    while schedule[random_day] == 0:
                        random_day = random.randrange(len(schedule))
                    #assign shift to that day in schedule
                    schedule[random_day] = random_shift
                    #check if solution is feasible still , if it is then change the individuals schedules to to the feasible mutated version
                    if feasibility_check(mutated, df, df_demand):
                        #break while loop
                        not_feasible = False
                        individual["schedule"] = mutated["schedule"]
    return population

def crossover(population, size, df, df_demand, type_population):

    days = len(population[0]["schedule"][0])
    workers = len(population[0]["schedule"])
    #hold children individuals
    children = np.zeros(len(population), dtype=type_population)
    counter = 0
    index = 0
    while index < len(children)-1:
            parent_one = selection(population, size)
            parent_two = selection(population, size)
            while id(parent_one) == id(parent_two):
                parent_two = selection(population, size)

            child_one = parent_one.copy()
            child_two = parent_two.copy()
            
            for w in range(workers):
                crossover_start = np.random.randint(0, days - 2)
                crossover_end = np.random.randint(crossover_start + 1, days)
                child_one["schedule"][w][crossover_start:crossover_end] = parent_two["schedule"][w][crossover_start:crossover_end] 
                child_two["schedule"][w][crossover_start:crossover_end] = parent_one["schedule"][w][crossover_start:crossover_end]
                

            #check feasibility seperately as 1 child may pass but the other might not 
            if feasibility_check(child_one, df, df_demand) and index < len(children) :
                children[index]["schedule"] = child_one["schedule"]
                #increment index for the next child
                index += 1
            if feasibility_check(child_two, df, df_demand) and index < len(children):
                children[index]["schedule"] = child_two["schedule"]
                #increment index for the next child
                index += 1
            else:
                #increment current number of attempts 
                counter +=1
    return children


def feasibility_check(child, df, df_demand):

    #check hours and group feasibility
    hours_feasbility = False
    demand_feasbility =  False
    #creat arrays of the min and max hours of employees from df 
    min_hours = df["Min_Hours"].values
    max_hours = df["Max_Hours"].values

    #create array of total hours of each employe from both children
    child_total_hours = np.sum(child["schedule"], axis=1)
    
    #check the totals against min and max
    #each element of the child feasbilties are True or false np.all is used to check if all of them satisfy true 
    child_feasiblity = np.all((child_total_hours >= min_hours) & (child_total_hours <= max_hours))

    #turn the hours to true as if its in constraint
    if child_feasiblity:
        hours_feasbility = True

    #demand check: 
    days = len(child["schedule"][0])
    groups = df["Group code"].nunique()
    group_mapping = {"a":0 , "b": 1 , "c":2, "d":3} 

    #stores the current group count scheduled for each day
    demand_current_matrix = np.zeros((days, groups))
    #stores whats needed for each day
    demand_needed_matrix = df_demand[["Group_a", "Group_b", "Group_c", "Group_d"]].to_numpy()

    #uses same logic as the total_skill function but adjusted without skill calculation
    for worker in range(len(child["schedule"])):
        worker_group = df.loc[worker, "Group code"]
        for index, hours in enumerate(child["schedule"][worker]):
            if hours != 0:
                demand_current_matrix[index, group_mapping[worker_group]] += 1
    #each element of the group feasbilties are True or false np.all is used to check if all of them satisfy true 
    group_feasibility = np.all(demand_current_matrix >= demand_needed_matrix)
    #turn the demand to true if group meets the constraitns
    if group_feasibility:
        demand_feasbility = True

    return hours_feasbility and demand_feasbility


#Tournement Selection
def selection(population, size): 

    #selction to choose 1 parent each time its called
        
    #random choice between all the indces of population, and the size being the tournement size
    bracket = np.random.choice(len(population), size=size)
    #get list of the values to compare between the individuals 
    fronts = population["front"][bracket]
    distances = population["crowding_distance"][bracket]
    #sort by front then distance as a deciding factor for the better individual
    sorted_bracket = bracket[np.lexsort((-distances, fronts))]
    #returns the winning parent
    return population[sorted_bracket[0]]

def calculate_fitness(individual, df):

    days = len(individual["schedule"][0])
    groups = df["Group code"].nunique()

    #fitness of minimise hours 
    hours_goal = df["Min_Hours"].sum() #sum of possible optimal minimium hours 
    total_hours = individual["schedule"].sum() #sum of current scheduled hours
    o1 = abs(total_hours - hours_goal)  # using the difference divided by 1 as better solutions will have lower hours difference

    #fitness of maximising skill per shift
    skills_goal_value = df["Special Skill"].sum() #give the total number of skilled individuals overall 
    skills_goal_matrix = np.full((days, groups), skills_goal_value) #7 days and 4 groups.
    
    current_skill = total_skill(individual, df, groups, days)
    o2 = np.sum(skills_goal_matrix) - np.sum(current_skill)
    
    #fitness of ???  (can add additional objectives if needed)
    
    return [o1,o2]

    
def create_population(population_size, df_num_workers, workers, shift_hours, population, demands):
    
    for individual in range(population_size):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = random.randint(1, 100)

        length_shifts = len(shift_hours) #used to iterate through shift_hours without calling len 
        days = 7  # 7 days in a week which can be hardcoded as this won't change.

        shifts = {}
        for w in range(workers):
            for d in range(days):
                for s in range(length_shifts):
                    #creates a boolean dictionary for each combination of employees as a unique key
                    shifts[(w, d, s)] = model.NewBoolVar(f"shift_w{w}_d{d}_s{s}")

        #Number of shifts has to be max 1:
        for w in range(workers):
            for d in range(days):
                #add constraint to model
                model.Add(sum(shifts[(w, d ,s)] for s in range(length_shifts)) <= 1)

        # Don't schedule if they aren't available:
        for w in range(workers):
            for d in range(days):
            #checks if the employee availability is off for the day 
                if df_num_workers.iloc[w, d+3] == 0:   # d+3 is the column in the csv where availability starts
                    #starts at 1 to avoid 0 in the list which would be 0hrs
                    for s in range(1, length_shifts):
                        #add constraint that the solution is infeasible when employee cant work on that day 
                        model.Add(shifts[(w, d, s)] == 0)

        #Hours constraint 
        for w in range(workers):
            total_hours = 0
            for d in range(days):
                for s in range(1, length_shifts):
                    #sum of the total hours of the week by multiplying the boolean value 1/0 if its feasible or not by the hours
                    total_hours = shift_hours[s] * shifts[(w,d,s)] + total_hours
            #add constraint to meet the min and max hours  
            model.Add(total_hours >= df_num_workers.loc[w, "Min_Hours"])
            model.Add(total_hours <= df_num_workers.loc[w, "Max_Hours"])

        #group constraint 
        for d in range(days):
            for group, demand in demands.items():  #loops through each group in the demands dictionary 
                #count for how many of each each group per day are in
                group_count = []
                for w in range(workers):
                    #if the current employee has the group that the loops on then add it to count
                    if df_num_workers.loc[w, "Group code"] == group:
                        group_count.append(w)
                total_group = 0
                #loop through each of the current groups total employees
                for w in group_count: 
                    for s in range(1, length_shifts):
                        total_group = total_group + shifts[(w, d, s)]
                #add constraint of the current group scheduled has to be equal or more than the demand needed for the day
                model.Add(total_group >= demand[d])
        
        #creating individual from solver
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Feasible", individual)
            for w in range(workers):
                for d in range(days):
                    for s in range(1, length_shifts):
                        #if the boolean value is true then the solution is feasible so append it to population
                        if solver.Value(shifts[(w, d, s)]):
                            population[individual]["schedule"][w][d] = shift_hours[s]
                            break
        else: 
            print("unfeasible")
        population[individual]["fitness"] = calculate_fitness(population[individual], df_num_workers)
    return population

def non_dominated_sorting(population):

    #loop through population twice tobe able to compare each individual with eachother 
    for i in range(len(population)):
        dominates_list = []
        for j in range(len(population)):
            #check if the current iteration is not itself
            if i != j:
                current = population[i]
                next = population[j]
                #check if current dominates next
                if dominates(current["fitness"] ,next["fitness"]):
                    #if it does dominate add next to list of indice that it dominates 
                    dominates_list.append(j)
                    #increment nexts "dominated by" count as its been dominated by current
                    next["domination_count"] += 1
        population[i]["dominates"] = np.array(dominates_list)

    #using np.where to find where each of the first fronts are by them having a dominated by count of 0
    current_front_indexes = np.where(population["domination_count"] == 0)[0]
    #increments after each iteration 
    front = 1
    while len(current_front_indexes) > 0:
        
        #loop through the indexes of the current front and apply the front value that its on
        for index in current_front_indexes:
            population[index]["front"] = front
            #changing domination count to -1 to mark it as processed
            population[index]["domination_count"] = -1
            #loop through the list of individuals the current dominates and if theyre not on current front then decrease domination count so they can be considered for next iteration of current front
            for dominated in population[index]["dominates"]:
                if population[dominated]["domination_count"] > 0:  
                    population[dominated]["domination_count"] -= 1
        front += 1
        #calcualting the same variables again to update the current_front
        current_front_indexes = np.where(population["domination_count"] == 0)[0]
        
    return population

def dominates(current, next):
   return np.all(current <= next) and np.any(current < next)

def crowding_distance(population):

    objective_length = len(population[0]["fitness"])
    #loop through the number of objective values
    for o in range(objective_length):
        #get each fitness so that they can be sorted to find the best and worst values
        fitnesses = np.array([individual["fitness"][o] for individual in population])
        #sorts fitnesses returns a list of the indices of the fitnesses in sorted order
        sorted_by_fitness = np.argsort(fitnesses)

        #choose best and worst fitness to preserve variety within fronts 
        population[sorted_by_fitness[0]]["crowding_distance"] = np.inf
        population[sorted_by_fitness[-1]]["crowding_distance"] = np.inf

        #if the max and min are the same then skip the calculation to the avoid the divsion by zero error which is more likely to happen when stuck in local optimas 
        if population[sorted_by_fitness[-1]]["fitness"][o] == population[sorted_by_fitness[0]]["fitness"][o]:
            continue

        for i in range(len(population)):
            #check if current isnt equal to infite alread before calcuating. if it is then continue to next iteration
            if population[sorted_by_fitness[i]]["crowding_distance"] == float("inf"):
                continue
            population[sorted_by_fitness[i]]["crowding_distance"] = population[sorted_by_fitness[i]]["crowding_distance"] + (population[sorted_by_fitness[i+1]]["fitness"][o] - population[sorted_by_fitness[i-1]]["fitness"][o]) \
                 / (population[sorted_by_fitness[-1]]["fitness"][o] - population[sorted_by_fitness[0]]["fitness"][o])
   
def total_skill(individual, df, groups, days):
    #loop to check if employee is working on day and if they are then what is there group and skill

    #using a numpy matrix as to run faster and can further make code efficent by applying vectorisation if needed 
    skill_count = np.zeros((days, groups)) # 7 representing the days and 4 representing the number of the groups this can be hardcoded as this wont change
    #used to map each group to a column in the skill count matrix 
    group_mapping = {"a":0 , "b": 1 , "c":2, "d":3} 
        
    # calcualte deviation from skill goal to skilled workers to currently assigned skilled workers per days
    for worker in range(len(individual["schedule"])):
        
        worker_group = df.loc[worker, "Group code"]
        worker_skill = df.loc[worker, "Special Skill"]
        
        #for each day of the schedule for the employee check if they have a shift (hours !=0 )and they have a skill before incrementing the index
        for index, hours in enumerate(individual["schedule"][worker]):
            if hours != 0 and worker_skill == 1:
                skill_count[index, group_mapping[worker_group]] += 1
            
    return skill_count

def group_demand(df_demand):

    #demand of each ggroup put into a list format 
    group_a = df_demand["Group_a"].tolist()
    group_b = df_demand["Group_b"].tolist()
    group_c = df_demand["Group_c"].tolist()
    group_d = df_demand["Group_d"].tolist()
    #dictionary is used as the values are easier to read from as the values in the employee df are the key values of the dictionary
    return {"a" : group_a, "b" : group_b, "c": group_c, "d": group_d}

if __name__ == "__main__":
    main()
    

