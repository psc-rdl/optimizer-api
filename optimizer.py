from gurobipy import Model, GRB, quicksum
import numpy as np
from datetime import datetime
from datetime import timedelta

# Get current time
current_time = datetime.now()

# Print or use the stored value
print("Current Time:", current_time)

class FixedEvent:
    def __init__(self, title, start_time, end_time, description=""):
        self.title = title # string
        self.start_time = start_time # datetime object
        self.end_time = end_time # datetime object
        self.description = description # string

    def __repr__(self):
        return f"{self.title} at {self.start_time.strftime('%Y-%m-%d %H:%M')}"
    
    def get_time_domain_secs(self):
        my_horizon = self.end_time - datetime.now() # Initializes change in datetime as a timedelta object
        return my_horizon.total_seconds()
    
# Get current time
current_time = datetime.now()

# Print or use the stored value
print("Current Time:", current_time)
    
class Calendar: 
    def __init__(self, fixed_tasks, priorities=None):
        
        self.fixed_tasks = fixed_tasks # List of FixedEvent objects

        # ======== Time horizon stuff ============== #

        # if time_horizon is not None:
        #     self.time_horizon = time_horizon
        # else:
        #     # Get how many 5 block intervals from current time to maximum end time of tasks in fixed_tasks
        #     max_horizon = 0

        #     # Find maximum end time of tasks in this calendar
        #     for task in fixed_tasks:
        #         if task.get_time_domain_secs() > max_horizon: 
        #             max_horizon = task.get_time_domain_secs # In seconds
        
        
        if priorities is not None:
            self.priorities = priorities
        else:
            self.priorities = [1 for _ in range(len(fixed_tasks))]

    """
    Returns an array of times that contain tasks already
    """
    def get_tasks(self):
        return self.fixed_tasks # List of FixedEvent objects


    """
    Adds tasks to this calendar
    """
    def add_tasks(self, tasks_dict):
        for task_name, task_start_end in tasks_dict.items():
            self.task_domains.append([i for i in range(task_start_end[0], task_start_end[1] + 1)])
            
class PSC_Optimizer:
    """
    task_times: List of integers in minutes
    deadlines: List of datetime objects
    task_names: List of Strings
    current_calendar: Calendar Object
    time_horizon: datetime object
    priorities: List of Integers
    priority_weight: Float in (0, 1)
    """
    def __init__(self, task_times, deadlines, task_names, current_calendar, time_horizon=None, priorities=None, priority_weight=0.15, time_weight=0.2):
        self.zero_block = self.get_zero_block()
        self.task_times = [int((task_times[j].total_seconds() // 300)) for j in range(len(task_times))] # 
        self.deadlines = [int((deadlines[i] - self.zero_block).total_seconds() // 300) for i in range(len(deadlines))]  # Array (length num tasks)
        print(f"deadlines: {self.deadlines}")
        self.task_names = task_names
        self.num_tasks = len(task_times)
        self.current_calendar = current_calendar # NOTE: TIME BLOCKS MUST MATCH UP TO THIS OPTIMIZER'S TIME BLOCKS
        print(self.num_tasks)
        print(f"time_horizon: {time_horizon}")
        self.time_weight = time_weight

        print(f"len(self.task_times): {len(self.task_times)} \n len(self.deadlines): {len(self.deadlines)} \n num_tasks: {self.num_tasks}")
        if time_horizon is not None:
            self.max_time = self.dt_to_block(time_horizon)
        else:
            print(max(self.dt_to_block(deadline) for deadline in deadlines))
            self.max_time = max(deadline for deadline in self.deadlines)

        if priorities is not None:
            self.priorities = priorities
        else:
            self.priorities = [1 for _ in range(len(task_times))]

        self.priority_weight = priority_weight 
        
        self.fixed_domain = [0 for _ in range(self.max_time)]
        for task in current_calendar.get_tasks():
            # Compare their datetime with the zero_block datetime
            task_start = self.dt_to_block(task.start_time)
            task_end = self.dt_to_block(task.end_time)
            for i in range(task_start, task_end):
                self.fixed_domain[i] = 1

        
        

    def OptimizeCalendar(self):
        if len(self.task_times) != len(self.deadlines):
            raise ValueError("Mismatch in task_times and deadlines length")

        # Create a new model
        model = Model("PSC-MIP-V2")

        # Decision Variables
        task_to_block = model.addVars(self.max_time, self.num_tasks, vtype=GRB.BINARY, name="x")  # Task assigned to time block
        task_starting_time = model.addVars(self.max_time, self.num_tasks, vtype=GRB.BINARY, name="t")  # Task start indicator

        
        # Define dicts for task times and deadlines
        task_times = {j: self.task_times[j] for j in range(self.num_tasks)}  # All tasks take task_times[j] time blocks
        deadlines = {j: self.deadlines[j] for j in range(self.num_tasks)}  # Deadline is deadlines[j] for each block

        # Decision Variables
        task_to_block = model.addVars(self.max_time, self.num_tasks, vtype=GRB.BINARY, name="x")  # x[i, j] binary
        task_starting_time = model.addVars(self.max_time, self.num_tasks, vtype=GRB.BINARY, name="t")  # t[i, j] binary
        anti_anxiety = model.addVars(self.num_tasks, vtype=GRB.INTEGER, name="a")

        print(f"task_time 0: {task_times[0]}")

        #Constraint 0: Cannot schedule tasks in blocks that already have something scheduled
        unavailable_blocks = self.fixed_domain
        print(self.fixed_domain)
        for i in range(len(unavailable_blocks)):
            if unavailable_blocks[i] > 0:
                for j in range(self.num_tasks):
                        model.addConstr(task_to_block[i, j] == 0)

        # Constraint 1: Ensure each task is assigned exactly `task_times[j]` blocks in the time horizon
        for j in range(self.num_tasks):
            model.addConstr(
                sum(task_to_block[i, j] for i in range(self.max_time)) == self.task_times[j],
                f"Sum_x_{j}"
            )

        # Constraint 2: Each task must take place in the blocks between its starting time and its dealine
        for i in range(self.max_time):
            model.addConstr(i*task_starting_time[i, j] <= (self.deadlines[j] - self.task_times[j]), f"Bound_t_{i}_{j}")

        # Constraint 2.5: Tasks cannot start on the same block
        for i in range(self.max_time):
            model.addConstr(sum(task_starting_time[i, j] for j in range(self.num_tasks)) <= 1, f"One starting time per time block")

        # Constraint 3: At most 1 task per time block
        for i in range(self.max_time):
            model.addConstr(sum(task_to_block[i, j] for j in range(self.num_tasks)) <= 1, f"One task per time block")

        # Constraint 4: Each task has exactly one starting time
        for j in range(self.num_tasks):
            model.addConstr(
                sum(task_starting_time[i, j] for i in range(self.deadlines[j])) == 1,
                f"Unique_t_{j}"
            )

        # Constraint 5: Linking task_starting_time with task_to_block
        for j in range(self.num_tasks):
            latest_start = min(self.max_time - task_times[j], deadlines[j] - task_times[j])
            for i in range(latest_start + 1):  # Include latest valid start
                for k in range(task_times[j]):
                    model.addConstr(
                        task_to_block[i + k, j] >= task_starting_time[i, j],
                        f"Start_link_{i}_{j}_{k}"
                    )

        # Constraint 6: (Consecutivity) If a task starts at time t, then it occupies time blocks t to t + d[j] - 1
        model.addConstrs(
            task_to_block[t_prime, j] >= task_starting_time[t, j]
            for j in range(self.num_tasks)
            for t in range(self.max_time - self.deadlines[j] + 1)
            for t_prime in range(t, t + self.task_times[j])
        )
    

        # Set Objective function    
        model.setObjective(
            sum(sum((1+self.time_weight*i)*task_starting_time[i, j] * (1 - self.priority_weight * self.priorities[j]) 
            for j in range(self.num_tasks)) 
            for i in range(self.max_time)),
            GRB.MINIMIZE
        )
        
        
        # Solve the model
        model.optimize()


        task_to_times_array = np.zeros((self.max_time, self.num_tasks), dtype=int)

        for i in range(self.max_time):
            for j in range(self.num_tasks):
                task_to_times_array[i, j] = task_to_block[i, j].X

        start_times_array = np.zeros((self.max_time, self.num_tasks), dtype=int)

        for i in range(self.max_time):
            for j in range(self.num_tasks):
                start_times_array[i, j] = task_starting_time[i, j].X

        print(f"=======task_to_times: ======= \n {task_to_times_array}")
        print(f"=======start_times: ======= \n {start_times_array}")


        # Store results
        results_dict = {}


        if model.status == GRB.OPTIMAL:
            print("Optimal Solution Found:")
            for j in range(self.num_tasks):
                for i in range(self.max_time):
                    if task_starting_time[i, j].x == 1:  # Check if task starts here
                        start_time = i
                        end_time = i + self.task_times[j]
                        results_dict[self.task_names[j]] = (start_time, end_time)
                        print(f"Task {self.task_names[j]} scheduled from {start_time} to {end_time}")

        else:
            print("No optimal solution found.")

        return results_dict

    # Take in a datetime and spit out its block in this optimizer
    def dt_to_block(self, dt):
        print("self.zero_block: {self.zero_block} \n dt: {dt}")
    # Ensure both are datetime.datetime objects
        delta = dt - self.zero_block
        block_num = delta.total_seconds() // 300 
        return int(block_num)  # optional: round or floor depending on your use

    def get_zero_block(self):
        dt = datetime.now() 
        dt = dt.replace(second=0, microsecond=0)

        minutes_to_add = (5 - dt.minute % 5) ## Check this logic 
        if minutes_to_add == 0:
            minutes_to_add = 5

        return dt + timedelta(minutes=minutes_to_add)