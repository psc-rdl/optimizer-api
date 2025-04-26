from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
from optimizer import PSC_Optimizer
import numpy as np
from datetime import timedelta

app = FastAPI()

class Task(BaseModel):
    name: str
    duration: int  # in minutes
    deadline: datetime
    priority: int

@app.post("/optimize")
def optimize_schedule(tasks: List[Task], calendar_events: List[Task]):
    if not tasks:
        return {"schedule": []}

    # Convert tasks to input for optimizer
    task_times = [task.duration for task in tasks]
    task_deadlines = [task.deadline for task in tasks]
    task_priorities = [task.priority for task in tasks]
    task_names = [task.name for task in tasks]

    # Convert calendar events to block ranges (assuming 5-minute blocks from 8am-12am = 192 blocks)
    def dt_to_block(dt: datetime):
        base = dt.replace(hour=8, minute=0, second=0, microsecond=0)
        minutes_since_start = (dt - base).total_seconds() / 60
        return int(minutes_since_start // 5)

    fixed_events = []
    for event in calendar_events:
        start_block = dt_to_block(event.deadline)
        duration_blocks = event.duration // 5
        fixed_events.append((start_block - duration_blocks, start_block))

    # Run 
    calendar = Calendar(fixed_events)
    optimizer = PSC_Optimizer(task_times, task_deadlines, task_names, calendar)
    schedule = optimizer.optimize()

    # Build schedule response
    result = []
    for i, block in enumerate(schedule):
        if block is not None:
            start_time = datetime.combine(task_deadlines[i].date(), datetime.min.time()) + timedelta(minutes=block * 5)
            result.append({
                "name": tasks[i].name,
                "start_time": start_time.isoformat(),
                "duration": tasks[i].duration,
                "deadline": task_deadlines[i].isoformat()
            })

    return {"schedule": result}
