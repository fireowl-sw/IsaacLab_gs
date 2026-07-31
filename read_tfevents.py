import os
import sys
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_scalars(log_dir):
    # Find all event files in log_dir
    event_files = sorted(glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print("No event files found.")
        return
    
    event_file = event_files[-1]
    print(f"📖 Reading latest event file: {os.path.basename(event_file)}\n")
    
    # Load accumulator
    acc = EventAccumulator(event_file)
    acc.Reload()
    
    # Get all scalar tags
    tags = acc.Tags()["scalars"]
    if not tags:
        print("No scalar tags found in the event file.")
        return
    
    # Print the latest value for each tag, grouped nicely
    print(f"==================================================")
    print(f"           LATEST RL TRAINING METRICS             ")
    print(f"==================================================")
    
    # Group tags by category
    categories = {}
    for tag in sorted(tags):
        parts = tag.split('/')
        category = parts[0] if len(parts) > 1 else 'General'
        if category not in categories:
            categories[category] = []
        categories[category].append(tag)
        
    for category, cat_tags in categories.items():
        print(f"\n📂 [{category}]")
        for tag in cat_tags:
            events = acc.Scalars(tag)
            if events:
                latest_event = events[-1]
                print(f"  - {tag.split('/')[-1]}: {latest_event.value:.6f} (Step: {latest_event.step})")
    
    print(f"==================================================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # Find latest run
        runs = sorted(glob("/workspace/IsaacLab_gs/logs/rsl_rl/walker_astron_flat/*"))
        # Exclude directories that are not valid run folders
        runs = [r for r in runs if os.path.isdir(r)]
        if not runs:
            print("No runs found.")
            sys.exit(1)
        log_dir = runs[-1]
        
    get_latest_scalars(log_dir)
