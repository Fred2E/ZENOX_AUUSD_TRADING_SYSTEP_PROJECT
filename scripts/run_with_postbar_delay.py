# run_with_postbar_delay.py

import time
import os  # <-- FIXED: Import os so you can use os.system
from datetime import datetime

POST_BAR_DELAY = 15  # seconds (10-30s is industry norm, you can adjust)

def wait_for_next_bar():
    now = datetime.utcnow()
    # Calculate seconds to next (minute % 5 == 0) plus delay
    mins = now.minute
    secs = now.second
    next_mod5 = ((mins // 5) + 1) * 5
    next_time = now.replace(minute=next_mod5 % 60, second=0, microsecond=0)
    if next_mod5 >= 60:
        # Next hour
        next_time = next_time.replace(hour=(now.hour + 1) % 24, minute=0)
    # Add the post-bar delay
    next_time = next_time.replace(second=POST_BAR_DELAY)
    wait_sec = (next_time - now).total_seconds()
    if wait_sec < 0:
        wait_sec += 3600  # Correct for edge case near hour boundary
    print(f"Waiting {wait_sec:.1f} seconds for post-bar time ({next_time.strftime('%H:%M:%S')})...")
    time.sleep(wait_sec)

if __name__ == "__main__":
    wait_for_next_bar()
    os.system("python zeno_master_pipeline.py")
