"""OS-managed leases for overlapping controllers of expensive posterior fits."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


@contextmanager
def posterior_lease(directory, specification, capacity=3):
    """Serialize each specification and cap concurrent four-chain fits at three."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"specification_{specification}.lock").open("a+") as unique:
        try:
            fcntl.flock(unique, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for the existing fit of {specification}; no duplicate sampling.", flush=True)
            fcntl.flock(unique, fcntl.LOCK_EX)
        slot = None
        waiting_announced = False
        try:
            while slot is None:
                for index in range(capacity):
                    candidate = (directory / f"capacity_{index}.lock").open("a+")
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                        continue
                    slot = candidate
                    slot.seek(0)
                    slot.truncate()
                    slot.write(json.dumps({"pid":os.getpid(), "specification":specification}))
                    slot.flush()
                    break
                if slot is None:
                    if not waiting_announced:
                        print("Waiting for posterior capacity; three fits already active.", flush=True)
                        waiting_announced = True
                    time.sleep(1)
            yield
        finally:
            if slot is not None:
                fcntl.flock(slot, fcntl.LOCK_UN)
                slot.close()
            fcntl.flock(unique, fcntl.LOCK_UN)
