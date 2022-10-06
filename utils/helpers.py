import sys
import time
from typing import Optional

from opacus.validators import ModuleValidator

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def count_num_lines(file_path):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        # print(sum(bl.count("\n") for bl in blocks(f)))
        return sum(bl.count("\n") for bl in blocks(f))

def validate_model(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model is not compatible for DF with opacus. Please fix errors.")
        print(errors)
        sys.exit(0)
    else:
        print("Model is compatible for DP with opacus.")


def fix_and_validate(model):
    model = ModuleValidator.fix_and_validate(model)
    return model


class TimeCode:
    def __init__(self):
        self.start_time = time.time()

    def how_long_since_start(self, prefix: Optional[str] = None):
        time_end = time.time()
        final_time_seconds = round(time_end - self.start_time, 2)
        final_time_minutes = round(final_time_seconds / 60, 2)
        print_string = f"Time it took: {final_time_seconds} seconds, {final_time_minutes} minutes"
        if prefix:
            print_string = prefix + print_string
        print(print_string)
