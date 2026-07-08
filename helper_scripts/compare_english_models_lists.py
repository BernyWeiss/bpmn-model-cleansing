
def load_list_as_set(path: str) -> set:
    with open(path, 'r') as f:
        return set(f.read().splitlines())


def print_set_differences(original_set, my_set) -> None:
    print("In my set, but not original")
    print(my_set-original_set)

    print("In original set, but not mine")
    print(original_set-my_set)


original = load_list_as_set("../data/reproduction/english_models_original.txt")
mine = load_list_as_set("../data/reproduction/english_models_generated.txt")

print_set_differences(original, mine)

