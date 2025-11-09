from collections import deque
from typing import List



def extract_names_from_model(
    model: 'BPMNModel',
    use_types: bool = False) -> 'BPMNModel':
    """
    TODO: Write docstring
    This method was adapted from the sapsam BpmnModelParser _get_elements_flat method.


    Returns the model with model.names or names_with_types set (depending on use_types).
    """

    model_dict = model.model_json
    names = list()
    names_with_types = list()

    stack = deque([model_dict])

    while len(stack) > 0:
        element = stack.pop()

        for c in element.get("childShapes", []):
            c["parent"] = element["resourceId"]
            stack.append(c)

        # don't parse the root element.
        if element["resourceId"] == model_dict["resourceId"]:
            continue

        name = element["properties"].get("name")
        if use_types:
            element_type = element["stencil"].get("id") if "stencil" in element else None
            names_with_types.append(f"{element_type}: {name}")
        else:
            names.append(name)

    if use_types:
        model.names_with_types = names_with_types
    else:
        model.names = names

    return model



