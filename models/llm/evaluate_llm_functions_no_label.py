import argparse
import json
from pathlib import Path

# Create a safe environment for the check_file functions
import parse_json_tools_no_label_constrain
from tqdm import tqdm


def get_parsing_function(query, query_functions, env):
    function_source = query_functions[query]
    try:
        exec(function_source, env)
    except SyntaxError as e:
        print(f"Function for query '{query}' contains invalid Python syntax: {e}")
        return None
    except Exception as e:
        print(f"Error while loading function for query '{query}': {e}")
        return None

    if "check_file" not in env or not callable(env["check_file"]):
        raise ValueError(
            "The provided source did not define a callable 'check_file' function"
        )

    return env["check_file"]


def evaluate_llm_functions(opt):

    parse_json_tools_no_label_constrain.JSON_FOLDER = Path(opt.input_json_folder)
    env = parse_json_tools_no_label_constrain.__dict__.copy()
    JSON_FOLDER = parse_json_tools_no_label_constrain.JSON_FOLDER
    json_files = [f for f in JSON_FOLDER.rglob("*.json")]

    with open(opt.input_query_functions, "r") as f:
        query_functions = json.load(f)

    # Initialize output dict with empty lists for each query
    out_queries_dict = {q: [] for q in query_functions}

    # Precompute parsing functions for all queries
    parsing_functions = {}
    for q in query_functions:
        parsing_functions[q] = get_parsing_function(q, query_functions, env)
        try:
            parsing_functions[q](next(Path(JSON_FOLDER).rglob("*.json")).stem)
        except Exception as e:
            print(f"Skiping evaluation for {q}")
            print(e)
            # print(query_functions[q])
            parsing_functions.pop(q)

    # Iterate over all files once, check all queries for each file
    for f in tqdm(json_files):
        for q in list(out_queries_dict.keys()):
            try:
                if q in parsing_functions and parsing_functions[q](f.stem):
                    out_queries_dict[q].append(f.stem)
            except Exception as e:
                print(f"Could not parse {f} for {q}")
                print(e)

    with open(opt.output_retrieval_json, "w") as f:
        json.dump(out_queries_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-IQF", "--input_query_functions")
    parser.add_argument("-IJ", "--input_json_folder")
    parser.add_argument("-OJ", "--output_retrieval_json")

    opt = parser.parse_args()
    evaluate_llm_functions(opt)
