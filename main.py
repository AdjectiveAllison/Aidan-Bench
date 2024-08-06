import numpy as np
from models import chat_with_model, embed, embed_octo
from prompts import questions, instructions, create_gen_prompt, create_judge_prompt, create_code_gen_prompt, create_code_judge_prompt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
from colorama import Fore, Style


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a language model.")
    parser.add_argument("model_name", type=str, help="Name of the model to benchmark")
    parser.add_argument("--single-threaded", action="store_true", help="Run in single-threaded mode")
    parser.add_argument("--benchmark-type", choices=["language", "code", "both"], default="both",
                        help="Select the type of benchmark to run")
    return parser.parse_args()

def get_novelty_score(new_answer: str, previous_answers: list):
    new_embedding = embed_octo(new_answer)

    # If there are no previous answers, return maximum novelty
    if not previous_answers:
        return 1.0

    previous_embeddings = [embed_octo(answer) for answer in previous_answers]

    similarities = [
        np.dot(new_embedding, prev_embedding) /
        (np.linalg.norm(new_embedding) * np.linalg.norm(prev_embedding))
        for prev_embedding in previous_embeddings
    ]

    max_similarity = max(similarities)
    novelty = 1 - max_similarity

    return novelty

def process_question(item, model_name, is_code_benchmark):
    start_time = time.time()
    print(Fore.RED + item + Style.RESET_ALL)
    previous_answers = []
    item_novelty = 0

    try:
        while True:
            if is_code_benchmark:
                gen_prompt = create_code_gen_prompt(item, previous_answers)
            else:
                gen_prompt = create_gen_prompt(item, previous_answers)

            try:
                new_answer = chat_with_model(prompt=gen_prompt, model=model_name)
            except Exception as e:
                print(Fore.RED + f"Error generating answer: {str(e)}" + Style.RESET_ALL)
                break

            if new_answer is None:
                print(Fore.RED + "Error: Received None as an answer." + Style.RESET_ALL)
                break

            if is_code_benchmark:
                judge_prompt = create_code_judge_prompt(item, new_answer)
            else:
                judge_prompt = create_judge_prompt(item, new_answer)

            judge = "openai/gpt-4o-mini"
            try:
                judge_response = chat_with_model(prompt=judge_prompt, model=judge)
            except Exception as e:
                print(Fore.RED + f"Error getting judge response: {str(e)}" + Style.RESET_ALL)
                break

            if judge_response is None:
                print(Fore.RED + "Error: Received None as judge response." + Style.RESET_ALL)
                break

            try:
                if is_code_benchmark:
                    score = int(judge_response.split("<score>")[1].split("</score>")[0])
                else:
                    score = int(judge_response.split("<coherence_score>")[1].split("</coherence_score>")[0])
            except (IndexError, ValueError) as e:
                print(Fore.RED + f"Error parsing judge response: {str(e)}" + Style.RESET_ALL)
                break

            if score <= 3:
                print(f"Output: {new_answer}")
                print(Fore.YELLOW + "Output is incoherent or incorrect. Moving to next item." + Style.RESET_ALL)
                break

            novelty_score = get_novelty_score(new_answer, previous_answers)

            if novelty_score < 0.1:
                print(f"Output: {new_answer}")
                print(Fore.YELLOW + "Output is redundant. Moving to next item." + Style.RESET_ALL)
                break

            print(f"New Answer:\n{new_answer}")
            print(Fore.GREEN + f"Score: {score}")
            print(f"Novelty Score: {novelty_score}" + Style.RESET_ALL)

            previous_answers.append(new_answer)
            item_novelty += novelty_score

    except Exception as e:
        print(Fore.RED + f"Unexpected error processing item: {str(e)}" + Style.RESET_ALL)

    time_taken = time.time() - start_time
    print(Fore.BLUE)
    print(f"Total novelty score for this item: {item_novelty}")
    print(f"Time taken: {time_taken} seconds")
    print(Style.RESET_ALL)

    return item_novelty

def benchmark_model_multithreaded(model_name, benchmark_type):
    novelty_score = 0
    print_lock = threading.Lock()

    items = []
    if benchmark_type in ["language", "both"]:
        items.extend([(q, False) for q in questions])
    if benchmark_type in ["code", "both"]:
        items.extend([(i, True) for i in instructions])

    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        future_to_item = {executor.submit(process_question, item, model_name, is_code): (item, is_code) for item, is_code in items}

        for future in as_completed(future_to_item):
            item, is_code = future_to_item[future]

            item_novelty = future.result()
            with print_lock:
                novelty_score += item_novelty

    print(Fore.YELLOW)
    print(f"Total novelty score across all items: {novelty_score}")
    print(Style.RESET_ALL)

    return novelty_score

def benchmark_model_sequential(model_name, benchmark_type):
    novelty_score = 0

    items = []
    if benchmark_type in ["language", "both"]:
        items.extend([(q, False) for q in questions])
    if benchmark_type in ["code", "both"]:
        items.extend([(i, True) for i in instructions])

    for item, is_code in items:
        item_novelty = process_question(item, model_name, is_code)
        novelty_score += item_novelty

    print(Fore.YELLOW)
    print(f"Total novelty score across all items: {novelty_score}")
    print(Style.RESET_ALL)

    return novelty_score

def benchmark_model(model_name, benchmark_type, multithreaded=False):
    if multithreaded:
        return benchmark_model_multithreaded(model_name, benchmark_type)
    else:
        return benchmark_model_sequential(model_name, benchmark_type)

if __name__ == "__main__":
    args = parse_arguments()
    benchmark_model(args.model_name, args.benchmark_type, multithreaded=not args.single_threaded)
