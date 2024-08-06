# Questions should be open-ended but demand concrete answers.
questions = [
    "Provide an explanation for Japan's Lost Decades.",
    "What is a cause of World War 1?",
    "Why might the United States government nationalize ASI development?",
    "How might you use a brick and a blanket?",
    "What architectural features might you include in a tasteful house?",
    "Provide coordinates for a point inside the unit circle (x^2 + y^2 < 1).",
    "What's one way to use oregano?",
    "How might we enable LLMs to spend more output tokens to get predictably better results?",
    "Propose a solution to Los Angeles traffic.",
    "What activities might I include at a party for firefighters?",
    "Why did Rome fall?",
    "How could we redesign schools to better prepare students for the 22nd century?",
    "Find a solution to the inequality 2x + 3y < 10 where x and y are positive real numbers.",
    "What might be an unexpected consequence of achieving nuclear fusion?",
    "Describe a plausible alien life form that doesn't rely on carbon-based biology.",
    "How could we modify the rules of chess to make it more exciting for spectators?",
    "What would be the implications of a universal basic income on society?",
    "Propose an alternative to democracy for governing a country.",
    "Provide a real number greater than π but less than 4.",
    "How might we terraform Venus instead of Mars, and why?",
    "Design a new sport that combines elements of three existing sports.",
    "What could be a novel use for blockchain technology outside of cryptocurrency?",
    "How might human evolution be affected by long-term space colonization?",
    "Invent a new musical instrument and describe how it would be played.",
    "What might be an unexpected solution to reducing plastic waste in oceans?",
]


def create_gen_prompt(question: str, previous_answers: list) -> str:
    prompt = (
        "Answer the following question:.\n"
        f"<question>{question}</question>\n"
        "Your response should be a single brief sentence.\n"
    )

    if len(previous_answers) > 0:

        previous_answers_str = "\n".join(
            [f"{i+1}. {answer}" for i, answer in enumerate(previous_answers)]
        )

        prompt += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            f"<previous_answers>\n{previous_answers_str}\n</previous_answers>"
        )

    return prompt


def create_judge_prompt(question: str, answer: str):
    prompt = f"""    Your task is to evaluate the coherence and plausibility of an answer to a given question. This involves assessing whether the answer makes sense and isn't nonsensical or implausible.

    Question: <question>{question}</question>
    Answer: <answer>{answer}</answer>

    Evaluation process:
    1. Understand the question: Analyze what the question is asking.
    2. Assess the answer: Determine if the answer is coherent and plausible.
    3. Check for nonsensical elements: Identify any aspects that are completely unrelated or absurd.

    Please think through each step carefully and show your reasoning:

    1. Question analysis:
    [Your brief analysis of the question here]

    2. Answer assessment:
    [Evaluate if the answer is coherent and plausible]

    3. Nonsensical check:
    [Identify any completely unrelated or absurd elements]

    Based on your analysis, provide a final Coherence and Plausibility Score on a scale of 1 - 10, where:
    1-3: Incoherent, implausible, or nonsensical
    4-6: Partially coherent and plausible, but with some issues
    7-8: Mostly coherent and plausible with minor issues
    9-10: Highly coherent and plausible

    Ensure that nonsensical or completely implausible answers receive very low scores (1-3).

    IMPORTANT: After your reasoning, you must provide your final Coherence and Plausibility Score as a single integer between 1 and 10, enclosed in <coherence_score></coherence_score> XML tags. For example:
    <coherence_score>7</coherence_score>

    Your response must end with this score in the specified format.
    """
    return prompt



instructions = [
    "Implement a function to generate the Fibonacci sequence with an unexpected twist.",
    "Create a sorting function that incorporates a unique, non-standard criterion.",
    "Design a function that encrypts text using an unconventional method.",
    "Develop a function to solve the Tower of Hanoi puzzle with an added constraint.",
    "Implement a search algorithm that includes an unusual optimization technique.",
    "Create a function that generates a unique pattern or sequence.",
    "Design a data compression function using an uncommon approach.",
    "Implement a function that solves mathematical equations in a non-traditional way.",
    "Create a function that generates procedural art or music.",
    "Develop a unique hashing function for strings.",
    "Implement a function that simulates a simple ecosystem with unexpected rules.",
    "Create a pathfinding algorithm with an unconventional heuristic.",
    "Design a function that performs language translation using an unusual method.",
    "Implement a function that generates poetry or prose algorithmically.",
    "Create a function that solves a classic puzzle (e.g., Sudoku, crossword) with a twist.",
    "Develop a function that predicts future values using an unconventional forecasting method.",
    "Implement a unique recommendation algorithm.",
    "Create a function that generates realistic-sounding but fake data.",
    "Design a voting or ranking system with an unexpected rule.",
    "Implement a function that solves optimization problems using an unusual approach.",
    "Create a function that generates fractal patterns with a unique variation.",
    "Develop a text summarization function using an unconventional technique.",
    "Implement a function that simulates physics with some 'impossible' rules.",
    "Create a function that generates music based on unexpected inputs.",
    "Design a function that solves graph theory problems with a unique constraint."
]


def create_code_gen_prompt(instruction: str, previous_solutions: list) -> str:
    prompt = (
        "Write code that satisfies the following instruction:\n"
        f"<instruction>{instruction}</instruction>\n"
        "Your response should be a single idiomatic function, no longer than 5 lines of code.\n"
        "You *ONLY* need to output the code, no explanations or comments.\n"
    )

    if len(previous_solutions) > 0:
        previous_solutions_str = "\n\n".join(
            [f"{i+1}: {solution}" for i, solution in enumerate(previous_solutions)]
        )

        prompt += (
            "IMPORTANT: Provide a solution that is as different from any previous solutions as possible. Novel, creative, and innovative solutions in any programming language are acceptable.\n"
            "Your previous solutions are inside of the <previous_solutions></previous_solutions> XML tags.\n"
            f"<previous_solutions>\n{previous_solutions_str}\n</previous_solutions>"
        )

    return prompt

def create_code_judge_prompt(instruction: str, solution: str) -> str:
    prompt = f"""    Your task is to evaluate the correctness, creativity, and efficiency of a coding solution to a given instruction. This involves assessing whether the solution is correct, innovative, and meets the requirements of the instruction.

    Instruction: <instruction>{instruction}</instruction>
    Solution: <solution>{solution}</solution>

    Evaluation process:
    1. Understand the instruction: Analyze what the instruction is asking.
    2. Assess the solution: Determine if the solution is correct, creative, efficient, and meets the requirements of the instruction.
    3. Check for innovation: Identify any novel or unexpected approaches in the solution.

    Please think through each step carefully and show your reasoning:

    1. Instruction analysis:
    [Your brief analysis of the instruction]

    2. Solution assessment:
    [Your assessment of the solution's correctness and efficiency]

    3. Innovation check:
    [Identify any creative or novel aspects of the solution]

    Based on your analysis, provide a final score on a scale of 1-10, where:
    1-3: Incorrect, inefficient, or lacks creativity
    4-6: Partially correct and somewhat creative, but with some issues
    7-8: Correct and creative, but with minor issues
    9-10: Highly correct, efficient, and innovative

    Ensure that incorrect or entirely uncreative answers receive very low scores (1-3).

    IMPORTANT: After your reasoning, you must provide your final score as a single integer between 1 and 10, enclosed in <score></score> XML tags. For example:
    <score>7</score>

    Your response must end with this score in the specified format.
    """
    return prompt
