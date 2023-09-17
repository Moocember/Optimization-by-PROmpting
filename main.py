
from Prompts import meta_prompt, scorer_prompt

import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import re

def create_chain_from_template(template, input_variables, temperature=0.5, callbacks=None, verbose=True):
    """
    Create a language model chain from a template.

    Args:
        template (str): The template for generating prompts.
        input_variables (list): List of input variables used in the template.
        temperature (float): The temperature parameter for the language model.
        callbacks (list): List of callback handlers.
        verbose (bool): Whether to print verbose output.

    Returns:
        LLMChain: The language model chain.
    """
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template
    )

    chain = LLMChain(
        llm=OpenAI(temperature=temperature),
        prompt=prompt,
        callbacks=callbacks if callbacks is not None else [],
        verbose=verbose,
    )

    return chain

def build_text_and_scores(performance_df):
    """
    Build a string containing text and scores from a DataFrame.

    Args:
        performance_df (pd.DataFrame): DataFrame containing text and scores.

    Returns:
        str: A formatted string containing text and scores.
    """
    return ''.join([f"text:\n{performance_df.iloc[i]['text']}\nscore:\n{performance_df.iloc[i]['score']}\n" for i in range(len(performance_df))])

def rank_instructions(performance_df, num_scores):
    """
    Rank instructions based on scores and limit the number of instructions.

    Args:
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        num_scores (int): Number of top instructions to keep.

    Returns:
        pd.DataFrame: A DataFrame containing the top-ranked instructions.
    """
    performance_df = performance_df.sort_values(by='score')
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def sample_exemplars(training_exemplar_df, num_exemplars=3):
    """
    Sample exemplars from a DataFrame.

    Args:
        training_exemplar_df (pd.DataFrame): DataFrame containing training exemplars.
        num_exemplars (int): Number of exemplars to sample.

    Returns:
        str: A string containing sampled exemplars.
    """
    exemplar_sample = training_exemplar_df.sample(num_exemplars)
    return ''.join([f"input:\nQ: {exemplar_sample.iloc[i]['question']}\nA: <INS>\noutput:\n{exemplar_sample.iloc[i]['raw_answer']}\n" for i in range(num_exemplars)])

def generate_prompts(optimizer_chain, texts_and_scores, exemplars, n_prompts=3):
    """
    Generate prompts using an optimizer chain.

    Args:
        optimizer_chain (LLMChain): Optimizer language model chain.
        texts_and_scores (str): Texts and scores string.
        exemplars (str): Exemplars string.
        n_prompts (int): Number of prompts to generate.

    Returns:
        list: A list of generated prompts.
    """
    return [optimizer_chain.predict(texts_and_scores=texts_and_scores, exemplars=exemplars).replace("[", "").replace("]", "") for i in range(n_prompts)]

def are_numbers_the_same(x, y):
    """
    Check if two strings contain the same numbers.

    Args:
        x (str): First string.
        y (str): Second string.

    Returns:
        bool: True if the strings contain the same numbers, False otherwise.
    """
    return ''.join(re.findall(r'\d+', x)) == ''.join(re.findall(r'\d+', y))

def score_prompts(scorer_chain, prompts, training_examples, performance_df):
    """
    Score prompts based on training examples.

    Args:
        scorer_chain (LLMChain): Scorer language model chain.
        prompts (list): List of prompts to score.
        training_examples (pd.DataFrame): DataFrame containing training examples.
        performance_df (pd.DataFrame): DataFrame containing text and scores.

    Returns:
        pd.DataFrame: Updated performance DataFrame.
    """
    for prompt in prompts:
        scores = []
        for index, example in training_examples.iterrows():
            question = example['question']
            answer = example['raw_answer']
            sample_answer = scorer_chain.predict(question=question, instruction=prompt)
            scores.append(are_numbers_the_same(answer, sample_answer))
        score = int(100 * sum(scores) / len(scores))
        performance_df = performance_df.append({'text': prompt, 'score': score}, ignore_index=True)
    return performance_df

def opro(optimizer_chain, scorer_chain, performance_df, training_exemplar_df, n_scores=20, n_exemplars=3, n_prompts=8, n_training_samples=10, max_iterations=3):
    """
    Optimize prompts using an optimizer chain and scorer chain.

    Args:
        optimizer_chain (LLMChain): Optimizer language model chain.
        scorer_chain (LLMChain): Scorer language model chain.
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        training_exemplar_df (pd.DataFrame): DataFrame containing training exemplars.
        n_scores (int): Number of top scores to consider.
        n_exemplars (int): Number of exemplars to sample.
        n_prompts (int): Number of prompts to generate.
        n_training_samples (int): Number of training samples to use.
        max_iterations (int): Maximum number of optimization iterations.

    Returns:
        pd.DataFrame: Updated performance DataFrame.
    """
    performance_df = rank_instructions(performance_df, n_scores)
    for _ in range(max_iterations):
        texts_and_scores = build_text_and_scores(performance_df)
        exemplars = sample_exemplars(training_exemplar_df, n_exemplars)
        prompts = generate_prompts(optimizer_chain, texts_and_scores, exemplars, n_prompts)
        training_examples = training_exemplar_df.sample(n_training_samples)
        performance_df = score_prompts(scorer_chain, prompts, training_examples, performance_df)
        performance_df = rank_instructions(performance_df, n_scores)
    return performance_df

if __name__ == '__main__':
    optimizer_chain = create_chain_from_template(meta_prompt,
                                                 ["texts_and_scores", "exemplars"],
                                                 temperature=1,
                                                 verbose=True)
    scorer_chain = create_chain_from_template(scorer_prompt,
                                              ["question", "instruction"],
                                              temperature=0,
                                              verbose=True)
    performance_df = pd.read_csv("data/performance.csv", index_col=0)

    training_exemplar_df = pd.read_csv("data/training_exemplars.csv", index_col=0)

    output = opro(optimizer_chain, scorer_chain, performance_df, training_exemplar_df, n_scores=20, n_exemplars=3, n_prompts=8, n_training_samples=10, max_iterations=5)
    print(output)
    output.to_csv("data/performance2.csv")
