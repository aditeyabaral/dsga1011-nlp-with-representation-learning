import json
import collections
import argparse
import random
import numpy as np
import requests
import re


# api key for query. see https://docs.together.ai/docs/get-started
def your_api_key():
    YOUR_API_KEY = ""
    return YOUR_API_KEY


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    prefix = """You are an expert at mathematics. Answer the math questions by strictly following the 
    provided instructions. For each question, perform addition from the rightmost digit to the leftmost digit 
    showing all carryovers. Write down the final answer as 'Final Answer: {a}+{b} = <final answer>'.
    
    Question: What is 3+4 ?
    Answer: Starting from the right:= 3+4 = 7.
    Final Answer: 3+4 = 7.
    
    Question: What is 2345678+7852789 ?
    Answer: Starting from the right:= 8+9 = 17, carry 1; 7+8+1 = 16, carry 1; 6+7+1 = 14, carry 1; 5+2+1 = 8; 4+5 = 9; 3+8 = 11, carry 1; 2+7+1 = 10, carry 1.
    Final Answer: 2345678+7852789 = 10198467.
    
    Question: What is 1234567+1234567 ?
    Answer: Starting from the right:= 7+7 = 14, carry 1; 6+6+1 = 13, carry 1; 5+5+1 = 11, carry 1; 4+4+1 = 9; 3+3 = 6; 2+2 = 4; 1+1 = 2.
    Final Answer: 1234567+1234567 = 2469134.

    Question: What is """

    suffix = " ?\nAnswer: "

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys.
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        "max_tokens": 300,  # max_tokens must be >= 50 because we don't always have prior on output length
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.7,
        "repetition_penalty": 0.8,
        "stop": [],
    }

    return config


def your_pre_processing(s):
    return s


def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    try:
        final_answer = re.findall(r"Final Answer: (\d+)\+(\d+) = (\d+)", output_string)
        final_answer = int(final_answer[0][-1])
    except:
        final_answer = 0
    return final_answer
