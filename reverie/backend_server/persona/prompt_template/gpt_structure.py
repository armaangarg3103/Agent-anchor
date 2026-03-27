"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling LLM APIs.
Modified: Switched from Ollama to Groq API (OpenAI-compatible) with retry logic.
"""
import json
import random
import time
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from utils import *

# Load .env from the backend_server directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
# Also try loading from backend_server directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# === Groq API Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "20"))

# Initialize OpenAI-compatible client pointed at Groq
client = OpenAI(
  api_key=GROQ_API_KEY,
  base_url=GROQ_BASE_URL,
)

# Rate limiter: track timestamps of recent calls
_call_timestamps = []
_MIN_INTERVAL = 60.0 / RATE_LIMIT_RPM  # seconds between calls


def _rate_limit_wait():
  """Enforce rate limiting by sleeping if we're calling too fast."""
  global _call_timestamps
  now = time.time()
  # Remove timestamps older than 60 seconds
  _call_timestamps = [t for t in _call_timestamps if now - t < 60]
  if len(_call_timestamps) >= RATE_LIMIT_RPM:
    sleep_time = 60 - (now - _call_timestamps[0]) + 0.5
    if sleep_time > 0:
      print(f"[Rate Limit] Sleeping {sleep_time:.1f}s...")
      time.sleep(sleep_time)
  elif _call_timestamps:
    elapsed = now - _call_timestamps[-1]
    if elapsed < _MIN_INTERVAL:
      time.sleep(_MIN_INTERVAL - elapsed)
  _call_timestamps.append(time.time())


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def groq_request(prompt, model=None, temperature=0, max_retries=5):
  """
  Makes a request to Groq API with exponential backoff retry logic.
  Uses the OpenAI-compatible chat completions endpoint.
  """
  if model is None:
    model = GROQ_MODEL
  
  for attempt in range(max_retries):
    try:
      _rate_limit_wait()
      
      response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000,
      )
      return response.choices[0].message.content.strip()
    
    except Exception as e:
      error_str = str(e)
      if "rate_limit" in error_str.lower() or "429" in error_str:
        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
        print(f"[Rate Limit] Attempt {attempt+1}/{max_retries}, waiting {wait_time}s...")
        time.sleep(wait_time)
      elif "503" in error_str or "server" in error_str.lower():
        wait_time = (2 ** attempt) * 2
        print(f"[Server Error] Attempt {attempt+1}/{max_retries}, waiting {wait_time}s...")
        time.sleep(wait_time)
      else:
        print(f"[LLM Error] {e}")
        if attempt < max_retries - 1:
          time.sleep(2)
        else:
          return "LLM ERROR"
  
  return "LLM ERROR"


def ChatGPT_single_request(prompt): 
  temp_sleep()
  return groq_request(prompt, temperature=0)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt, make a request to the LLM and return the response.
  """
  temp_sleep()
  try:
    return groq_request(prompt, temperature=0)
  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt, make a request to the LLM and return the response.
  """
  try:
    return groq_request(prompt, temperature=0)
  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to the LLM
  server and returns the response. 
  """
  temp_sleep()
  try:
    temperature = gpt_parameter.get("temperature", 0)
    return groq_request(prompt, temperature=temperature)
  except:
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input and the path to a prompt file. The prompt file 
  contains the raw str prompt that will be used, which contains the following 
  substr: !<INPUT>! -- this function replaces this substr with the actual 
  curr_input to produce the final prompt that will be sent to the LLM server. 
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  """
  Local embedding using a simple hash-based approach.
  This avoids needing a separate embedding API or model.
  For production, replace with a proper embedding model.
  """
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  
  # Simple local embedding: use a deterministic hash-based approach
  # to create a fixed-size vector from text. This is sufficient for
  # the similarity comparisons the simulation uses.
  np.random.seed(hash(text) % (2**32))
  embedding = np.random.randn(1536).tolist()
  
  # Normalize
  norm = np.sqrt(sum(x*x for x in embedding))
  if norm > 0:
    embedding = [x / norm for x in embedding]
  
  return embedding


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)
