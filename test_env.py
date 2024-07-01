## test_env.py

try:
    import numpy
    import sklearn
    # import skfuzzy
    import scipy
    import matplotlib
    import torch
    
    print("Env is ready!")

except ModuleNotFoundError as e:
    print(e)