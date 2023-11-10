import dhg 
import numpy as np
import matplotlib.pyplot as plt 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic hypergraph data')
    parser.add_argument('--N', type=int, help = 'number of graphs to generate')
    parser.add_argument('--type', type=str, options=['HSBM', ''])
    parser.add_argument

