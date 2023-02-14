#!/usr/bin/env python

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Get PRS thresholds',epilog='performs np.geomspace or np.linspace')
    parser.add_argument('--lower',type=float,help='lower bound')
    parser.add_argument('--upper',type=float,help='upper bound')
    parser.add_argument('--number',type=int,help='number of values evenly spaced')
    parser.add_argument('--log',help='evenly spaced on log 10 space',action='store_true')
    parser.add_argument('--lin',help='evenly spaced on lin space',action='store_true')
    parser.add_argument('--precision',help='number of value to print after the decimal point. Default is 3',type=int, const=3,nargs='?')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = main()
    if args.log:
        all_vals = np.geomspace(args.lower,args.upper,args.number)
    elif args.lin:
        all_vals = np.linspace(args.lower,args.upper,args.number)
    print(np.array2string(all_vals,max_line_width=100000000,precision=args.precision,separator=',').replace('[','').replace(']',''))
        
