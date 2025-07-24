import argparse

# # 1. This example does nothing
'''
command:
    python argparse_demo.py --help
output:   
    usage: argparse_demo.py [-h]

    optional arguments:
    -h, --help  show this help message and exit   
'''

# # If you want to see the output of this example, uncomment the following lines:

parser = argparse.ArgumentParser()
parser.parse_args()



# # 2. This example tells the user how to use positional arguments
"""
command:
    python argparse_demo.py --help
output:   
    usage: argparse_demo.py [-h] echo

    positional arguments:
    echo        echo the string you use here

    optional arguments:
    -h, --help  show this help message and exit

command:
    python argparse_demo.py hello 
output:   
    hello
"""
# # If you want to see the output of this example, uncomment the following lines:

parser = argparse.ArgumentParser()
# # default type=string, if we want to echo an integer, we can use type=int
parser.add_argument("echo", help="echo the string you use here")
args = parser.parse_args()
print(args.echo)



# # 3. This example shows how to use optional arguments
"""
command:
    python argparse_demo.py --help
output:   
    usage: argparse_demo.py [-h] [--verbosity VERBOSITY]

    optional arguments:
    -h, --help            show this help message and exit
    --verbosity VERBOSITY
                            increase output verbosity

command:
    python argparse_demo.py --verbosity 1 
output:   
    verbosity turned on
    
command:
    python argparse_demo.py --verbosity 1 
output:   
    nothing
"""
# If you want to see the output of this example, uncomment the following lines:
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity")

# parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
# python argparse_demo.py --verbose
# verbosity turned on
# python argparse_demo.py --verbose 1
# usage: argparse_demo.py [-h] [--verbose]
# argparse_demo.py: error: unrecognized arguments: 1

args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")


