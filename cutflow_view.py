import pickle
import sys

def print_cutflow(filename):
    try:
        with open(filename, 'rb') as f:
            output = pickle.load(f)
            if 'cutflow' in output:
                print("Cutflow:")
                print(output['cutflow'])
            else:
                print("The file does not contain 'cutflow' data.")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
    else:
        filename = sys.argv[1]
        print_cutflow(filename)