# Python skripts, lai failā saglabātu nejaušu Game of Life režģi

import random
import string
import argparse

def generate_write_grid(width, height, output_file):
    with open(output_file, "w", buffering=1024*1024*1024) as f:
        write_f = f.write
        randbit_f = random.getrandbits

        line = ["0"] * width
        for _ in range(height):
            for i in range (width):
                line[i] = "1" if randbit_f(1) else "0"
            write_f("".join(line) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate a Game Of Life grid c ontaining random 1s and 0s.")
    parser.add_argument("width", type=int, help="Grid width")
    parser.add_argument("height", type=int, help="Grid height")
    parser.add_argument("output_file", help="Path to the output file")
    
    args = parser.parse_args()
    generate_write_grid(args.width, args.height, args.output_file)   

if __name__ == "__main__":
    main()

