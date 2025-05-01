# Python skripts, lai failā saglabātu nejaušu Game of Life režģi

import random
import string
import argparse

def generate_grid(width, height):
    return [[str(random.randint(0, 1)) for _ in range(width)] for _ in range(height)]

def save(grid, output_file):
    with open(output_file, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate a Game Of Life containing random 1s and 0s.")
    parser.add_argument("width", type=int, help="Grid width")
    parser.add_argument("height", type=int, help="Grid height")
    parser.add_argument("output_file", help="Path to the output file")
    
    args = parser.parse_args()
    
    grid = generate_grid(args.width, args.height)
    save(grid, args.output_file)

if __name__ == "__main__":
    main()

