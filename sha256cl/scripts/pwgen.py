# Vienkarss python skripts, lai faila saglabatu nejausas paroles
# Paredzets lielu parolu failu testiem

import random
import string
import sys

def generate_pws(n):
    passwords = []
    for _ in range(n):
        # Garums 6 - 16 simboli
        length = random.randint(6, 16)
        # ASCII burti un skaitli
        current_pw = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        passwords.append(current_pw)
    return passwords

def save(passwords, filename):
    with open(filename, 'w') as file:
        for password in passwords:
            file.write(password + '\n')

if __name__ == "__main__": # ta ka padod argumentus, tad no main scope jasanem CLI argv
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <number_of_passwords> <output_file>")
        sys.exit(1)

    try:
        # Parolu skaita un faila nosaukuma ievade caur CLI argumentiem
        n = int(sys.argv[1])
        filename = sys.argv[2] 

        passwords = generate_pws(n)
        save(passwords, filename)

        print(f"Generated {n} passwords and saved to '{filename}'")
    except ValueError:
        print("Error: Argument must be an integer representing the number of passwords!")
        sys.exit(1) 
