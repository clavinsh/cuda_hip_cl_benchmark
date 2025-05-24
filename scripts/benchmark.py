# Skripts, kas izpilda sha256 un gol programmu docker konteinerus ar dažādiem ievadfailiem
# Log failiem izveido jēdzīgus nosaukumus

import os
import subprocess
import time
import json
import pandas as pd
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
import random
import gc

class BenchmarkStuff:
    def __init__(self, base_dir="./benchmark_data"):

        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.log_dir = self.base_dir / "logs"
        
        for directory in [self.input_dir, self.output_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.implementations = ["cl", "hip", "cuda"]
        
        self.gol_confs = {
            "grid_sizes": [(100, 100), (1000, 1000), (10000, 10000)],
            "steps": [1, 100, 1000, 10000]
        }
        
        self.sha256_confs = {
            "password_counts": [10000, 1000000, 100000000],
            "positions": ["q1", "q2", "q3", "q4", "not_found"]
        }
       
        self.results = {
            "gol": [],
            "sha256": []
        }
        
        self.target_pws = {}
        self.target_hashes = {}

        self.set_dir_permissions()


    def set_dir_permissions(self):
        for directory in [self.output_dir, self.log_dir]:
            os.system(f"chmod -R 777 {directory.absolute()}")
 

    def generate_input_files(self):
        print("generate_input_files")
        
        for width, height in self.gol_confs["grid_sizes"]:
            output_file = self.input_dir / f"grid_{width}x{height}.txt"
            if not output_file.exists():
                print(f"Generating {width}x{height} grid file...")
                cmd = ["python3", "./gridfile_gen.py", str(width), str(height), str(output_file)]
                subprocess.run(cmd, check=True)

        for count in self.sha256_confs["password_counts"]:
            output_file = self.input_dir / f"passwords_{count}.txt"
            if not output_file.exists():
                print(f"Generating {count} passwords file...")
                cmd = ["python3", "./pwgen.py", str(count), str(output_file)]
                subprocess.run(cmd, check=True)


    def precompute_target_passwords(self):
        print("precompute_target_passwords")
        
        for count in self.sha256_confs["password_counts"]:
            for position in self.sha256_confs["positions"]:
                key = f"{count}_{position}"
                
                if position == "not_found":
                    password = f"NotInFile_{random.randint(10000000, 99999999)}"
                    self.target_pws[key] = password
                    self.target_hashes[key] = hashlib.sha256(password.encode()).hexdigest()
                    continue
                
                pw_file = self.input_dir / f"passwords_{count}.txt"

                if not pw_file.exists():
                    print(f"Warning: Password file {pw_file} not found, skipping precomputation")
                    continue

                total_lines = count 
                quartile_size = total_lines // 4
                
                q_positions = {
                    "q1": quartile_size // 2,                   
                    "q2": quartile_size + (quartile_size // 2), 
                    "q3": 2 * quartile_size + (quartile_size // 2), 
                    "q4": 3 * quartile_size + (quartile_size // 2) 
                }
                
                target_line = q_positions[position]
                
                password = self._get_specific_line(pw_file, target_line)
                
                self.target_pws[key] = password
                self.target_hashes[key] = hashlib.sha256(password.encode()).hexdigest()
       
        # piespiedu kārtu garbabe savāc, lai minimizētu footprint
        gc.collect()
        print(f"Generated {len(self.target_pws)} passwords and their respective hashes")

    def _get_specific_line(self, file_path, line_number):
        with open(file_path, 'r') as f:
            for _ in range(line_number):
                f.readline()
            
            line = f.readline().strip()
            return line if line else "password" 
    
    def run_gol_benchmark(self, implementation, grid_size, steps, run_id=0, cooldown=1):
        width, height = grid_size

        grid_file = f"grid_{width}x{height}.txt"
        output_file = f"grid_{width}x{height}_steps_{steps}_{implementation}_{run_id}.txt"
        log_file = f"gol_{width}x{height}_steps_{steps}_{implementation}_{run_id}.log"

        benchmar_config_name = f"gol_{width}x{height}_steps_{steps}_{implementation}_{run_id}"
        gpu_process, gpu_log_file = self.start_gpu_monitoring(benchmar_config_name) 

        try:

            volumes = [
                f"{self.input_dir.absolute()}:/input",
                f"{self.output_dir.absolute()}:/output",
                f"{self.log_dir.absolute()}:/logs"
            ]
           
            # docker CLI principā kā teksta virkni sakabina kopā
            docker_cmd = [
                "docker", "run", "--rm"
            ]
            for volume in volumes:
                docker_cmd.extend(["-v", volume])
            docker_cmd.extend([
                "--gpus=all",
                f"gol{implementation}",
                f"/input/{grid_file}",
                f"/output/{output_file}",
                str(steps),
                f"/logs/{log_file}"
            ])
            
            print(f"Executing GoL {implementation} with {width}x{height} grid for {steps} steps (run idx {run_id})...")
            
            # redundanti, bet kkas negāja bez tiešas log failu iestatīšanas
            os.system(f"chmod -R 777 {self.log_dir.absolute()}")
           
            # keša iztīrīšana, konsekventiem laidieniem
            self._clear_caches()
            
            gc.collect()
            
            start_time = time.time() #gnjau nevajadzēs, bet better safe than sorry
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            if cooldown > 0:
                time.sleep(cooldown)
            
            if result.returncode != 0:
                print(f"Error! {result.stdout}\n{result.stderr} ")
                return None

            log_path = self.log_dir / log_file
            
            return {
                "implementation": implementation,
                "grid_size": f"{width}x{height}",
                "steps": steps,
                "run_id": run_id,
                "total_time": elapsed_time,
                "log_file": str(log_path)
            }

        finally:
            self.stop_gpu_monitoring(gpu_process)

    def run_sha256_benchmark(self, implementation, password_count, position, run_id=0, cooldown=1):
        """Run one SHA256 benchmark with directory-based volume mounting"""
        pw_file = f"passwords_{password_count}.txt"
        log_file = f"sha256_{password_count}_{position}_{implementation}_{run_id}.log"
        
        key = f"{password_count}_{position}"
        if key not in self.target_hashes:
            print(f"Error! No hash for {key}")
            return None
            
        pw_hash = self.target_hashes[key]
        target_pw = self.target_pws[key]

        benchmar_config_name = f"gol_{width}x{height}_steps_{steps}_{implementation}_{run_id}"
        gpu_process, gpu_log_file = self.start_gpu_monitoring(benchmar_config_name) 

        try:
        
            volumes = [
                f"{self.input_dir.absolute()}:/input:ro",
                f"{self.log_dir.absolute()}:/logs"
            ]
            
            docker_cmd = [
                "docker", "run", "--rm"
            ]
            
            for volume in volumes:
                docker_cmd.extend(["-v", volume])
            
            docker_cmd.extend([
                "--gpus=all",
                f"sha256{implementation}",
                f"/input/{pw_file}",
                pw_hash,
                f"/logs/{log_file}"
            ])
            
            print(f"Executing SHA256 {implementation} with {password_count} passwords, target pw position {position} (run idx {run_id})...")
            
            # redundanti, bet kkas negāja bez tiešas log failu iestatīšanas
            os.system(f"chmod -R 777 {self.log_dir.absolute()}")

            self._clear_caches()
            
            gc.collect()
            
            start_time = time.time()
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            # Optional cooldown period between benchmarks
            if cooldown > 0:
                time.sleep(cooldown)
            
            if result.returncode != 0:
                print(f"Error! {result.stderr}")
                return None

            log_path = self.log_dir / log_file
            
            return {
                "implementation": implementation,
                "password_count": password_count,
                "position": position,
                "target_hash": pw_hash,
                "target_password": target_pw,
                "run_id": run_id,
                "total_time": elapsed_time,
                "log_file": str(log_path)
            }

        finally:
            self.stop_gpu_monitoring(gpu_process)
    
    def _clear_caches(self):
        try:
            if os.path.exists("/proc/sys/vm/drop_caches"):
                subprocess.run(["sync"], check=False)
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3")
        except Exception:
            pass
    
    def run_all_benchmarks(self, runs=3, cooldown=1):
        """Run all benchmark configurations with minimal Python footprint"""
        self.generate_input_files()
        
        self.precompute_target_passwords()
        
        print("\nGoL benchmarks")
        for implementation in self.implementations:
            for grid_size in self.gol_confs["grid_sizes"]:
                for steps in self.gol_confs["steps"]:
                    for run_id in range(runs):
                        result = self.run_gol_benchmark(
                            implementation, 
                            grid_size, 
                            steps, 
                            run_id, 
                            cooldown
                        )
                        if result:
                            self.results["gol"].append(result)
        
        print("\nSHA256 Benchmarks")
        for implementation in self.implementations:
            for password_count in self.sha256_confs["password_counts"]:
                for position in self.sha256_confs["positions"]:
                    for run_id in range(runs):
                        result = self.run_sha256_benchmark(
                            implementation, 
                            password_count, 
                            position, 
                            run_id, 
                            cooldown
                        )
                        if result:
                            self.results["sha256"].append(result)
        
        self.save_results_to_csv()
        
        print("\nBenchmark runs done. Log at:", self.log_dir)
    
    def save_results_to_csv(self):
        """Save benchmark metadata results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.results["gol"]:
            gol_df = pd.DataFrame(self.results["gol"])
            gol_csv = self.base_dir / f"gol_benchmark_metadata_{timestamp}.csv"
            gol_df.to_csv(gol_csv, index=False)
            print(f"GoL benchmark metadata at {gol_csv}")
        
        if self.results["sha256"]:
            sha256_df = pd.DataFrame(self.results["sha256"])
            sha256_csv = self.base_dir / f"sha256_benchmark_metadata_{timestamp}.csv"
            sha256_df.to_csv(sha256_csv, index=False)
            print(f"SHA256 benchmark metadata at {sha256_csv}")
        
        config = {
            "timestamp": timestamp,
            "implementations": self.implementations,
            "gol_configs": {
                "grid_sizes": [f"{w}x{h}" for w, h in self.gol_confs["grid_sizes"]],
                "steps": self.gol_confs["steps"]
            },
            "sha256_configs": {
                "password_counts": self.sha256_confs["password_counts"],
                "positions": self.sha256_confs["positions"]
            },
            "log_dir": str(self.log_dir),
            "target_passwords": self.target_pws,
            "target_hashes": self.target_hashes
        }
        
        config_file = self.base_dir / f"benchmark_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Benchmark configuration at {config_file}")

    def _get_gpu_monitoring_command(self, log_file):
        if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:
            return [
                "nvidia-smi",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,clocks.current.graphics,clocks.current.memory",
                "--format=csv",
                "--loop=1",
                "-f", str(log_file)
            ]
    
        print("Error! Couldn't find GPU monitoring util")
        return None

    def start_gpu_monitoring(self, benchmark_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_log_file = self.log_dir / f"gpu_metrics_{benchmark_id}_{timestamp}.csv"
        gpu_cmd = self._get_gpu_monitoring_command(gpu_log_file)

        if gpu_cmd:
            print(f"GPU montitoring util found, starting it")
            process = subprocess.Popen(gpu_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return process, gpu_log_file
        
        # lai konsekventi ar to, ko tiek atgriezts, ja ir gpu monitoringa util
        return None, None

    def stop_gpu_monitoring(self, process):
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
 



def main():
    parser = argparse.ArgumentParser(description='Benchmarking GoL and SHA256 Docker implementations')
    parser.add_argument('--base-dir', type=str, default='./benchmark_data',
                        help='Root directory for benchmark data')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs for each configuration')
    parser.add_argument('--cooldown', type=float, default=1.0,
                        help='Cooldown time in seconds between runs')
    args = parser.parse_args()
    
    benchmark = BenchmarkStuff(base_dir=args.base_dir)
    benchmark.run_all_benchmarks(runs=args.runs, cooldown=args.cooldown)

if __name__ == "__main__":
    main()
