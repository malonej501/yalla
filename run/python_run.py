import subprocess

command = ["nvcc", "-std=c++14", "-arch=native", "-o", "exec"]
call = subprocess.run(command)

command = ["./exec"]
call = subprocess.run(command)
