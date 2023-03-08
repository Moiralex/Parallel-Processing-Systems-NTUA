import matplotlib.pyplot as plt

steps_64=[]
steps_1024=[]
steps_4096=[]
steps=[steps_64, steps_1024, steps_4096]
threads=[1,2,4,6,8]

with open('results.out') as f:
    lines = f.readlines()
    index=0
    for line in lines:
        if line.startswith("GameOfLife"):
            steps[index].append(float(line.split()[-1]))
            index+=1
            if index>2:
                index=0

print(steps)  

fig = plt.figure(figsize=(5,12))

ax = fig.add_subplot(111)
plt.subplot(311)
plt.plot(threads, steps_64)
plt.xlabel="# threads"
plt.ylabel="Execution time (s)"
plt.title("Execution time vs threads for 64x64 board")  

plt.subplot(312)
plt.plot(threads, steps_1024)
plt.xlabel="# threads"
plt.ylabel="Execution time (s)"
plt.title("Execution time vs threads for 1024x1024 board")

plt.subplot(313)
plt.plot(threads, steps_4096)
plt.xlabel="# threads"
plt.ylabel="Execution time (s)"
plt.title("Execution time vs threads for 4096x4096 board")

plt.suptitle("Execution time vs threads for different sized boards")
plt.savefig("results.png") 
