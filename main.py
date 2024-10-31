lines = []
with open(r"C:\Users\maxfo\pythonProjects\fusionAPIimg2Sketch\out.txt", "r") as inFile:
    for line in inFile:
        line = line.split()
        lines.append([line[0], line[1], line[2], line[3]])

for line in lines:
    x0, y0, x1, y1 = int(line[0])/10, int(line[1])/10, int(line[2])/10, int(line[3])/10
    print(x0, x1, y0, y1)
    
print(lines)