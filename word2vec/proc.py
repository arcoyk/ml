import random
char = ['a', 'b', 'c', 'd', 'e']
s = ''
for i in range(len(char)):
    for x in range(100):
        if x % 2 == 0:
            s += char[i]
            s += ' '
        else:
            s += char[(i + 1) % len(char)]
            s += ' '
print(s)
for i in range(len(char)):
    for x in range(100):
        if x % 2 == 0:
            s += char[i]
            s += ' '
        else:
            s += char[(i + 1) % len(char)]
            s += ' '
print(s)
for i in range(len(char)):
    for x in range(100):
        if x % 2 == 0:
            s += char[i]
            s += ' '
        else:
            s += char[(i + 1) % len(char)]
            s += ' '
print(s)
