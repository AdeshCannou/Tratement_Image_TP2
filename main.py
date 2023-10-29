# Exercice 1

# Q1
# Q1.1 50
print(len(range(1,101))//2) # 50
# Q1.2
print(len(range(1,101))//.5) # 200

# Q2
r = range(1,101)
c = range(1,101)
rad = range(5,int(100*2**.5)+1)
print(len(r)*len(c)*len(rad)) # 1 370 000

# Q3
def acc(i,j,k):
    return [r[i-1],c[j-1],rad[k-1]]

print(acc(1,1,1)) # (1, 1, 5)
print(acc(10,7,30)) # (0, 7, 34)

# Q4
def reverse_acc(i,j,k):
    return [r.index(i)+1, c.index(j)+1, rad.index(k)+1]

print(reverse_acc(40,40,13)) # (40, 40, 9)