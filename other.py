import math


N = 10 ** 3

ans = 0
a_i = 0

for i in range(1, N):
    # if i % 100 == 0:
    #     print(i)
    for k in range(i // 2 + 1):
        b_i = 1 / (math.factorial(k) * math.factorial(i - 2*k) * 2 ** k)
        a_i += b_i

    ans += a_i
    print(i, a_i)
    if a_i < 10 ** -30 and i > 5:
        break
    a_i = 0

print(ans)


for i in range(2, 100, 2):
    print(math.factorial(i-1) / math.factorial(i/2))

