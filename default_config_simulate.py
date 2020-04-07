import random
def choice3_a():
    return random.choices(population=['a', 'b', 'c'],weights=[0.8, 0.19, 0.01])[0]
def choice3_b():
    return random.choices(population=['a', 'b', 'c'],weights=[0.27, 0.7, 0.03])[0]
def choice3_c():
    return random.choices(population=['a', 'b', 'c'],weights=[0.0, 0.1, 0.9])[0]

high_risk_threshold = 10


trans_mat = {'a':choice3_a,'b':choice3_b,'c':choice3_c}

cities_dist = {}
for jj in range(30):
    if jj%15 in list(range(12)):
        cities_dist["ii_%d"%jj] = 'a'
    if jj%15 in [12,13]:
        cities_dist["ii_%d"%jj] = 'b'
    if jj%15 in [14]:
        cities_dist["ii_%d"%jj] = 'c'
