import pandas as pd
import numpy as np

def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)

# Function from XYZT's Kernel on the same topic. 
def total_distance(dfcity,path, prime_cities):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance + \
            np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

def sub_distance(dfcity, path, start, end, prime_cities): # 部分パスの距離を求める
    prev_city = path[start] # 部分パスの開始を指定
    sub_distance = 0
    step_num = start + 1 # pathは０番目から始ま理、path０番目が第一ステップになることに注意
    for city_num in path[(start+1): (end+1)]:
        # print("prev_city: %d" %prev_city)
        # print("next_city: %d" %city_num)
        next_city = city_num
        sub_distance = sub_distance + \
            np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + 
                    pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return sub_distance    
