from main_tx import *

ans = genetic_alg(nice_func, [-100, -100], [100, 100], 2) # Размер популяции 2
print("Population size: 2; x: %s, y: %s, f: %s" % (ans.x, ans.y, ans.f))
ans = genetic_alg(nice_func, [-100, -100], [100, 100], 5) # Размер популяции 5
print("Population size: 5; x: %s, y: %s, f: %s" % (ans.x, ans.y, ans.f))
ans = genetic_alg(nice_func, [-100, -100], [100, 100], 10) # Размер популяции 10
print("Population size: 10; x: %s, y: %s, f: %s" % (ans.x, ans.y, ans.f))
ans = genetic_alg(nice_func, [-100, -100], [100, 100], 100) # Размер популяции 100
print("Population size: 100; x: %s, y: %s, f: %s" % (ans.x, ans.y, ans.f))
