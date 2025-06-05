import random
import math

cities = [
    (0, 3), (0, 0), (0, 2), (0, 1),
    (1, 0), (1, 3), (2, 0), (2, 3),
    (3, 0), (3, 3), (3, 1), (3, 2)
]

def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def total_path_distance(path):
    distance = 0
    for i in range(len(path)):
        distance += euclidean_distance(cities[path[i]], cities[path[(i + 1) % len(path)]])
    return distance

def generate_neighbor(path):
    new_path = path[:]
    i, j = sorted(random.sample(range(len(path)), 2))
    new_path[i:j+1] = reversed(new_path[i:j+1])
    return new_path

def hill_climbing(initial_path, distance_fn, neighbor_fn, max_attempts=10000):
    current_path = initial_path
    current_cost = distance_fn(current_path)
    best_path = current_path[:]
    best_cost = current_cost
    attempts = 0

    print(f"Initial route cost: {best_cost:.2f}, Route: {best_path}")

    while attempts < max_attempts:
        neighbor = neighbor_fn(current_path)
        cost = distance_fn(neighbor)

        if cost < best_cost:
            best_cost = cost
            best_path = neighbor[:]
            current_path = neighbor
            attempts = 0
            print(f"Improved route: {best_cost:.2f}, Route: {best_path}")
        else:
            attempts += 1

    print(f"Final best route: {best_cost:.2f}, Route: {best_path}")
    return best_path

initial_route = list(range(len(cities)))
random.shuffle(initial_route)

best_solution = hill_climbing(initial_route, total_path_distance, generate_neighbor)
