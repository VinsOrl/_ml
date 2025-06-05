import sympy as sp

x, y, z = sp.symbols('x y z')

f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

grad_f = [sp.diff(f, var) for var in (x, y, z)]

critical_points = sp.solve(grad_f, (x, y, z))

min_value = f.subs(critical_points)

critical_points, min_value