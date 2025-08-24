import sympy as sp
from math import acos, degrees

a = sp.Matrix([1, 2, 3])
b = sp.Matrix([4, 5, 6])
c = sp.Matrix([3, 2, 1])
d = sp.Matrix([6, 5, 4])

e1 = sp.Matrix([1,0,0])
e2 = sp.Matrix([0,1,0])
e3 = sp.Matrix([0,0,1])

def angle_between(u, v):
    
    dot = float(u.dot(v))
    nu = float(u.norm())
    nv = float(v.norm())
    if nu == 0 or nv == 0:
        return None, None  
    cos_theta = dot / (nu * nv)
    
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_rad = acos(cos_theta)
    theta_deg = degrees(theta_rad)
    return theta_rad, theta_deg

# (a)

s1 = a + b + c + d
s2 = a + b - c - d
s3 = a - b + c - d
s4 = -a + b - c + d

# (b) 
angles_to_basis = {}
for name, vec in [('a',a), ('b',b), ('c',c), ('d',d)]:
    angles_to_basis[name] = {
        'e1': angle_between(vec, e1),
        'e2': angle_between(vec, e2),
        'e3': angle_between(vec, e3),
    }

# (c) 
mags = { 'a': float(a.norm()), 'b': float(b.norm()), 'c': float(c.norm()), 'd': float(d.norm()) }

# (d) 
angle_a_b = angle_between(a,b)
angle_c_d = angle_between(c,d)

# (e) 
proj_a_on_b = (a.dot(b) / b.dot(b)) * b

# (f) 
# en R^3, si el rango de la matriz con las columnas a,b,c,d es <= 2 entonces son coplanares
M = sp.Matrix.hstack(a, b, c, d)
rank_M = M.rank()
coplanar = (rank_M <= 2)

# (g)
dot_sum = (a + b).dot(c + d)

# (h) 
cross_ab = a.cross(b)
cross_bc = b.cross(c)
cross_cd = c.cross(d)

angle_crossab_d = angle_between(cross_ab, d)
angle_crossbc_d = angle_between(cross_bc, d)
angle_crosscd_d = angle_between(cross_cd, d)

# (i) 
scalar_triple = c.dot(a.cross(b))
print("Vectores:")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("\n(a) Sumas:")
print("a + b + c + d =", s1)
print("a + b - c - d =", s2)
print("a - b + c - d =", s3)
print("-a + b - c + d =", s4)

print("\n(b) Ángulos entre a,b,c,d y e1,e2,e3 (en radianes, grados):")
for name, data in angles_to_basis.items():
    print(f"{name}:")
    for base, (rad, deg) in data.items():
        if rad is None:
            print(f"  ángulo con {base}: indefinido (norma cero)")
        else:
            print(f"  ángulo con {base}: {rad:.4f} rad, {deg:.2f}°")

print("\n(c) Magnitudes:")
for name, val in mags.items():
    print(f"||{name}|| = {val:.6f}")

print("\n(d) Ángulos entre pares:")
if angle_a_b[0] is not None:
    print(f"ángulo entre a y b: {angle_a_b[0]:.4f} rad, {angle_a_b[1]:.2f}°")
else:
    print("ángulo entre a y b: indefinido")
if angle_c_d[0] is not None:
    print(f"ángulo entre c y d: {angle_c_d[0]:.4f} rad, {angle_c_d[1]:.2f}°")
else:
    print("ángulo entre c y d: indefinido")

print("\n(e) Proyección de a sobre b (vector):")
print("proj_b(a) =", proj_a_on_b)

print("\n(f) ¿Los vectores a, b, c, d son coplanares?")
print("rango de la matriz [a b c d] =", rank_M)
print("coplanares ->", coplanar)

print("\n(g) (a + b) · (c + d) =", dot_sum)

print("\n(h) Productos cruz:")
print("a × b =", cross_ab)
print("b × c =", cross_bc)
print("c × d =", cross_cd)
print("\nÁngulos entre cada producto cruz y d:")
if angle_crossab_d[0] is not None:
    print(f"(a×b) con d: {angle_crossab_d[0]:.4f} rad, {angle_crossab_d[1]:.2f}°")
else:
    print("(a×b) con d: indefinido")
if angle_crossbc_d[0] is not None:
    print(f"(b×c) con d: {angle_crossbc_d[0]:.4f} rad, {angle_crossbc_d[1]:.2f}°")
else:
    print("(b×c) con d: indefinido")
if angle_crosscd_d[0] is not None:
    print(f"(c×d) con d: {angle_crosscd_d[0]:.4f} rad, {angle_crosscd_d[1]:.2f}°")
else:
    print("(c×d) con d: indefinido")

print("\n(i) c · (a x b) =", scalar_triple)



