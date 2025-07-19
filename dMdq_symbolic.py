import sympy as sp

R1, R2, L1, L2, x, y, theta1, theta2, theta1_dot, theta2_dot, ms, m1, m2, msc, I1, I2 = sp.symbols(
    'R1 R2 L1 L2 x y theta1 theta2 theta1_dot theta2_dot ms m1 m2 msc I1 I2', real=True)

q = sp.Matrix([theta1, theta2, x, y])

A = sp.Matrix([[0, 0, 1, 0], 
                [0, 0, 0, 1]])
B = sp.Matrix([[-R1*sp.sin(theta1), 0, 1, 0],
                [R1*sp.cos(theta1), 0, 0, 1]])
C = sp.Matrix([[-L1*sp.sin(theta1)-R2*sp.sin(theta1+theta2), -R2*sp.sin(theta1+theta2), 1, 0],
                [L1*sp.cos(theta1)+R2*sp.cos(theta1+theta2), R2*sp.cos(theta1+theta2), 0, 1]])
D = sp.Matrix([[-L1*sp.sin(theta1)-L2*sp.sin(theta1+theta2), -L2*sp.sin(theta1+theta2), 1, 0],
                [L1*sp.cos(theta1)+L2*sp.cos(theta1+theta2), L2*sp.cos(theta1+theta2), 0, 1]])
E = sp.Matrix([1, 0, 0, 0]).reshape(1, 4)  # Ensure E is a column vector
F = sp.Matrix([1, 1, 0, 0]).reshape(1, 4)  # Ensure F is a column vector

M = ms * (A.T @ A) + m1 * (B.T @ B) + m2 * (C.T @ C) + msc * (D.T @ D) + I1 * (E.T @ E) + I2 * (F.T @ F)

L = sp.Matrix([[1, 1],
                [0, 1]])

dMdq = sp.derive_by_array(M, q)
print(dMdq)