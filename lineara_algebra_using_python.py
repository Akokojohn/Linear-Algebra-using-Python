import scipy.stats as scs
import math
import numpy as np
import pandas as pd
import matplotlib as mp

col_vector = np.array([[3,7],[4,6],[4,5]])#column vector with 3 rows and 2 column
print(col_vector)
print("shape of the vector is = ",col_vector.shape)
print(np.zeros(6).reshape(6,-1))
vec=np.array([[4],[-1],[2]])
print(vec)
print(vec[2])

#magnitude of a vector
vector=np.array([3,3])
print("The magnitude or norm of the vector is ", round(np.linalg.norm(vector),3))

#Matrix
matrix_2x2 = np.array([[2,2],[3,3]])
matrix_3x3 = np.array([[2,2,2],[3,3,3],[4,4,4]])
print(matrix_2x2)
print(matrix_3x3)
print(matrix_3x3[2,1])#Accessing index of a matrix

#Transpose of a matrix
A = np.array([[2,1],[3,2],[4,5]])
print(A)
print(np.transpose(A))

"""Types of Matrices"""
"""
-Diagonal
-Identity
-Symmetric
-Triangular
"""
"""Diagonal Matrix: has zeroes outside the main diagonal or principal diagonal"""
diagonal=np.diag((3,4,5))
print(diagonal)
mat_range=np.diag(np.arange(1,6,2))
print(mat_range)

"""Identity Matrix: has 1 across diagonal and 0 in the rest"""
identity = np.identity(3)
print(identity)
indentity2 = np.eye(3)
print(indentity2)

"""Symmetric Matrix:  if A = A transpose"""
B = np.array([[2,3,1],[3,4,-1],[1,-1,1]])
transpose_of_B = B.transpose()
print(B)
print(transpose_of_B)
#comparing each element of both matrices and saving it in a variable comparison
comparison = (B == transpose_of_B)
#checking if all the elements in the matrix are true
equal_arrays = comparison.all()
print(equal_arrays)

"""Triangular Matrix:
    - Lower: a square matrix in which all the elements above the main diagonal are zero
    -Upper: elements above the main diagonal are zero
"""
lower = np.tril([[1,2,3],[4,5,6],[7,8,9]])
print(lower)
upper = np.triu([[1,2,3],[4,5,6],[7,8,9]])
print(upper)
print("")

"""OPERATIONS OVER VECTORS AND MATRICES"""
"""Vector Addition"""
print("Vector Addition\n")
vec1 = np.array([1,2,3])
vec2 = np.array([4,5,6])
add = np.add(vec1,vec2)
print(add)

"""Matrix Addition"""
# Creation of 2 matrices
matrix_1 = np.array([[10, 20, 30],
                     [-30, -40, -50]])
matrix_2 = np.array([[100, -200, 300],
                     [30, 50, 70]])

print("1st  Matrix : \n", matrix_1)
print("2nd  Matrix : \n", matrix_2)
# Addition of the matrices using np.add() function
out = np.add(matrix_1, matrix_2)
print("Added Matrix : \n", out)

"""Multiplication of Vector by a scaler"""
v = np.array([[2],[-1],[3]])
vmul = v*2.5
print(vmul)

#Defining a Vector 'v' and scalar 's'
v = np.array([[2],[-1],[3]])
s = 2.5
#Scalar Vector Multiplication
vector_mul = v*s
print("Vector: \n",v,"\nScalar:",s,"\nScalar Vector multiplication:\n",vector_mul)

"""Multiplication of Matrix by a Scaler"""

mat = np.array([[10, 20, 30],[-30, -40, -50]])
matmul = mat*2.5
print(matmul)

#Defining a Vector 'v' and scalar 's'
v = np.array([[2],[-1],[3]])
s = 2.5
#Scalar Vector Multiplication
vector_mul = v*s
print("Vector: \n",v,"\nScalar:",s,"\nScalar Vector multiplication:\n",vector_mul)

"""Inner Product or Dot Product"""
#Multipying the inner elements of a matrix and the result becomes a scaler
#Syntax is  np.inner(v1,v2)
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
v12 = np.dot(v1,v2)
v13 = np.inner(v1,v2)
print(v12)
print(v13)

#Othogonal vectors: if the inner products of twon no-zero is 0
v11 = np.array([[1],[2],[3]])
v22 = np.array([[4],[5],[6]])
trans = np.transpose(v11)
res = np.dot(trans,v22)
print(res)

#Creating vectors
Vector_1 = np.array([[3],[-1],[2]])
Vector_2 = np.array([[2],[4],[-1]])
print("Vector 1\n",Vector_1)
print("Vector 2\n",Vector_2)
#Finding the transpose of Vector_1
trans = np.transpose(Vector_1)
#Finding the dot product
result = np.dot(trans,Vector_2)
print("Dot Product\n",result)

"""Angle between the Vectors"""
# Function to find angle between two vectors
def angle_between(vector_1, vector_2):
    dot_pr = vector_1.dot(vector_2)
    norms = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

    return np.rad2deg(np.arccos(dot_pr / norms))
# Create two vectors
vector_1 = np.array([3, -1, 2])
vector_2 = np.array([2, 4, -1])
print("v1 = ", vector_1, "\nv2 = ", vector_2)
# Find the angle between them by using the function angle_between()
print("Angle Between the vectors v1 and v2 in 'degree' is :", angle_between(vector_1, vector_2))

"""Cauchy-Schwarz Inequality"""

"""Matrix - Vector Multiplication"""
A =  np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("A :\n", A)
#Define a column vector
v = np.array([[1], [2], [3]])
print("v :\n", v)
# Find product of the matrix and Vector
product = np.matmul(A,v)
print("Product of A and v is: \n", product)

"""Matrix Multiplication"""
"""
-It is not always commutative i.e. A.B may not be equal to B.A
-It is associative i.e. A.(B.C) = (A.B).C
-It is distributive over addition i.e, A.(B+C) = A.B+B.C
"""
A = np.array([[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]])
B = np.array([[4, 2, 3],
            [2, 0, 7],
            [1, 3, 0]])
print("A :\n", A)
print("B :\n", B)
# Multiplying A and B
result=np.matmul(A,B)
print("Matrix Multiplication: A.B :\n",result)

"""Permutation Matrix"""
#Permutation matrix is obtained by permuting (or interchanging) the rows or columns of an Identity matrix.

"""Determinants"""
#if the determinant of a matrix is zero then the matrix is called a Singular Matrix
matx=np.array([[10,2],[5,2]])
det_a = np.linalg.det(matx)
print("The Determinant is  ",det_a)

"""Matrix Inversion"""
Ainv = np.linalg.inv(matx)
zx = np.matmul(matx,Ainv)
print("The Inverse of the Matrix is  ",Ainv)#Gives the inverse of the matrix
print(zx)

"""Orthogonal Matrix"""
A = np.array([[1.0,0.0],
                [0.0,1.0]])
print("A:\n", A)
#Checking for A.AT=AT.A
comparison_1 = np.dot(A.transpose(),A) == np.dot(A,A.transpose())
print(comparison_1)
#Checking for A.AT=Identity Matrix
comparison_2 =  np.dot(A.transpose(),A)== np.eye(2)
print(comparison_2)
# Comparing both the comparison done earlier
comparison_3 = comparison_1 == comparison_2
#Checking if all elements of matrix comparision are true.
equal_arrays = comparison_3.all()
print("A it an orthogonal matrix: ",equal_arrays)

"""Rank of Matrix"""
#The rank of matrix A of dimension (m, n) is the number of linearly independent rows (or columns) of A.
A = np.array([[1,1.0],[3,3]])
print('The matrix A is:\n', A)
B = np.eye(4)
print('The matrix B is:\n', B)
# Finding the rank of matrix a using np.linalg.matrix_rank() function
print(" Rank of A is:",np.linalg.matrix_rank(A))
# Finding the rank of matrix b using np.linalg.matrix_rank() function
print(" Rank of B is:",np.linalg.matrix_rank(B))

AA= np.array([4,5,6])
BB = np.array([-1,4,8])
print(np.inner(AA,BB))

"""Linear Transformations"""
"""
-Stretching
-Reflection
-Rotation
-Projection

-Stretching: The matrix A stretches the vector x up to c units when A is multiplied with x
"""
"""Reflection """

"""Linear Equations"""
"""Solving Linear Equation"""

matrix_1 = np.array([[1, 3, -1],
              [2, 5, 4],
              [2, 3, -1]])
# Define the vector
matrix_2 = np.array([4,19,7])
# Find the solution for the system of equations using the solve() method
x= np.linalg.solve(matrix_1, matrix_2)
print("The value of x1 is: ",x[0])
print("The value of x2 is: ",x[1])
print("The value of x3 is: ",x[2])

"""Techniques of solving a system of linear equations"""
"""
-Gaussian Elimination
-Cramer's Rule
"""
"""
-Gaussian Elimination
Gaussian Elimination is an algorithm that helps in solving a System of Linear Equations through the following 
elementary row operations on Augmented matrix:

Swapping rows
Multiplying a row by a non-zero scalar
Adding a multiple of one row to another row
The output of this algorithm is an Upper Triangular matrix which helps in finding the solution of the System of Equations.
"""

"""
Cramer’s Rule

Cramer’s Rule is a method to solve a System of Linear Equations. 
Unlike Gaussian Elimination, this rule helps in finding solutions to a subset of variables 
rather than solving the entire System of Linear Equations. 
This is helpful when linear equations contain large number of variables.
"""
ux = np.array([1,0,1])
vx = np.array([0,0,1])
print(np.inner(ux,vx))

uc = np.array([[1],[2],[3]])
vc = np.array([[4],[5],[6]])
xc = np.transpose(uc)
xz = np.matmul(xc,vc)
print(xz)

rk = np.eye(3)
print(rk)
print(np.linalg.matrix_rank(rk))