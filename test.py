import numpy

def angle_between(v1, v2):
    v1_u = v1 / numpy.linalg.norm(v1)
    v2_u = v2 / numpy.linalg.norm(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))


v1 = [1, 1, 0]

b1 = [0.1,1,0]
p1 = [0,1, 0]

real_phi = angle_between(v1, b1)
beta = angle_between(v1, p1)

print(numpy.radbeta)
print(real_phi)



