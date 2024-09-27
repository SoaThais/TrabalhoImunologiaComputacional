from fenics import *
from ufl import nabla_grad, nabla_div
import numpy as np

# Tempo total de simulação
end_time = 10.0   
# Número de passos de tempo          
num_steps = 50
# Tamanho do passo de tempo
dt = Constant(end_time / num_steps)

################## Parâmetros ##################

# Porosidade
phi_f     = Constant(0.2)           
# Coeficiente de Difusão do Patógeno [cm^2 / h]
D_b       = Constant(0.00005)
# Taxa de Reprodução do Patógeno  [1 / h]
c_p       = Constant(0.15)
# Taxa de Fagocitose [cm^3 / (h 10^7 cell)]
lambda_nb = Constant(1.8)

# Coeficiente de Difusão do Leucócito
D_n       = Constant(0.00005)
# Taxa de Quimiotaxia [cm^5 / (h 10^7 cell)]
X_nb      = Constant(0.0001) 
# Taxa de Apoptose Induzida [cm^3 / (h 10^10 cell)]
lambda_bn = Constant(0.1)
# Permeabilidade Capilar dos Leucócitos [cm^3 / (h 10^7 cell)]
gamma_n   = Constant(0.1)
# Concentração dos Leucócitos no Sangue [0.55 * 10^7 cell]
C_n_max   = Constant(0.55)
# Taxa de Apoptose [1 / h]
mu_n      = Constant(0.2)

# ---> Obs: 1mmHg = 0.133322 kPa
# ---> Obs: 1kPa  = 7.50062 mmHg
# ---> Obs: 1 / s = 3600 * 1 / h 
# ---> Obs: 1 Pa  = 0.00750064 mmHg
# ---> Obs: ...   = 1.3595101 ... 

# Pressão Capilar [mmHg]
P_c      = Constant(20.0 * 1.3595101)
# Coeficiente de Filtragem (Lp0 * S/V) [1 / (s mmHg)]
k_f      = Constant(3.6e-7 * 17.4 * 1.3595101)
# Coeficiente de Pressão Osmótica
sigma_0  = Constant(0.91)
# Pressão Oncótica Capilar [mmHg]
pi_c     = Constant(20.0 * 1.3595101)
# Pressão Oncótica Intersticial [mmHg]
pi_i     = Constant(10.0 * 1.3595101)
# Influência do Patógeno na Permeabilidade Hidráulica [cm^3 / (10^10 cell)]
c_bp     = Constant(60)
# Fluxo Linfático Normal [cm / s]
q_0      = Constant(0.0001) 
# Limiar de Fluxo Linfático 
V_max    = Constant(20)
# Aumento na Velocidade do Fluxo [mmHg]
K_m      = Constant(6.5) 
# Pressão Inicial [mmHg]
P_0      = Constant(0.0)
# Expoente (n)
n        = Constant(5)
# Primeiro Parâmetro de Lamé [kPa]
lambda_s = Constant(27.293)
# Módulo de Cisalhamento [kPa]
mu_s     = Constant(3.103)
# Tensor de Mobilidade [cm^2 / (s mmHg)]
mobility_tensor = Constant(2.5e-7 * 1.3595101)

################## Malha ##################

# Malha
# mesh = Mesh("Imunologia.xml")
mesh = Mesh("malha.xml")  

# print(f'Número de células: {mesh.num_cells()}')
# print(f'Número de vértices: {mesh.num_vertices()}')

# Contorno
boundary = MeshFunction("size_t", mesh, 'malha_facet_region.xml')

# boundary_values = np.unique(boundary.array())
# print(boundary_values)

# Geração Aleatória dos Vasos Linfáticos
# Vaso Linfático   -> Id 1
# Demais elementos -> Id 0
np.random.seed(123456)
lymph = np.random.randint(0, mesh.num_cells() - 1, size = int((2.9 / 100) * mesh.num_cells()))

lymph_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

for cell in cells(mesh):
    if cell.index() in lymph:
        lymph_subdomains[cell] = 1

dx = dx(subdomain_data = lymph_subdomains)

################## Espaços de Funções ##################

# Espaço de funções para concentração de patógenos, leucócitos e pressão
P1      = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1, P1])
V       = FunctionSpace(mesh, element)

# Espaço de Funções para deslocamento
U       = VectorFunctionSpace(mesh, 'P', 1)

# Espaço de funções para velocidade
W       = VectorFunctionSpace(mesh, 'P', 2)

# Espaço de funções para porosidade
V_solid = FunctionSpace(mesh, 'P', 1)

################## Funções ##################

# Funções teste para concentração de patógenos, leucócitos e pressão
v_1, v_2, v_3 = TestFunctions(V)

# Concentração de patógenos (u_1), leucócitos (u_2) e pressão (u_3) no passo de tempo atual
u              = Function(V)
u_1, u_2, u_3  = split(u)

# Concentração de patógenos (u_n1), leucócitos (u_n2) e pressão (u_n3) no passo de tempo anterior
u_n              = Function(V)
u_n1, u_n2, u_n3 = split(u_n)

# Função teste para deslocamento
v_4 = TestFunction(U)

# Deslocamento no passo de tempo atual
u_4  = Function(U)
# Deslocamento no passo de tempo anterior
u_n4 = Function(U)

# Função teste para fase sólida
v_solid = TestFunction(V_solid)

# Quimiotaxia Leucocitária
w = Function(W)

# Porosidade no passo de tempo atual
u_solid   = Function(V_solid)
# Porosidade no passo de tempo anterior
u_n_solid = Function(V_solid)

# Velocidade da fase sólida
w_solid  = Function(W)

wf_solid = Function(W)

################## Condição Inicial ##################

# x1 = 6.38
# y1 = 4.98
# x2 = 7.05
# y2 = 3.84
# A = y2 - y1
# B = x1 - x2
# C = x2 * y1 - x1 * y2
# d = abs(A * x[0] + B * x[1] + C) / np.sqrt(A ** 2 + B ** 2)
# if d < 0.04:
#   return True
# else:
#   return False

# u_n1_0 = Expression('abs((384 - 498) * x[0] + (638 - 705) * x[1] + (705 * 498 - 638 * 384)) / sqrt(pow((384 - 498), 2) + pow((638 - 705), 2)) < 4 ? 0.2 : 0.0', element, degree = 2)
u_n1_0 = Expression('abs((3.84 - 4.98) * x[0] + (6.38 - 7.05) * x[1] + (7.05 * 4.98 - 6.38 * 3.84)) / sqrt(pow((3.84 - 4.98), 2) + pow((6.38 - 7.05), 2)) < 0.04 ? 0.2 : 0.0', element, degree = 2)
u_n1   = project(u_n1_0, V.sub(0).collapse())
assign(u_n.sub(0), u_n1)

u_n_solid = interpolate(Constant(0.2), V_solid)

u.assign(u_n)
u_4.assign(u_n4)
u_solid.assign(u_n_solid)

u_n1, u_n2, u_n3 = split(u_n)

################## Condição de Contorno ##################

# boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# boundary_markers.set_all(0)

# class Boundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary

# boundary = Boundary()
# boundary.mark(boundary_markers, 1)

# bc = DirichletBC(U, Constant((0, 0)), boundary_markers, 1)

bc = DirichletBC(U, Constant((0, 0)), boundary, 10)

################## Resolução do Problema ##################

# --> Deslocamento

# Tensor de Deformação de Green-Lagrange Linearizado
def epsilon(u_4):
    return 0.5 * (nabla_grad(u_4) + nabla_grad(u_4).T)

# Tensor de Tensões de Cauchy
def sigma(u_4):
    return lambda_s * nabla_div(u_4) * Identity(d) + 2 * mu_s * epsilon(u_4)

u_4 = TrialFunction(U)

# Dimensão geométrica do espaço u_4 
d = u_4.geometric_dimension()
# Termo Fonte
f = - nabla_grad(u_3)

a = inner(sigma(u_4), epsilon(v_4)) * dx
L = dot(f, v_4) * dx

u_4  = Function(U)
u_n4 = Function(U)

u_n4 = project(Constant((0.0, 0.0)), U)

# --> Concetração de patógenos, leucócitos e pressão
# Fagocitose de patógenos
rb = lambda_nb * u_1 * u_2
# Fonte de patógenos
qb = c_p * u_1

# Morte de leucócitos
rn = lambda_bn * u_1 * u_2 + mu_n * u_2
# Fonte de leucócitos
qn = gamma_n * u_1 * (C_n_max - u_2)

cf = (1.0 + c_bp * u_1)

# Rede de vasos capilares
qc = cf * k_f * ((P_c - u_3) - (sigma_0 / cf) * (pi_c - pi_i))
# Rede de vasos linfáticos
ql = q_0 * (1 + (V_max * (u_3 - P_0) ** n) / (K_m ** n + (u_3 - P_0) ** n)) 

beta = 3.0 / (3.0 * lambda_s + 2.0 * mu_s)

F = (u_1 - u_n1) * v_1 * dx  \
    + (dt / phi_f) * (D_b * dot(grad(u_1), grad(v_1)) * dx - (qb - rb) * v_1 * dx)  \
    + (u_2 - u_n2) * v_2 * dx + (dt / phi_f) * (dot(w, grad(u_2)) * v_2 * dx \
    + D_n * dot(grad(u_2), grad(v_2)) * dx - (qn - rn) * v_2 * dx) \
    + (u_3 - u_n3) * v_3 * dx \
    + (dt / beta) * (mobility_tensor * dot(grad(u_3), grad(v_3)) * dx - qc * v_3 * dx \
    + ql * v_3 * dx(1))

# Fase sólida
F_solid = ((u_solid - u_n_solid)) * v_solid * dx + dt * (dot(u_solid * w_solid, grad(v_solid)) * dx)

vtkfile_u_1     = File('output/C_p.pvd')
vtkfile_u_2     = File('output/C_l.pvd')
vtkfile_u_3     = File('output/P.pvd')
vtkfile_u_4     = File('output/U.pvd')
vtkfile_u_solid = File('output/Phi_f.pvd')
vtkfile_w_solid = File('output/vel.pvd')

t = 0.0

_u_1, _u_2, _u_3 = u.split()
vtkfile_u_1     << (_u_1, t)
vtkfile_u_2     << (_u_2, t)        
vtkfile_u_3     << (_u_3, t)
vtkfile_u_4     << u_4
vtkfile_u_solid << u_solid
vtkfile_w_solid << w_solid

for i in range(num_steps):
    print(f'Step {i} de {num_steps}')
    t += dt  
    
    wf = X_nb * grad(u.sub(0))        
    wf = project(wf, W)
    w.assign(wf)

    solver_parameters = {"newton_solver": {"maximum_iterations": 50,
                                           "relative_tolerance": 1e-6,
                                           "absolute_tolerance": 1e-8}}
        
    print('Resolvendo C_p, C_l e P')
    solve(F == 0, u, solver_parameters = solver_parameters)

    print('Resolvendo U')
    solve(a == L, u_4, bc)

    wf_solid = ((u_4 - u_n4) / dt)
    wf_solid = project(wf_solid, W)
    w_solid.assign(wf_solid)
    
    print('Resolvendo Phi')
    solve(F_solid == 0, u_solid, solver_parameters = solver_parameters)

    _u_1, _u_2, _u_3 = u.split()
    vtkfile_u_1     << (_u_1, t)
    vtkfile_u_2     << (_u_2, t)        
    vtkfile_u_3     << (_u_3, t)
    vtkfile_u_4     << u_4
    vtkfile_u_solid << u_solid
    vtkfile_w_solid << w_solid

    u_n.assign(u)
    u_n4.assign(u_4)
    u_n_solid.assign(u_solid)