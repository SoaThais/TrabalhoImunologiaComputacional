from fenics import *
from ufl import nabla_grad, nabla_div
import numpy as np

# Tempo total de simulação
end_time = 5.0   
# Número de passos de tempo          
num_steps = 500         
# Tamanho do passo de tempo
dt = end_time / num_steps

################## Parâmetros ##################

# Porosidade
phi_f     = Constant(0.2)             
# Coeficiente de Difusão do Patógeno
D_b       = Constant(0.00005)
# Taxa de Reprodução do Patógeno
c_p       = Constant(0.15)
# Taxa de Fagocitose
lambda_nb = Constant(1.8)

# Coeficiente de Difusão do Leucócito
D_n       = Constant(0.00005)
# Taxa de Quimiotaxia
X_nb      = Constant(0.0001)
# Taxa de Apoptose Induzida
lambda_bn = Constant(0.1)
# Permeabilidade Capilar dos Leucócitos
gamma_n   = Constant(0.1)
# Concentração dos Leucócitos no Sangue
C_n_max   = Constant(0.55)
# Taxa de Apoptose
mu_n      = Constant(0.2)

# ---> Obs: 1mmHg = 0.133322kPa

# Pressão Capilar
P_c      = Constant(20.0)
# Permeabilidade Hidráulica
L_p_0    = Constant(3.6 * 10 ** (-8))
# Coeficiente de Filtragem 
k_f      = Constant(626.4)
# Coeficiente de Pressão Osmótica
sigma_0  = Constant(0.91)
# Pressão Oncótica Capilar
pi_c     = Constant(20.0)
# Pressão Oncótica Intersticial
pi_i     = Constant(10.0)
# Influência do Patógeno na Permeabilidade Hidráulica
c_bp     = Constant(60)
# Fluxo Linfático Normal
q_0      = Constant(0.0001)
# Limiar de Fluxo Linfático
V_max    = Constant(20.0)
# Aumento na Velocidade do Fluxo
K_m      = Constant(6.5)
# Pressão Inicial
P_0      = Constant(0.0)
# Expoente (n)
n        = Constant(5.0)
# Primeiro Parâmetro de Lamé
lambda_s = Constant(27.293)
# Módulo de Cisalhamento
mu_s     = Constant(3.103)
# Tensor de Mobilidade
mobility_tensor = Constant(2.5e-7)

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

# Velocidade da fase sólida
w_solid = Function(W)

# Quimiotaxia Leucocitária
w = Function(W)

# Porosidade no passo de tempo atual
u_solid   = Function(V_solid)
# Porosidade no passo de tempo anterior
u_n_solid = Function(V_solid)

################## Condição Inicial ##################

# x1 = 638
# y1 = 498
# x2 = 705
# y2 = 384
# A = y2 - y1
# B = x1 - x2
# C = x2 * y1 - x1 * y2
# d = abs(A * x[0] + B * x[1] + C) / np.sqrt(A ** 2 + B ** 2)
# if d < 4:
#   return True
# else:
#   return False

u_n1_0 = Expression('abs((384 - 498) * x[0] + (638 - 705) * x[1] + (705 * 498 - 638 * 384)) / sqrt(pow((384 - 498), 2) + pow((638 - 705), 2)) < 4 ? 0.2 : 0.0', element, degree = 2)
u_n1   = project(u_n1_0, V.sub(0).collapse())
assign(u_n.sub(0), u_n1)

u.assign(u_n)
u_4.assign(u_n4)
u_solid.assign(u_n_solid)

# u_n_solid = interpolate(Constant(0.2), V_solid)

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
ql = - q_0 * (1 + (V_max * (u_3 - P_0) ** n) / (K_m ** n + (u_3 - P_0) ** n)) 

beta = 3.0 / (3.0 * lambda_s + 2.0 * mu_s)

F = (u_1 - u_n1) * v_1 * dx  \
    + (dt / phi_f) * (dot(D_b * grad(u_1), grad(v_1)) * dx - (qb - rb) * v_1 * dx)  \
    + (u_2 - u_n2) * v_2 * dx + (dt / phi_f) * (dot(grad(X_nb * u_2), w) * v_2 * dx \
    + dot(D_n * grad(u_2), grad(v_2)) * dx - (qn - rn) * v_2 * dx) \
    + (u_3 - u_n3) * v_3 * dx \
    + (dt / beta) * (dot(mobility_tensor * grad(u_3), grad(v_3)) * dx + qc * v_3 * dx \
    + ql * v_3 * dx(1))

# SUPG (Streamline Upwind Petrov-Galerkin)
res   = u_2 - u_n2 + (dt / phi_f) * (dot(grad(X_nb * u_2), w) - div(D_n * grad(u_2)) - (-rn  + qn))
h     = CellDiameter(mesh)
vnorm = sqrt(dot(w, w) + 1e-10)
F    += (h / (2.0 * vnorm)) * dot(w, grad(v_2)) * res * dx

dt_c = Constant(dt)

# Fase sólida
F_solid = ((u_solid - u_n_solid)) * v_solid * dx - dt_c * (dot(u_solid * w_solid, grad(v_solid)) * dx)

vtkfile_u_1     = File('edema_mecanico_lymph_porosity_v2/C_p.pvd')
vtkfile_u_2     = File('edema_mecanico_lymph_porosity_v2/C_l.pvd')
vtkfile_u_3     = File('edema_mecanico_lymph_porosity_v2/P.pvd')
vtkfile_u_4     = File('edema_mecanico_lymph_porosity_v2/U.pvd')
vtkfile_u_solid = File('edema_mecanico_lymph_porosity_v2/Phi_f.pvd')

t = 0.0

_u_n1, _u_n2, _u_n3 = u_n.split()
vtkfile_u_1     << (_u_n1, t)
vtkfile_u_2     << (_u_n2, t)        
vtkfile_u_3     << (_u_n3, t)
vtkfile_u_4     << u_n4
vtkfile_u_solid << u_solid

for i in range(num_steps):
    print(f'Step {i} de {num_steps}')
    t += dt  
    
    wf = grad(u.sub(0))        
    wf = project(wf, W)
    w.assign(wf)

    solver_parameters = {"newton_solver": {"maximum_iterations": 50,
                                           "relative_tolerance": 1e-6,
                                           "absolute_tolerance": 1e-8}}
        
    print('Resolvendo C_p, C_l e P')
    solve(F == 0, u, solver_parameters = solver_parameters)

    print('Resolvendo U')
    solve(a == L, u_4, bc)
    
    print('Resolvendo Phi')
    solve(F_solid == 0, u_solid)

    _u_1, _u_2, _u_3 = u.split()
    vtkfile_u_1     << (_u_1, t)
    vtkfile_u_2     << (_u_2, t)        
    vtkfile_u_3     << (_u_3, t)
    vtkfile_u_4     << u_4
    vtkfile_u_solid << u_solid
        
    u_n.assign(u)
    u_n4.assign(u_4)
    u_n_solid.assign(u_solid)