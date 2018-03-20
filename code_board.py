"""Supply code to

Theory and implementation of electromagnetic fields and thermomechanical structure interaction for computation of systems under finite deformations

by B. E. Abali and A. F. Queiruga
"""
__author__ = "B. Emek Abali and Alejandro F. Queiruga"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from cbcpost import *
from fenics import *
import numpy
import sympy
from util import *
from afqsfenicsutil import *
from afqsfenicsutil import my_restriction_map
from cbcpost.utils import create_submesh
# from cbcpost.utils import restriction_map
parameters['allow_extrapolation'] = True
#set_log_level(DEBUG)
set_log_level(ERROR)
#
cfg = {
    'fname':'../meshes/board.h5',

    'boundary':3,
    'volt_in':4,
    'volt_out':5,
    'clamp':6,
    'cool_down':7,
    'interface':8,

    'chip':1,
    'board':2,
    'air':9,
    'gel':10,
    'te':11,
    'trace':12,
}

mat_marking = [cfg['air'],cfg['board'],cfg['chip'],cfg['gel'],cfg['te'],cfg['trace']]
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(),cfg['fname'], 'r')
hdf.read(mesh, '/mesh', False)
cells = CellFunction('size_t', mesh)
hdf.read(cells, '/cells')
facets = FacetFunction('size_t', mesh)
hdf.read(facets, '/facets')

info(mesh)

# units: s, A, V, K, m, kg
# N=kg m/s2, J=N m, C=A s, V=J/C, W=J/s=A V, S=A/V, Pa=N/m2, 

nu = 0.1 #in Hz
period = 1./nu
tMax = 2.*period
Dt_float = period/50.
t = 0.
Dt = Constant(Dt_float)
V_amp = 12.

f = Constant((0., 0., 0.)) #N/kg
r = Constant(0.0)
Tref = 300. # K
Tamb = Tref
eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)
null=1E-25 #for numerical reasons it is not zero
h_interface = 1E5 #in J/(s m2 K)

#air
la_a = 0.1
mu_a = 0.1
C_voigt_a = numpy.array([ \
[la_a+2.*mu_a, la_a, la_a, 0, 0, 0],\
[la_a, la_a+2.*mu_a, la_a, 0, 0, 0],\
[la_a, la_a, la_a+2.*mu_a, 0, 0, 0],\
[0, 0, 0, mu_a, 0, 0],\
[0, 0, 0, 0, mu_a, 0],\
[0, 0, 0, 0, 0, mu_a]  ])
dtilde_a = 0.0 #in m/V
alpha_a = 3.43E-3 #in 1/K
varsigma_a = 3E-15 #in S/m
kappa_a = 0.0257 #in W/(K m)
pi_a = null #in V/K
eps_rel_el_a = 1.
chi_ma_a = 0.
mu_rel_ma_a = chi_ma_a + 1.
c_a = 1005. #in J/(kg K)
rho_0_a = 1.205 #in kg/m3


# board epoxy
Ex, Ey, Ez = 25E9, 25E9, 11E9 #in Pa
nu_xy, nu_xz, nu_yz = 0.10, 0.16, 0.16
G_yz, G_zx, G_xy = 3.5E9, 3.5E9, 3.6E9 #in Pa
S_voigt_e = numpy.array([ \
[1./Ex, -nu_xy/Ey, -nu_xz/Ez, 0, 0, 0],\
[-nu_xy/Ey, 1./Ey, -nu_yz/Ez, 0, 0, 0],\
[-nu_xz/Ez, -nu_yz/Ez, 1./Ez, 0, 0, 0],\
[0, 0, 0, 1./G_yz, 0, 0],\
[0, 0, 0, 0, 1./G_zx, 0],\
[0, 0, 0, 0, 0, 1./G_xy]  ])
C_voigt_e = numpy.linalg.inv(S_voigt_e)
dtilde_e = 0.0 #in m/V
alpha11_e, alpha22_e, alpha33_e = 13.1E-6, 13.1E-6, 20.4E-6 #in 1/K
varsigma_e = null #in S/m or in 1/(Ohm m)
kappa_e = 1.3 #in W/(K m)
pi_e = null #in V/K
eps_rel_el_e = 1.0
chi_ma_e = 0.0
mu_rel_ma_e = chi_ma_e + 1.
c_e = 800. #in J/(kg K)
rho_0_e = 2500. #in kg/m3

# IC, chip
E_ic = 250E9 #in Pa
G_ic = 75E9 #in Pa
la_ic = (E_ic-2.*G_ic)*G_ic/(3.*G_ic-E_ic)
mu_ic = G_ic
C_voigt_ic = numpy.array([ \
[la_ic+2.*mu_ic, la_ic, la_ic, 0, 0, 0],\
[la_ic, la_ic+2.*mu_ic, la_ic, 0, 0, 0],\
[la_ic, la_ic, la_ic+2.*mu_ic, 0, 0, 0],\
[0, 0, 0, mu_ic, 0, 0],\
[0, 0, 0, 0, mu_ic, 0],\
[0, 0, 0, 0, 0, mu_ic]  ])
dtilde_ic = 0.0 #in m/V
alpha_ic = 13E-6 #in 1/K
varsigma_ic = 0.1 #in S/m or in 1/(Ohm m)
kappa_ic = 1.3 #in W/(K m)
pi_ic = null #in V/K
eps_rel_el_ic = 1.0
chi_ma_ic = 4.0 # schubert2016
mu_rel_ma_ic = chi_ma_ic + 1.
c_ic = 800. #in J/(kg K)
rho_0_ic = 2500. #in kg/m3

# gel
E_g = 2.5E9 #in Pa
G_g = 0.6E9 #in Pa
la_g = (E_g-2.*G_g)*G_g/(3.*G_g-E_g)
mu_g = G_g
C_voigt_g = numpy.array([ \
[la_g+2.*mu_g, la_g, la_g, 0, 0, 0],\
[la_g, la_g+2.*mu_g, la_g, 0, 0, 0],\
[la_g, la_g, la_g+2.*mu_g, 0, 0, 0],\
[0, 0, 0, mu_g, 0, 0],\
[0, 0, 0, 0, mu_g, 0],\
[0, 0, 0, 0, 0, mu_g]  ])
dtilde_g = 0.0 #in m/V
alpha_g = 13E-6 #in 1/K
varsigma_g = null #in S/m or in 1/(Ohm m)
kappa_g = 1.3 #in W/(K m)
pi_g = null #in V/K
eps_rel_el_g = 1.0
chi_ma_g = 4.0 
mu_rel_ma_g = chi_ma_g + 1.
c_g = 800. #in J/(kg K)
rho_0_g = 2500. #in kg/m3

# thermoelectric ceramic
E_te = 105E9 #in Pa
G_te = 36E9 #in Pa
la_te = (E_te-2.*G_te)*G_te/(3.*G_te-E_te)
mu_te = G_te
C_voigt_te = numpy.array([ \
[la_te+2.*mu_te, la_te, la_te, 0, 0, 0],\
[la_te, la_te+2.*mu_te, la_te, 0, 0, 0],\
[la_te, la_te, la_te+2.*mu_te, 0, 0, 0],\
[0, 0, 0, mu_te, 0, 0],\
[0, 0, 0, 0, mu_te, 0],\
[0, 0, 0, 0, 0, mu_te]  ])
alpha_te = 19E-6 #in 1/K
varsigma_te = null #in S/m or in 1/(Ohm m)
kappa_te = 109. #in W/(K m)
pi_te = 68E-6 #in V/K
eps_rel_el_te = 10.
chi_ma_te = -1E-5 #-16.6E-5 #of Bizmuth not copper
mu_rel_ma_te = chi_ma_te + 1.
c_te = 380. #in J/(kg K)
rho_0_te = 8400. #in kg/m3

# trace, copper
E_c = 105E9 #in Pa
G_c = 36E9 #in Pa
la_c = (E_c-2.*G_c)*G_c/(3.*G_c-E_c)
mu_c = G_c
C_voigt_c = numpy.array([ \
[la_c+2.*mu_c, la_c, la_c, 0, 0, 0],\
[la_c, la_c+2.*mu_c, la_c, 0, 0, 0],\
[la_c, la_c, la_c+2.*mu_c, 0, 0, 0],\
[0, 0, 0, mu_c, 0, 0],\
[0, 0, 0, 0, mu_c, 0],\
[0, 0, 0, 0, 0, mu_c]  ])
alpha_c = 17E-6 #in 1/K
varsigma_c = 100. #in S/m or in 1/(Ohm m)
kappa_c = 109. #in W/(K m)
pi_c = 68E-6 #in V/K
eps_rel_el_c = 10.
chi_ma_c = -1E-5 #-16.6E-5 #of Bizmuth not copper
mu_rel_ma_c = chi_ma_c + 1.
c_c = 380. #in J/(kg K)
rho_0_c = 8400. #in kg/m3


delta = Identity(3)

class Thermomechanics():
	def __init__(self, mesh, cell_ids, facet_ids):
		self.mesh_glob = mesh
		self.cell_ids = cell_ids
		self.facet_ids = facet_ids
		self.mesh_sub = SubMesh2(self.mesh_glob, cell_ids, (cfg['board'],cfg['chip'],cfg['gel'],cfg['te'],cfg['trace']) )

		self.mesh_sub_a = SubMesh2(self.mesh_glob, cell_ids, (cfg['board'],cfg['chip'],cfg['trace']) )
		self.mesh_sub_b = SubMesh2(self.mesh_glob, cell_ids, (cfg['te'], ) )
		self.mesh_sub_c = SubMesh2(self.mesh_glob, cell_ids, (cfg['gel'], ) )

		self.Scalar = FunctionSpace(self.mesh_sub, 'P', 1)
		self.Vector = VectorFunctionSpace(self.mesh_sub, 'P', 1)
		self.Tensor = TensorFunctionSpace(self.mesh_sub, 'P', 1)
		scalar = FiniteElement('P', tetrahedron, 1)
		vector = VectorElement('P', tetrahedron, 1)
		tensor = TensorElement('P', tetrahedron, 1)

		# TM element for u, T
		tm_mixed_element = MixedElement([vector, scalar])
		space_tm_glob = FunctionSpace(self.mesh_glob, tm_mixed_element)
		space_tm_sub  = FunctionSpace(self.mesh_sub, tm_mixed_element)

		# (Dofmap->Dofmap)maps
		# Scalar
		space_scalar_glob = FunctionSpace(self.mesh_glob, scalar)
		space_scalar_sub  = FunctionSpace(self.mesh_sub, scalar)
		self.scalar_sub2glob = my_restriction_map.restriction_map(space_scalar_glob, space_scalar_sub)
		self.scalar_glob2sub = {v:k for k,v in self.scalar_sub2glob.iteritems() }
		self.scalar_sub2glob_lhs = np.empty(len(space_scalar_sub.dofmap().dofs()),dtype=np.intc)
		self.scalar_sub2glob_rhs = np.empty(len(space_scalar_sub.dofmap().dofs()),dtype=np.intc)
		for i,(k,v) in enumerate(self.scalar_sub2glob.iteritems()):
			self.scalar_sub2glob_lhs[i] = v
			self.scalar_sub2glob_rhs[i] = k
		# Vector
		space_vector_glob = FunctionSpace(self.mesh_glob, vector)
		space_vector_sub  = FunctionSpace(self.mesh_sub, vector)
		self.vector_sub2glob = my_restriction_map.restriction_map(space_vector_glob, space_vector_sub)
		self.vector_glob2sub = {v:k for k,v in self.vector_sub2glob.iteritems() }
		self.vector_sub2glob_lhs = np.empty(len(space_vector_sub.dofmap().dofs()),dtype=np.intc)
		self.vector_sub2glob_rhs = np.empty(len(space_vector_sub.dofmap().dofs()),dtype=np.intc)
		for i,(k,v) in enumerate(self.vector_sub2glob.iteritems()):
			self.vector_sub2glob_lhs[i] = v
			self.vector_sub2glob_rhs[i] = k
		# Tensor
		space_tensor_glob = FunctionSpace(self.mesh_glob, tensor)
		space_tensor_sub  = FunctionSpace(self.mesh_sub, tensor)
		self.tensor_sub2glob = my_restriction_map.restriction_map(space_tensor_glob, space_tensor_sub)
		self.tensor_glob2sub = {v:k for k,v in self.tensor_sub2glob.iteritems() }
		self.tensor_sub2glob_lhs = np.empty(len(space_tensor_sub.dofmap().dofs()),dtype=np.intc)
		self.tensor_sub2glob_rhs = np.empty(len(space_tensor_sub.dofmap().dofs()),dtype=np.intc)
		for i,(k,v) in enumerate(self.tensor_sub2glob.iteritems()):
			self.tensor_sub2glob_lhs[i] = v
			self.tensor_sub2glob_rhs[i] = k

		# Move the material and BC labeling onto the sub mesh
		sub_nodes = self.mesh_sub.data().array('parent_vertex_indices',0)
		facet_local_to_global_map = Mesh2Submesh_FacetMap(self.mesh_glob, self.mesh_sub)
		sub_facets = FacetFunction('size_t', self.mesh_sub)
		sub_facets.array()[:] = facet_ids.array()[facet_local_to_global_map]
		cell_local_to_global_map = Mesh2Submesh_CellMap(self.mesh_glob, self.mesh_sub)
		sub_cells = CellFunction('size_t', self.mesh_sub)
		sub_cells.array()[:] = cell_ids.array()[ cell_local_to_global_map ]
		self.sub_nodes = sub_nodes
		self.sub_cells = sub_cells
		self.sub_facets = sub_facets

		self.dI = Measure('dS', domain=self.mesh_sub, subdomain_data=self.sub_facets, metadata={'quadrature_degree': 2})
		self.dA = Measure('ds', domain=self.mesh_sub, subdomain_data=self.sub_facets, metadata={'quadrature_degree': 2})
		self.dV = Measure('dx', domain=self.mesh_sub, subdomain_data=self.sub_cells, metadata={'quadrature_degree': 2})

		bc = []
		bc.append( DirichletBC(space_tm_sub.sub(0), Constant((0.0, 0.0, 0.0)), self.sub_facets, cfg['clamp']) )
		bc.append( DirichletBC(space_tm_sub.sub(1), Tref, self.sub_facets, cfg['cool_down']) )

		self.bcs = bc
		self.dunkn = TrialFunction(space_tm_sub)
		self.test = TestFunction(space_tm_sub)
		self.unkn   = Function(space_tm_sub)
		self.unkn0  = Function(space_tm_sub)
		self.unkn00 = Function(space_tm_sub)

		self.q_sub = Function(self.Scalar)
		self.q_fr_sub = Function(self.Scalar)
		self.E_sub = Function(self.Vector)
		self.E0_sub = Function(self.Vector)
		self.B_sub = Function(self.Vector)
		self.B0_sub = Function(self.Vector)
		self.gradu_sub = Function(self.Tensor)
		self.gradu0_sub = Function(self.Tensor)
		self.detF_sub = Function(self.Scalar)
		self.vel_sub = Function(self.Vector)
		self.T_sub = Function(self.Scalar)
		self.T0_sub = Function(self.Scalar)

		self.P_sub = Function(self.Vector)
		self.P0_sub = Function(self.Vector)
		self.MM_sub = Function(self.Vector)
		
		unkn_init = Expression(('0.','0.','0.', 'Tref'), degree=1, Tref=Tref)
		self.unkn00 = interpolate(unkn_init,space_tm_sub)
		self.unkn0.assign(self.unkn00)
		self.unkn.assign(self.unkn0)

		# Now assign material properties
		i,j,k,l,o = indices(5)
	

		C11 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[0,0], C_voigt_e[0,0], C_voigt_ic[0,0], C_voigt_g[0,0], C_voigt_te[0,0], C_voigt_c[0,0] ], mat_marking)
		C12 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[0,1], C_voigt_e[0,1], C_voigt_ic[0,1], C_voigt_g[0,1], C_voigt_te[0,1], C_voigt_c[0,1] ], mat_marking)
		C13 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[0,2], C_voigt_e[0,2], C_voigt_ic[0,2], C_voigt_g[0,2], C_voigt_te[0,2], C_voigt_c[0,2] ], mat_marking)
		C22 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[1,1], C_voigt_e[1,1], C_voigt_ic[1,1], C_voigt_g[1,1], C_voigt_te[1,1], C_voigt_c[1,1] ], mat_marking)
		C33 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[2,2], C_voigt_e[2,2], C_voigt_ic[2,2], C_voigt_g[2,2], C_voigt_te[2,2], C_voigt_c[2,2] ], mat_marking)
		C44 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [C_voigt_a[3,3], C_voigt_e[3,3], C_voigt_ic[3,3], C_voigt_g[3,3], C_voigt_te[3,3], C_voigt_c[3,3] ], mat_marking)
		self.C = VoigtToTensorRank4(A11=C11,A12=C12,A13=C13,A22=C22,A23=C13,A33=C33,A44=C44,A55=C44,A66=C44)
		
		self.Stilde = VoigtToTensorRank3() 
		self.Ttilde = VoigtToTensorRank3()
		self.Rtilde = VoigtToTensorRank2()

		eps_rel_el11 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [eps_rel_el_a, eps_rel_el_e, eps_rel_el_ic, eps_rel_el_g, eps_rel_el_te, eps_rel_el_c], mat_marking)
		eps_rel_el = VoigtToTensorRank2(A11=eps_rel_el11, A22=eps_rel_el11, A33=eps_rel_el11)
		self.chi_el = eps_rel_el - delta

		mu_rel_mag11 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [mu_rel_ma_a, mu_rel_ma_e, mu_rel_ma_ic, mu_rel_ma_g, mu_rel_ma_te, mu_rel_ma_c], mat_marking)
		mu_rel_mag = VoigtToTensorRank2(A11=mu_rel_mag11, A22=mu_rel_mag11, A33=mu_rel_mag11)
		self.chi_mag = mu_rel_mag - delta
		self.mu_mag = mu_rel_mag*mu_0

		alpha11 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [alpha_a, alpha11_e, alpha_ic, alpha_g, alpha_te, alpha_c], mat_marking)
		alpha22 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [alpha_a, alpha22_e, alpha_ic, alpha_g, alpha_te, alpha_c], mat_marking)
		alpha33 = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [alpha_a, alpha33_e, alpha_ic, alpha_g, alpha_te, alpha_c], mat_marking)
		self.alpha = VoigtToTensorRank2(A11=alpha11, A22=alpha22, A33=alpha33)

		self.varsigma = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [varsigma_a, varsigma_e, varsigma_ic, varsigma_g, varsigma_te, varsigma_c], mat_marking)
		self.kappa    = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [kappa_a, kappa_e, kappa_ic, kappa_g, kappa_te, kappa_c], mat_marking)
		self.pi       = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [pi_a, pi_e, pi_ic, pi_g, pi_te, pi_c], mat_marking)
		self.c        = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [c_a, c_e, c_ic, c_g, c_te, c_c], mat_marking)
		self.rho_0    = AssignMaterialCoefficients(self.mesh_sub, self.sub_cells, [rho_0_a, rho_0_e, rho_0_ic, rho_0_g, rho_0_te, rho_0_c], mat_marking)
		self.v_0      = 1./self.rho_0
		
	def make_forms(self):
		# All of these are defined on the self.mesh_TM
		E, E0 = self.E_sub, self.E0_sub
		B, B0 = self.B_sub, self.B0_sub
		q, q_fr = self.q_sub, self.q_fr_sub

		del_u, del_T = split(self.test)
		u, T = split(self.unkn)
		u0, T0 = split(self.unkn0)
		u00, T00 = split(self.unkn00)
		
		dI, dA, dV = self.dI, self.dA, self.dV
		N = FacetNormal(self.mesh_sub)

		C,Ttilde,alpha,Stilde,Rtilde,chi_el,chi_mag,mu_mag,varsigma,kappa,pi,c,rho_0,v_0 = self.C,self.Ttilde,self.alpha,self.Stilde,self.Rtilde,self.chi_el,self.chi_mag,self.mu_mag,self.varsigma,self.kappa,self.pi,self.c,self.rho_0,self.v_0

		v = (u-u0)/Dt
		gradu = as_tensor( u[i].dx(j) , (i,j))
		gradu0 = as_tensor( u0[i].dx(j) , (i,j))
		F = gradu + delta
		F0 = gradu0 + delta
		EE = as_tensor(E[i] + epsilon[i,j,k]*v[j]*B[k], (i,))

		D = eps_0*E
		D0 = eps_0*E0
		H = 1./mu_0*B
		P = as_tensor( 1./det(F)*Ttilde[i,k,l]*(-alpha[k,l]*(T-Tref) + gradu[k,l] ) + eps_0*chi_el[i,k]*E[k] + (2.-1./det(F))*Rtilde[k,i]*B[k] , (i,))
		P0 = as_tensor( 1./det(F0)*Ttilde[i,k,l]*(-alpha[k,l]*(T0-Tref) + gradu0[k,l] ) + eps_0*chi_el[i,k]*E0[k] + (2.-1./det(F0))*Rtilde[k,i]*B0[k] , (i,))
		mD = D + P
		mD0 = D0 + P0
		MM = as_tensor( 1./det(F)*Stilde[i,k,l]*(-alpha[k,l]*(T-Tref) + gradu[k,l] ) + Rtilde[i,k]*E[k] + (2.-1./det(F))*inv(mu_mag)[i,j]*chi_mag[j,k]*B[k] , (i,))
		q = as_tensor( D[i].dx(i) , ())
		q_fr = as_tensor( mD[i].dx(i) , ())
		JJ_fr = as_tensor( varsigma*pi*T.dx(i) + varsigma*EE[i], (i,))
		J_fr = as_tensor( JJ_fr[i] + q_fr*v[i] , (i,))
		J = as_tensor( J_fr[i] + (P-P0)[i]/Dt + epsilon[i,j,k]*MM[k].dx(j) , (i,))
		f_EM = as_tensor(det(F)*q*E[i] + det(F)*epsilon[i,j,k]*J[j]*B[k] - det(F)*epsilon[i,j,k]*(P-P0)[j]/Dt*B[k] - det(F)*epsilon[i,j,k]*P[j]*(B-B0)[k]/Dt , (i,))

		eta = as_tensor( c*ln(T/Tref) + v_0*C[l,k,i,j]*alpha[i,j]*u[k].dx(l) - v_0*Ttilde[k,i,j]*alpha[i,j]*E[k] - v_0*(2.-1./det(F))*Stilde[k,i,j]*alpha[i,j]*B[k] , ())
		eta0 = as_tensor( c*ln(T0/Tref) + v_0*C[l,k,i,j]*alpha[i,j]*u0[k].dx(l) - v_0*Ttilde[k,i,j]*alpha[i,j]*E0[k] - v_0*(2.-1./det(F0))*Stilde[k,i,j]*alpha[i,j]*B0[k] , ())


		NN = as_tensor( C[j,i,k,l]*(-alpha[k,l]*(T-Tref)+u[k].dx(l)) - Ttilde[k,i,j]*E[k] - (2.-1./det(F))*Stilde[k,i,j]*B[k] , (i,j))
		sigma = as_tensor( 1./det(F)*F[j,k]*NN[k,i] + P[j]*E[i] - MM[i]*B[j] , (j,i))
		Q = as_tensor(-kappa*T.dx(i) + varsigma*pi*T*det(F)*EE[i], (i,))

		F_u = (rho_0*(u-2.*u0+u00)[i]/Dt/Dt*del_u[i] + det(F)*inv(F)[k,j]*sigma[j,i] *del_u[i].dx(k) - (rho_0*f+f_EM)[i]*del_u[i] )*dV #- self.traction[i]*del_u[i]*dA(cfg['load'])
		F_T = ( rho_0*(eta-eta0)*del_T - Dt*Q[i]/T*del_T.dx(i) - Dt*rho_0*r/T*del_T + Dt*Q[j]/T/T*T.dx(j)*del_T - Dt*det(F)/T*EE[i]*JJ_fr[i]*del_T )*dV + Dt*h_interface*(T-Tref)/T*del_T*(dA(cfg['interface'])+dA(cfg['clamp'])+dA(cfg['volt_in'])+dA(cfg['volt_out']) )

		self.Form = F_u + F_T
		self.Gain = derivative(self.Form, self.unkn, self.dunkn)
        
	def solve_tm(self):
		solve(self.Form == 0, self.unkn, self.bcs, J=self.Gain, \
			#solver_parameters={"newton_solver":{"linear_solver": "cg","preconditioner": "petsc_amg", "absolute_tolerance": 1E-4, "relative_tolerance": 1E-3} }, \
			solver_parameters={"newton_solver":{"linear_solver": "mumps", "absolute_tolerance": 1E-6, "relative_tolerance": 1E-3} }, \
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )


		Ttilde,alpha,Stilde,Rtilde,chi_el,chi_mag,mu_mag,varsigma,kappa,pi,c,rho_0,v_0 = self.Ttilde,self.alpha,self.Stilde,self.Rtilde,self.chi_el,self.chi_mag,self.mu_mag,self.varsigma,self.kappa,self.pi,self.c,self.rho_0,self.v_0


		E, E0 = self.E_sub, self.E0_sub
		B, B0 = self.B_sub, self.B0_sub

		u, T = split(self.unkn)
		u0, T0 = split(self.unkn0)
		v = (u-u0)/Dt
		gradu = as_tensor( u[i].dx(j) , (i,j))
		gradu0 = as_tensor( u0[i].dx(j) , (i,j))
		F = gradu + delta
		F0 = gradu0 + delta
		EE = as_tensor(E[i] + epsilon[i,j,k]*v[j]*B[k], (i,))

		D = eps_0*E
		D0 = eps_0*E0
		H = 1./mu_0*B
		P = as_tensor( 1./det(F)*Ttilde[i,k,l]*(-alpha[k,l]*(T-Tref) + gradu[k,l] ) + eps_0*chi_el[i,k]*E[k] + (2.-1./det(F))*Rtilde[k,i]*B[k] , (i,))
		P0 = as_tensor( 1./det(F0)*Ttilde[i,k,l]*(-alpha[k,l]*(T0-Tref) + gradu0[k,l] ) + eps_0*chi_el[i,k]*E0[k] + (2.-1./det(F0))*Rtilde[k,i]*B0[k] , (i,))
		MM = as_tensor( 1./det(F)*Stilde[i,k,l]*(-alpha[k,l]*(T-Tref) + gradu[k,l] ) + Rtilde[i,k]*E[k] + (2.-1./det(F))*inv(mu_mag)[i,j]*chi_mag[j,k]*B[k] , (i,))

		solvertype="mumps"
		self.vel_sub = project(v, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.T_sub = project(T, self.Scalar, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.T0_sub = project(T0, self.Scalar, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.P_sub = project(P, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.P0_sub = project(P0, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.MM_sub = project(MM, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})

		#print 'max Temperature ', max(self.T_sub.vector().array()[:])


	def output(self, fname):
		u_, T_ = self.unkn.split(deepcopy=True)
		#print  ' max displacement ', max(u_.vector().array())
		#write_vtk_f(fname.format(onum),self.mesh_sub,
		#	    nodefunctions={'u':u_, 'T':T_, 'phi':self.phi_sub, 'E':self.E_sub, 'B':self.B_sub, 'P':self.P_sub, 'M':self.MM_sub}
		#	    )
		write_vtk_f(fname.format(onum),self.mesh_sub,
			    nodefunctions={'u':u_, 'T':T_, 'E':self.E_sub, 'B':self.B_sub, 'P':self.P_sub, 'M':self.MM_sub}
			    )

	def push_tm_to_global_mesh(self, vel_glob, T_glob, T0_glob, P_glob, P0_glob, MM_glob):
		# Use them to do index reassignment
		def push_scalar(u,ub):
		  for l,r in zip(self.scalar_sub2glob_lhs,self.scalar_sub2glob_rhs):
			u.vector()[int(l)] = ub.vector().array()[int(r)]
		def push_vector(u,ub):
		  for l,r in zip(self.vector_sub2glob_lhs,self.vector_sub2glob_rhs):
			u.vector()[int(l)] = ub.vector().array()[int(r)]
		def push_tensor(u,ub):
		  for l,r in zip(self.tensor_sub2glob_lhs,self.tensor_sub2glob_rhs):
			u.vector()[int(l)] = ub.vector().array()[int(r)]

		push_vector(vel_glob, self.vel_sub)
		push_scalar(T_glob, self.T_sub)
		push_scalar(T0_glob, self.T0_sub)
		push_vector(P_glob, self.P_sub)
		push_vector(P0_glob, self.P0_sub)
		push_vector(MM_glob, self.MM_sub)
		
	def pull_em_from_global_mesh(self, E_glob, E0_glob, B_glob, B0_glob):
		# Use them to do index reassignment
		def push_scalar( u,ub ):
		  for l,r in zip(self.scalar_sub2glob_lhs,self.scalar_sub2glob_rhs):
			ub.vector()[int(r)] = u.vector().array()[int(l)]
		def push_vector( u,ub ):
		  for l,r in zip(self.vector_sub2glob_lhs,self.vector_sub2glob_rhs):
			ub.vector()[int(r)] = u.vector().array()[int(l)]

		push_vector(E_glob,  self.E_sub)
		push_vector(E0_glob,  self.E0_sub)
		push_vector(B_glob,  self.B_sub)
		push_vector(B0_glob,  self.B0_sub)


	def morph_mesh(self, w_em, mesh_glob_em):
		bmesh = BoundaryMesh(mesh_glob_em,'exterior')
		fixed_nodes = bmesh.entity_map(0).array()
		Du = self.unkn.split(deepcopy=True)[0]
		u0 = self.unkn0.split(deepcopy=True)[0]
		Du.vector()[:] -= u0.vector()[:]
		mesh_new = Mesh(mesh_glob_em)
		morph_fenics(mesh_new, self.sub_nodes, Du, other_fix = fixed_nodes)
		return mesh_new

class Electromagnetism():

	def __init__(self, mesh, cell_ids, facet_ids):
		self.mesh = mesh
		self.cell_ids = cell_ids
		self.facet_ids = facet_ids

		self.di = Measure('dS', domain=self.mesh, subdomain_data=self.facet_ids, metadata={'quadrature_degree': 2})
		self.da = Measure('ds', domain=self.mesh, subdomain_data=self.facet_ids, metadata={'quadrature_degree': 2})
		self.dv = Measure('dx', domain=self.mesh, subdomain_data=self.cell_ids, metadata={'quadrature_degree': 2})

		self.Scalar = FunctionSpace(self.mesh, 'P', 1)
		self.Vector = VectorFunctionSpace(self.mesh, 'P', 1)
		self.Tensor = TensorFunctionSpace(self.mesh, 'P', 1)
		scalar = FiniteElement('P', tetrahedron, 1)
		vector = VectorElement('P', tetrahedron, 1)
		tensor = TensorElement('P', tetrahedron, 1)
		self.em_mixed_element = MixedElement([scalar, vector])
		self.space_em_glob = FunctionSpace(self.mesh, self.em_mixed_element)
		
		bc = []
		self.applied_Vin = Expression('amp*sin(2.0*pi*freq*t)' , degree=1, amp=V_amp/2., freq=nu, t=0, domain=self.mesh)
		self.applied_Vout = Expression('amp*sin(2.0*pi*freq*t)' , degree=1, amp=-V_amp/2., freq=nu, t=0, domain=self.mesh)
		bc.append( DirichletBC(self.space_em_glob, Constant([0,0,0,0]), self.facet_ids, cfg['boundary']) ) 
		bc.append( DirichletBC(self.space_em_glob.sub(0), self.applied_Vin, self.facet_ids, cfg['volt_in']) ) 
		bc.append( DirichletBC(self.space_em_glob.sub(0), self.applied_Vout, self.facet_ids, cfg['volt_out']) ) 

		self.bcs = bc
		self.dunkn = TrialFunction(self.space_em_glob)
		self.test = TestFunction(self.space_em_glob)
		self.unkn = Function(self.space_em_glob)
		self.unkn0 = Function(self.space_em_glob)
		self.unkn00 = Function(self.space_em_glob)

		unkn_init = Expression(('0.', '0.','0.','0.'), degree=1)
		self.unkn00 = interpolate(unkn_init,self.space_em_glob)
		self.unkn0.assign(self.unkn00)
		self.unkn.assign(self.unkn0)

		self.w = Function(self.Vector)

		self.E_glob = Function(self.Vector)
		self.E0_glob = Function(self.Vector)
		self.B_glob = Function(self.Vector)
		self.B0_glob = Function(self.Vector)

		self.P_glob = Function(self.Vector)
		self.P0_glob = Function(self.Vector)
		self.MM_glob = Function(self.Vector)

		self.vel_glob = Function(self.Vector)
		self.T_glob = Function(self.Scalar)
		self.T0_glob = Function(self.Scalar)

		i,j,k,l,o = indices(5)

		self.varsigma = AssignMaterialCoefficients(self.mesh, self.cell_ids, [varsigma_a, varsigma_e, varsigma_ic, varsigma_g, varsigma_te, varsigma_c], mat_marking)
		self.pi       = AssignMaterialCoefficients(self.mesh, self.cell_ids, [pi_a, pi_e, pi_ic, pi_g, pi_te, pi_c], mat_marking)

		
	def make_forms(self):
		n = FacetNormal(self.mesh)
		di,da,dv = self.di,self.da,self.dv

		# All of these are on the global mesh
		del_phi, del_A = split(self.test)
		# These get touched by pull
		phi, A = split(self.unkn)
		phi0, A0 = split(self.unkn0)
		phi00, A00 = split(self.unkn00)

		# These get touched by push
		T, T0 = self.T_glob, self.T0_glob
		vel = self.vel_glob
		P, P0 = self.P_glob, self.P0_glob
		MM = self.MM_glob

		varsigma, pi = self.varsigma, self.pi

		E = -grad(phi) - (A-A0)/Dt
		E0 = -grad(phi0) - (A0-A00)/Dt
		B = as_tensor(epsilon[i,j,k]*A[k].dx(j) , (i,))
		EE = as_tensor(E[i] + epsilon[i,j,k]*vel[j]*B[k], (i,))

		D = eps_0*E
		D0 = eps_0*E0
		H = 1./mu_0*B
		mD = D + P
		q_fr = as_tensor( mD[j].dx(j) , ())
		JJ_fr = as_tensor( varsigma*pi*T.dx(i) + varsigma*EE[i], (i,))
		J_fr = as_tensor( JJ_fr[i] + q_fr*vel[i] , (i,))
		J = as_tensor( J_fr[i] + (P-P0)[i]/Dt + epsilon[i,j,k]*MM[k].dx(j) , (i,))

		F_phi = ( -(D-D0)[i] - Dt*J[i]  )*del_phi.dx(i) *dv 
		F_A = (-eps_0*(A-2.*A0+A00)[j]/Dt/Dt*del_A[j] - 1./mu_0*A[j].dx(k)*del_A[j].dx(k) + J_fr[j]*del_A[j] + (P-P0)[j]/Dt*del_A[j] - epsilon[j,k,i]*MM[i]*del_A[j].dx(k)) * dv

		self.Form =  F_phi + F_A
		self.Gain = derivative(self.Form, self.unkn, self.dunkn)
        
	def solve_em(self):

		solve(self.Form == 0, self.unkn, self.bcs, J=self.Gain, \
			#solver_parameters={"newton_solver":{"linear_solver": "cg", "preconditioner": "petsc_amg", "absolute_tolerance": 1E-4, "relative_tolerance": 1e-3} }, \
			solver_parameters={"newton_solver":{"linear_solver": "mumps", "absolute_tolerance": 1E-6, "relative_tolerance": 1E-4} }, \
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

		phi, A = self.unkn.split(deepcopy=True)
		phi0, A0 = self.unkn0.split(deepcopy=True)
		phi00, A00 = self.unkn00.split(deepcopy=True)
		E = -grad(phi) - (A-A0)/Dt
		E0 = -grad(phi0) - (A0-A00)/Dt
		B = as_tensor(epsilon[i,j,k]*A[k].dx(j) , (i,))
		B0 = as_tensor(epsilon[i,j,k]*A0[k].dx(j) , (i,))
		
		solvertype="mumps"
		self.E_glob = project(E, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.E0_glob = project(E0, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.B_glob = project(B, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.B0_glob = project(B0, self.Vector, solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})

    
	def output(self, fname):
		phi_, A_ = self.unkn.split(deepcopy=True)
		write_vtk_f(fname.format(onum),self.mesh,
			    nodefunctions={'phi':phi_, 'A':A_, 'B':self.B_glob, 'E':self.E_glob }
			    )

	def map_fields_to_new_mesh(self, mesh_new):
		# new_mesh was just spat at us from TM_prob. We now want our fields to be projected
		mesh_old = Mesh(self.mesh)
		space_em_glob_old = FunctionSpace(mesh_old, self.em_mixed_element)
		Vector_old = VectorFunctionSpace(mesh_old, 'P', 1)
		Scalar_old = FunctionSpace(mesh_old, 'P', 1)

		em_unkn_old = Function(space_em_glob_old)
		em_unkn0_old = Function(space_em_glob_old)
		em_unkn00_old = Function(space_em_glob_old)
		E_glob_old = Function(Vector_old)
		E0_glob_old = Function(Vector_old)
		B_glob_old = Function(Vector_old)
		B0_glob_old = Function(Vector_old)
		P_glob_old = Function(Vector_old)
		P0_glob_old = Function(Vector_old)
		MM_glob_old = Function(Vector_old)

		em_unkn_old.vector()[:] = self.unkn.vector().array()
		em_unkn0_old.vector()[:] = self.unkn0.vector().array()
		em_unkn00_old.vector()[:] = self.unkn00.vector().array()
		E_glob_old.vector()[:] = self.E_glob.vector().array()
		E0_glob_old.vector()[:] = self.E0_glob.vector().array()
		B_glob_old.vector()[:] = self.B_glob.vector().array()
		B0_glob_old.vector()[:] = self.B0_glob.vector().array()
		P_glob_old.vector()[:] = self.P_glob.vector().array()
		P0_glob_old.vector()[:] = self.P0_glob.vector().array()
		MM_glob_old.vector()[:] = self.MM_glob.vector().array()

		solvertype="mumps"
		self.mesh.coordinates()[:] = mesh_new.coordinates()[:]
		self.unkn = project(em_unkn_old , self.space_em_glob , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.unkn0 = project(em_unkn0_old ,self.space_em_glob , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.unkn00 = project(em_unkn00_old ,self.space_em_glob , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.E_glob = project(E_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.E0_glob = project(E0_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.B_glob = project(B_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.B0_glob = project(B0_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.P_glob = project(P_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.P0_glob = project(P0_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})
		self.MM_glob = project(MM_glob_old , self.Vector , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
				              "representation": "uflacs",
				              "quadrature_degree": 2})

		#print 'max fields ', max(self.unkn.vector().array()), max(self.E_glob.vector().array()), max(self.B_glob.vector().array()), max(self.P_glob.vector().array()), max(self.MM_glob.vector().array())


TM_prob = Thermomechanics(mesh, cells, facets)
EM_prob = Electromagnetism(mesh, cells, facets)

TM_prob.make_forms()
EM_prob.make_forms()

pwd='board/'
onum = 0
data1= []
def output(t):
	global onum
	EM_prob.output(pwd+'em_{0}.vtk'.format(onum))
	TM_prob.output(pwd+'tm_{0}.vtk'.format(onum))
	onum += 1
	phi_ = EM_prob.unkn.split(deepcopy=True)[0]
	pot_diff = phi_([0,0,0.001]) - phi_([0,0,0.002])
	print 'potential difference: ', pot_diff
	data1.append( [ t, pot_diff,  ])
	numpy.savetxt( pwd+'_time_potdiff.csv', numpy.array(data1) , delimiter=';' )
	

output(t)
tic()
while t<tMax:
	t += Dt_float
	time_emerged = "%.2f" % (toc()/60.)
	print "time: ", t, " computation lasted ", time_emerged, " min" 

	print "Solving Mech... "
	TM_prob.pull_em_from_global_mesh( EM_prob.E_glob, EM_prob.E0_glob, EM_prob.B_glob, EM_prob.B0_glob )
	TM_prob.unkn00.assign(TM_prob.unkn0)
	TM_prob.unkn0.assign(TM_prob.unkn)
	#if t<= 0. : TM_prob.traction.time = t
	#else : TM_prob.traction.time = 0.
	TM_prob.solve_tm() # Only works on submesh
	TM_prob.push_tm_to_global_mesh( EM_prob.vel_glob, EM_prob.T_glob, EM_prob.T0_glob, EM_prob.P_glob, EM_prob.P0_glob, EM_prob.MM_glob ) # This changes the fields in EM_prob

	mesh_new = TM_prob.morph_mesh(EM_prob.w, EM_prob.mesh)
	
	print "Solving EM... "
	EM_prob.map_fields_to_new_mesh(mesh_new)
	EM_prob.unkn00.assign(EM_prob.unkn0)
	EM_prob.unkn0.assign(EM_prob.unkn)
	#if t>0. : EM_prob.applied_A.t = t
	#else : EM_prob.applied_A.t = 0.
	EM_prob.applied_Vin.t = t
	EM_prob.applied_Vout.t = t
	EM_prob.solve_em() # Only works on global mesh

	# Output
	output(t)

