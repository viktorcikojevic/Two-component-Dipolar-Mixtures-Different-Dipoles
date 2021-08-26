#!/usr/bin/python
import numpy as np
#from numpy import linalg
import math, sys, os, time
'''
DFT code by Viktor Cikojevic. In case you find any bugs please report them here: cikojevic.viktor@gmail.com
'''
print("SIM. started at ", time.ctime())
start_time = time.time()

#units: https://en.wikipedia.org/wiki/Hartree_atomic_units
mu_m = 1.E-06 / (0.529 * 1.E-10)
amu = 1822.88849
bohr_magneton = 1.

mass = 161.9 * amu #atomic mass units
a_11 = 95 
a_22 = 95 
a_12 = 95  #variable
mu_1 = 9.93 * bohr_magneton
mu_2 = 9.93 * bohr_magneton


g_11 = 4*np.pi*a_11/mass
g_22 = 4*np.pi*a_22/mass
g_12 = 2*np.pi*a_12*(1/mass + 1/mass) #mass is the same

# https://www.wolframalpha.com/input/?i=%28electron+mass%29+*+%28magnetic+constant%29+*+%28bohr+magneton%29%5E2+%2F+%28hbar%5E2+*++%28bohr+radius%29%29
prefac_dd = 1.67294064*1.E-04
a_11_dd = prefac_dd * mu_1 * mu_1 * mass / (12 * np.pi)
a_22_dd = prefac_dd * mu_2 * mu_2 * mass / (12 * np.pi)
a_12_dd = prefac_dd * mu_1 * mu_2 * mass / (12 * np.pi)
g_11_dd = 4*np.pi*a_11_dd/mass
g_22_dd = 4*np.pi*a_22_dd/mass
g_12_dd = 2*np.pi*a_12_dd*(1/mass + 1/mass) #mass is the same


fx = 229  # Hz
fy = 37
fz = 165 # Hz
fun_unit = 7.85057874E-6 
w_ho = np.array([fx, fy, fz])
a_ho = fun_unit / np.sqrt(w_ho) / 1.E-06 * mu_m

# sqrt(hbar / ((164 atomic mass unit) * 2 * pi * Hertz)) = 7.85057874×10^-6 meters
# taken from: https://www.wolframalpha.com/input/?i=sqrt%28hbar+%2F+%28%28164+atomic+mass+unit%29+*+2+*+pi+*+Hertz%29%29


nparticles = 1800
np1, np2 = 1.4E+04    , 1.E-05 #number of particles: type 1, type 2
m2 = 1.
# [ 10.70953852  21.41907704   0.           0.         220.00181649  21.34811108]
# [ 10.70953853  21.41907706   0.           0.         220.00540222  21.35340471]
# 
nxyz = np.ones(3) * 64 #if you want 1D or 2D, just change this array
nxyz = np.array([64, 64, 64])
nxyz = np.array(nxyz, dtype=np.int32)
L = 0.75*np.array([15, 60, 40]) * mu_m # if you want 1D or 2D, just change this array. This is box size

#np.random.seed(seed=42) #set the seed

L_ok = L * 0.8 # np.array([10000., 6000., 6000.])

EN_TOL = 1.E-8 # if the energy in the new iteration is within the EN_TOL tolerance, exit the simulation
t_equil = 1.5E+05
t_prod  = 0. #np.inf for infinite      #total propagation time. Count from the real (imaginary) timestep if you propagate in real (imaginary) time
deltat_prod    = 0.1 # real timestep
deltat_equil = 5 # this is abs value of imaginary part


print_every      = 300 # (int)(t_equil/deltat_equil / 100)


def potential_external(x, y, z):
	return 0.5 * (x**2/a_ho[0]**4 + y**2/a_ho[1]**4 + z**2/a_ho[2]**4)
	

roll_dist = 0
def init_psi_1(x, y, z):
	#return np.load("../psi_1.npy")
	noise = np.random.uniform(0, 1, np.shape(x))
	return noise * np.roll(np.exp(-0.5 * 0.05 * ((x/a_ho[0])**2 + (y/a_ho[1])**2 + (z/a_ho[2])**2) ) + 0.j, roll_dist, axis=2)
	
def init_psi_2(x, y, z):
	noise = np.random.uniform(0, 1, np.shape(x))
	return noise * np.roll(np.exp(-0.5  * 0.05 * ((x/a_ho[0])**2 + (y/a_ho[1])**2 + (z/a_ho[2])**2) ) + 0.j, -roll_dist, axis=2)
	



################################################################################
################################################################################
################################################################################
################################################################################



#exit()

imProp = False
#setting up auxiliary variables
N_DIM = len(nxyz)
LHalf = L / 2
x_lin = np.linspace(-LHalf[0], LHalf[0], num = nxyz[0], endpoint=False)
y_lin = np.linspace(-LHalf[1], LHalf[1], num = nxyz[1], endpoint=False)
z_lin = np.linspace(-LHalf[2], LHalf[2], num = nxyz[2], endpoint=False)

dx = L/nxyz
d3r = np.prod(dx)

kx = np.fft.fftfreq(nxyz[0], dx[0]/( 2. * np.pi))
ky = np.fft.fftfreq(nxyz[1], dx[1]/( 2. * np.pi))
kz = np.fft.fftfreq(nxyz[2], dx[2]/( 2. * np.pi))

d3k = (kx[1] - kx[0]) * (ky[1] - ky[0]) * (kz[1] - kz[0])
x, y, z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')





### global variables
lhy_energy, mu_a, mu_b = 1, 1, 1
sumk2_1 = (kx**2 + ky**2 + kz**2)/2
sumk2_2 = (kx**2 + ky**2 + kz**2)/2/m2
kinprop_1, kinprop_2 = 1., 1.
dt_equil = -1.j*deltat_equil
dt_prod = deltat_prod
psi_1, psi_2 = 1, 1
###
print(" *** Parameters of the simulation ***")
print("nxyz ", nxyz)
print("L ", L)
print("d3r " , d3r)
print("d3k", d3k )
x2 = x**2
z2 = z**2
pot_ext = 1.
kvec = np.sqrt(kx**2 + ky**2 + kz**2)
ft_dip =  np.nan_to_num(3 / (4*np.pi) * (2*kz**2 - kx**2 - ky**2) / kvec**2, posinf=0)


def get_phis(den1, den2):
	phi_1  = np.fft.ifftn(ft_dip * np.fft.fftn(den1))
	phi_2  = np.fft.ifftn(ft_dip * np.fft.fftn(den2))
	return phi_1, phi_2

def energy_density_interaction(n1, n2, phi_1, phi_2):
	en_mf  = 0.5*g_11*n1**2 + 0.5*g_22*n2**2 + g_12*n1*n2 + 0.j
	en_mf += 0.5*g_11_dd*phi_1*n1 + 0.5*g_22_dd*phi_2*n2 + g_12_dd * (phi_1 * n2 + phi_2 * n1)
	en_lhy = lhy_energy(n1, n2, grid=False)
	return np.array([d3r * np.sum(en_mf.real), d3r * np.sum(en_lhy)])


def dEps_dPsi(na, nb, phi_a, phi_b):
	lmu_a = g_11*na + g_12*nb  + g_11_dd * phi_a  + g_12_dd * phi_b
	lmu_b = g_22*nb + g_12*na  + g_22_dd * phi_b  + g_12_dd * phi_a
	lmu_a += mu_a(na, nb, grid=False)
	lmu_b += mu_b(na, nb, grid=False)
	return lmu_a, lmu_b

def chemical_potential(psi_1, psi_2):
	den_1 = np.abs(psi_1) ** 2
	den_2 = np.abs(psi_2) ** 2
	phi_1, phi_2 = get_phis(den_1, den_2)
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	c1 = np.sum(pot_1 * den_1 * d3r) / np1 
	c2 = np.sum(pot_2 * den_2 * d3r) / np2
	return np.array([c1.real, c2.real])


def T2_operator(psi_1, psi_2, den_1, den_2, phi_1, phi_2):
	global dt, pot_ext, imProp
	#exp(-1/2 * i dt * V)  
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	psi_1 *= np.exp(-0.5j * (pot_1+pot_ext) * dt)
	psi_2 *= np.exp(-0.5j * (pot_2+pot_ext) * dt)
	
	psi_1 = np.fft.fftn(psi_1)
	psi_2 = np.fft.fftn(psi_2)
	psi_1 *= kinprop_1
	psi_2 *= kinprop_2
	psi_1 = np.fft.ifftn(psi_1)
	psi_2 = np.fft.ifftn(psi_2)

	den_1 = np.abs(psi_1) ** 2
	den_2 = np.abs(psi_2) ** 2
	phi_1, phi_2 = get_phis(den_1, den_2)
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	psi_1 *= np.exp(-0.5j * (pot_1+pot_ext) * dt)
	psi_2 *= np.exp(-0.5j * (pot_2+pot_ext) * dt)
	
	if(imProp): #normalize		
		psi_1 *= np.sqrt(np1)/np.sqrt(d3r * np.sum(np.abs(psi_1) ** 2))
		psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))
	
	return psi_1, psi_2
	


def energy(psi_1, psi_2, den_1, den_2, phi_1, phi_2):
	p_ext = np.array([d3r * np.sum(den_1 * pot_ext), d3r * np.sum(den_2 * pot_ext)])
	p_int = energy_density_interaction(den_1, den_2, phi_1, phi_2)
	k_en = np.array([ d3r * np.sum((np.conj(psi_1) * np.fft.ifftn(np.fft.fftn(psi_1) * sumk2_1)).real), 
					  d3r * np.sum((np.conj(psi_2) * np.fft.ifftn(np.fft.fftn(psi_2) * sumk2_2)).real)])
	return np.concatenate((k_en, p_ext, p_int), axis=None)

output_dir = 'snapshots_time_evolution_0'; num=0
while os.path.exists(output_dir): num+=1; output_dir="snapshots_time_evolution_"+str(num)
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(output_dir+'/npy_files'): os.makedirs(output_dir+'/npy_files')
file_en_equil = open(output_dir + '/en_equil.dat', 'w', buffering=1)
file_en_prod = open(output_dir + '/en_prod.dat', 'w', buffering=1)
delta_t_crit=0.


np.save(f"{output_dir}/npy_files/x_axis", x_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/y_axis", y_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/z_axis", z_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/L_box", L)



def dft_simulation(t_max,delta_t):
	global psi_1, psi_2	
	timestep = 0
	time = 0.
	en_old = np.inf; en = -np.inf
	while time <= t_max:
		den_1, den_2 = np.abs(psi_1)**2, np.abs(psi_2)**2
		phi_1, phi_2 = get_phis(den_1, den_2)
		
		if timestep % print_every == 0 or timestep==0:
			ts = (int)(timestep / print_every)

			if(imProp == False):
				nc1, nc2 = d3r * np.sum(den_1), d3r * np.sum(den_2)
			else:
				nc1, nc2 = np1, np2
			en_contributions = energy(psi_1, psi_2, den_1, den_2, phi_1, phi_2)
			en_old = en
			en = np.sum(en_contributions)

			string_out= f"{time:.3e} {en:.5e}  \t {np.max(den_1):.4e} {np.max(den_2):.4e} \n"
			if(imProp):
				file_en_equil.write(string_out)
			else:
				file_en_prod.write(string_out)
			print(string_out)

			
			if(imProp==True):
				file= f"{output_dir}/npy_files/psi_equil_{ts}"
			else:
				file= f"{output_dir}/npy_files/psi_prod_{ts}"
			
			np.save(f"{output_dir}/psi_1", psi_1)
			np.save(f"{output_dir}/psi_2", psi_2)
			
			np.save(file + "_xy_1", dx[2] * np.sum(den_1, axis=2) )  # use exponential notation
			np.save(file + "_xy_2", dx[2] * np.sum(den_2, axis=2) )  # use exponential notation
			np.save(file + "_xz_1", dx[2] * np.sum(den_1, axis=1) )  # use exponential notation
			np.save(file + "_xz_2", dx[2] * np.sum(den_2, axis=1) )  # use exponential notation
			np.save(file + "_yz_1", dx[2] * np.sum(den_1, axis=0) )  # use exponential notation
			np.save(file + "_yz_2", dx[2] * np.sum(den_2, axis=0) )  # use exponential notation

			np.save(file + "_x_int_1", dx[2]*dx[1] * np.sum(den_1, axis=(1, 2)) )  # use exponential notation
			np.save(file + "_y_int_1", dx[2]*dx[0] * np.sum(den_1, axis=(0, 2)) )  # use exponential notation
			np.save(file + "_z_int_1", dx[0]*dx[1] * np.sum(den_1, axis=(0, 1)) )  # use exponential notation
			np.save(file + "_x_int_2", dx[2]*dx[1] * np.sum(den_2, axis=(1, 2)) )  # use exponential notation
			np.save(file + "_y_int_2", dx[2]*dx[0] * np.sum(den_2, axis=(0, 2)) )  # use exponential notation
			np.save(file + "_z_int_2", dx[0]*dx[1] * np.sum(den_2, axis=(0, 1)) )  # use exponential notation

			np.save(file + "_x_1", np.swapaxes(den_1,0,2)[int(nxyz[2]/2)][int(nxyz[1]/2)])   # use exponential notation
			np.save(file + "_x_2", np.swapaxes(den_2,0,2)[int(nxyz[2]/2)][int(nxyz[1]/2)])   # use exponential notation
			np.save(file + "_y_1", np.swapaxes(den_1,1,2)[int(nxyz[0]/2)][int(nxyz[2]/2)])   # use exponential notation
			np.save(file + "_y_2", np.swapaxes(den_2,1,2)[int(nxyz[0]/2)][int(nxyz[2]/2)])   # use exponential notation
			np.save(file + "_z_1", den_1[int(nxyz[0]/2)][int(nxyz[1]/2)])   # use exponential notation
			np.save(file + "_z_2", den_2[int(nxyz[0]/2)][int(nxyz[1]/2)])   # use exponential notation
			



		psi_1, psi_2 = T2_operator(psi_1, psi_2, den_1, den_2, phi_1, phi_2)

		time += delta_t
		timestep += 1
		if np.abs((en - en_old) / en) < EN_TOL:
			break
	return 0

pot_ext = potential_external(x, y, z)


print("Loading interpolating pickles ... ")
import pickle
with open(f"../interpolators_fine-tuning/interpolator_lhy_energy_{a_12:.1f}.pkl", 'rb') as f:
    lhy_energy = pickle.load(f)
with open(f"../interpolators_fine-tuning/interpolator_mu_a_{a_12:.1f}.pkl", 'rb') as f:
    mu_a = pickle.load(f)
with open(f"../interpolators_fine-tuning/interpolator_mu_b_{a_12:.1f}.pkl", 'rb') as f:
    mu_b = pickle.load(f)
print("pickles sucessfully loaded!")
#Trotter operator, O(dt^2) global error


print("Initializing wf's... ")
psi_1, psi_2 = init_psi_1(x, y, z), init_psi_2(x, y, z)
psi_1 *= np.sqrt(np1)/np.sqrt(d3r * np.sum(np.abs(psi_1) ** 2))  #normalize
psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))  #normalize
print("wfs initiated ")

print(np1, np2)
den_1, den_2 = np.abs(psi_1)**2, np.abs(psi_2)**2
print(f"\nMAX DENSITY IS {np.max(den_1):.3e} {np.max(den_2):.3e}\n")





print("Initializing kinetic energy propagator ... ")
imProp = True
dt = dt_equil
kinprop_1 = np.exp(-1j * dt * sumk2_1)
kinprop_2 = np.exp(-1j * dt * sumk2_2)
print(" *** EQUILIBRATION ... **** ")



N2_actual = np2
np2 = 1.E-08
psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))  #normalize
print("Equilibrating state 1 ...")
dft_simulation(t_equil, deltat_equil)


print("Doping condensate 2 ...")
np2 = N2_actual
psi_2 = init_psi_2(x, y, z)
psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))  #normalize
dft_simulation(t_equil, deltat_equil)


print("EQUILIBRATION ENDED at ", time.ctime())



print( f"Equilibration took {time.time() - start_time} seconds")



imProp = False
dt = dt_prod
kinprop_1 = np.exp(-1j * dt * sumk2_1)
kinprop_2 = np.exp(-1j * dt * sumk2_2)
print(" *** PRODUCTION ... **** ")
print(" *** delta_t_crit = %.5e" % delta_t_crit)
dft_simulation(t_prod, deltat_prod)



print("SIM. ENDED at ", time.ctime())
print( time.time() - start_time)


	