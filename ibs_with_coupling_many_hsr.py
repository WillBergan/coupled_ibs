import numpy as np
import sys

pi = 4.*np.arctan(1.)

def rd(x_safe, y_safe, z_safe):
    """Compute RD function, as described in Algorithm 4 of B. C. Carlson,
    "Computing Elliptic Integrals by Duplication," Numerische Mathematik 33,
    1-16 (1979).

    x_safe, y_safe, and z_safe are the three arguments of the function
    RD(x_safe, y_safe ,z_safe).
    """

    err_tol = 1.e-5

    # Likely unneccessary, but just avoids anyweird case of overwriting the initial values.
    x = x_safe
    y = y_safe
    z = z_safe

    # Coefficients from Eq. 2.27
    c1 = 3./7.
    c2 = 1./3.
    c3 = 3./22.
    c4 = 3./11.
    c5 = 3./13.

    s_vec = np.zeros(4)

    # Keep track of iterations and ensure we run at least once.
    first_run = 1

    # Eq. 2.23
    mu = (x+y+3.*z)/5.
    lamb = np.sqrt(x*y) + np.sqrt(x*z) + np.sqrt(y*z)

    sigma = 0.

    count = 0

    # Halting condition from Eq. 2.25 - 2.26
    while(max(max(1-x/mu, 1-y/mu), 1-z/mu) > err_tol or first_run):
        first_run = 0

        # Eq. 2.26
        s_vec[0] = (x**2. + y**2. + 3.*z**2.)/4.
        s_vec[1] = (x**3. + y**3. + 3.*z**3.)/6.
        s_vec[2] = (x**4. + y**4. + 3.*z**4.)/8.
        s_vec[3] = (x**5. + y**5. + 3.*z**5.)/10.

        # First line of Eq. 2.27
        sigma = sigma + 3. * 4**(-count)/(np.sqrt(z)*(z+lamb))

        # Eq. 2.24
        x = (x + lamb)/4.
        y = (y + lamb)/4.
        z = (z + lamb)/4.

        # Eq. 2.23
        mu = (x+y+3.*z)/5.
        lamb = np.sqrt(x*y) + np.sqrt(x*z) + np.sqrt(y*z)
        count = count + 1

    # Eq. 2.26
    s_vec[0] = (x**2. + y**2. + 3.*z**2.)/4.
    s_vec[1] = (x**3. + y**3. + 3.*z**3.)/6.
    s_vec[2] = (x**4. + y**4. + 3.*z**4.)/8.
    s_vec[3] = (x**5. + y**5. + 3.*z**5.)/10.

    # Eq. 2.27
    return sigma + 4**(-count)*mu**(-1.5)*(1 + c1*s_vec[0] + c2*s_vec[1]\
            + c3*s_vec[0]**2. + c4*s_vec[2] + c5*s_vec[0]*s_vec[1]\
            + c5*s_vec[3])

def single_ibs_rate(beta_1x, alpha_1x, beta_1y, alpha_1y, beta_2x, alpha_2x, beta_2y, alpha_2y, Dx, Dpx, Dy, Dpy, nu_1, nu_2, u):
    """Compute IBS rates for coupled lattice using the methods of V. Lebedev and S. Nagaitsev, "Multiple Intrabeam Scattering in x-y Coupled Focusing Systems," arXiv:1812.09275v5.
    beta_1x: horizontal beta for first normal mode
    alpha_1x: horizontal alpha for first normal mode
    beta_1y: vertical beta for first normal mode
    alpha_1y: vertical alpha for first normal mode
    beta_2x: horizontal beta for second normal mode
    alpha_2x: horizontal alpha for secondnormal mode
    beta_2y: vertical beta for second normal mode
    alpha_2y: vertical alpha for second normal mode
    Dx: horizontal dispersion
    Dpx: horizontal dispersion derivative
    Dy: vertical dispersion
    Dpy: vertical dispersion derivative
    nu_1: phase factor for mode 1 to vertical, as in lattice file, converted to radians
    nu_2: phase factor for mode 2 to horizontal, as in lattice file, converted to radians
    u: coupling factor from lattice file

    returns ibs rates for the three normal modes, in units of inverse seconds
    """

    imag = complex(0,1)
   
    # Hardcoded bunch parameters
    N = 67965000000.0  # Bunch population
    r_h = 1.5346982671792561e-18  # Classical particle radius q^2/(4 pi epsilon_0 m c^2)
    clight = 2.99792458e8  # Speed of light
    gamma_rel = 293.09195431694604  # Relativistic gamma
    beta_rel = np.sqrt(1-1./gamma_rel**2.)
    sigma_z = 0.05897210792573457  # RMS Bunch Length
   
    # Mode 1 and 2 emittances
    emit_1 = 1.1123620381607244e-08
    emit_2 = 9.967443976342154e-10
   
    # RMS fractional energy spread
    sigma_p = 0.0006146712730120792
  
    # Eq. 4.7 of V.A. Lebedev and S. A. Bogacz, “Betatron motion with coupling of horizontal and vertical degrees of freedom”, 2010 JINST 5 P10010.
    kx = np.sqrt(beta_2x/beta_1x)
    ky = np.sqrt(beta_1y/beta_2y)
   
    Ax = kx*alpha_1x - 1./kx*alpha_2x
    Ay = ky*alpha_2y - 1./ky*alpha_1y

    # Look at Eq. A.2 of V.A. Lebedev and S. A. Bogacz, “Betatron motion with coupling of horizontal and vertical degrees of freedom”,2010 JINST 5 P10010.
    Xi = np.zeros((5,5))
    Xi[0,0] = ((1-u)**2. + alpha_1x**2.)/(emit_1*beta_1x) + (u**2. + alpha_2x**2.)/(emit_2*beta_2x)
    Xi[1,1] = beta_1x/emit_1 + beta_2x/emit_2
    Xi[2,2] = (u**2. + alpha_1y**2.)/(emit_1*beta_1y) + ((1-u)**2. + alpha_2y**2.)/(emit_2*beta_2y)
    Xi[3,3] = beta_1y/emit_1 + beta_2y/emit_2
    Xi[0,1] = alpha_1x/emit_1 + alpha_2x/emit_2
    Xi[1,0] = Xi[0,1]
    Xi[2,3] = alpha_1y/emit_1 + alpha_2y/emit_2
    Xi[3,2] = Xi[2,3]
    Xi[0,2] = ((alpha_1x*alpha_1y+ u*(1-u))*np.cos(nu_1) + (alpha_1y*(1-u) - alpha_1x*u)*np.sin(nu_1))/(emit_1*np.sqrt(beta_1x*beta_1y)) + ((alpha_2x*alpha_2y+ u*(1-u))*np.cos(nu_2) + (alpha_2x*(1-u) - alpha_2y*u)*np.sin(nu_2))/(emit_2*np.sqrt(beta_2x*beta_2y))
    Xi[2,0] = Xi[0,2]
    Xi[0,3] = np.sqrt(beta_1y/beta_1x)*(alpha_1x*np.cos(nu_1) + (1-u)*np.sin(nu_1))/emit_1 + np.sqrt(beta_2y/beta_2x)*(alpha_2x*np.cos(nu_2) - u*np.sin(nu_2))/emit_2
    Xi[3,0] = Xi[0,3]
    Xi[1,2] = np.sqrt(beta_1x/beta_1y)*(alpha_1y*np.cos(nu_1) - u*np.sin(nu_1))/emit_1 + np.sqrt(beta_2x/beta_2y)*(alpha_2y*np.cos(nu_2) + (1-u)*np.sin(nu_2))/emit_2
    Xi[2,1] = Xi[1,2]
    Xi[1,3] = np.sqrt(beta_1x*beta_1y)*np.cos(nu_1)/emit_1 + np.sqrt(beta_2x*beta_2y)*np.cos(nu_2)/emit_2
    Xi[3,1] = Xi[1,3]
   
    # Vector of dispersions
    D_long = np.zeros((5,1))
    D_long[0,0] = Dx
    D_long[1,0] = Dpx
    D_long[2,0] = Dy
    D_long[3,0] = Dpy
   
    D_long_transpose = D_long.transpose()
   
    D = np.zeros((4,1), dtype=complex)
    D[0,0] = D_long[0]
    D[1,0] = D_long[1]
    D[2,0] = D_long[2]
    D[3,0] = D_long[3]
   
    D_transpose = D.transpose()
   
    # Eq. 31 to expand Xi matrix with dispersion
    extra_column = np.matmul(Xi, D_long)
   
    Xi[0,4] = extra_column[0]
    Xi[1,4] = extra_column[1]
    Xi[2,4] = extra_column[2]
    Xi[3,4] = extra_column[3]
    Xi[4,0] = extra_column[0]
    Xi[4,1] = extra_column[1]
    Xi[4,2] = extra_column[2]
    Xi[4,3] = extra_column[3]
    Xi[4,4] = 1./sigma_p**2. + np.matmul(D_long_transpose, np.matmul(Xi, D_long))
   
    # Eq. 47
    A = np.zeros((3,3))
    A[0,0] = Xi[1,1]
    A[0,1] = Xi[1,3]
    A[0,2] = Xi[1,4]*gamma_rel
    A[1,0] = Xi[1,3]
    A[1,1] = Xi[3,3]
    A[1,2] = Xi[3,4]*gamma_rel
    A[2,0] = Xi[1,4]*gamma_rel
    A[2,1] = Xi[3,4]*gamma_rel
    A[2,2] = Xi[4,4]*gamma_rel**2.
   
    # V matrices from Eq. 23
    v1 = np.zeros((4,1), dtype=complex)
    v2 = np.zeros((4,1), dtype=complex)
   
    v1[0,0] = np.sqrt(beta_1x)
    v1[1,0] = -(imag*(1-u)+alpha_1x)/np.sqrt(beta_1x)
    v1[2,0] = np.sqrt(beta_1y)*np.exp(imag*nu_1)
    v1[3,0] = -(imag*u + alpha_1y)/np.sqrt(beta_1y)*np.exp(imag*nu_1)
   
    v2[0,0] = np.sqrt(beta_2x)*np.exp(imag*nu_2)
    v2[1,0] = -(imag*u + alpha_2x)/np.sqrt(beta_2x)*np.exp(imag*nu_2)
    v2[2,0] = np.sqrt(beta_2y)
    v2[3,0] = -(imag*(1-u) + alpha_2y)/np.sqrt(beta_2y)
   
    v1_plus = (v1.conjugate()).transpose()
    v2_plus = (v2.conjugate()).transpose()
   
    # Eq. 26
    U = np.zeros((4,4))
    U[0,1] = 1
    U[1,0] = -1
    U[2,3] = 1
    U[3,2] = -1
   
    U_transpose = U.transpose()
   
    # Eq. 38
    V1_tilde_short = np.zeros((4,4), dtype=complex)
    V2_tilde_short = np.zeros((4,4), dtype=complex)
   
    V1_tilde_short = np.real(np.matmul(U_transpose, np.matmul(v1,np.matmul(v1_plus,U)))) 
    V2_tilde_short = np.real(np.matmul(U_transpose, np.matmul(v2,np.matmul(v2_plus,U))))
   
    # Expand V_tilde matrix to include dispersion according to Eq. 48
    V1_tilde = np.zeros((5,5))
    V2_tilde = np.zeros((5,5))
   
    V1_tilde[0:4,0:4] = V1_tilde_short[:,:]
    V2_tilde[0:4,0:4] = V2_tilde_short[:,:]
   
    extra_column = np.real(np.matmul(V1_tilde_short, D))
   
    V1_tilde[0,4] = extra_column[0]
    V1_tilde[1,4] = extra_column[1]
    V1_tilde[2,4] = extra_column[2]
    V1_tilde[3,4] = extra_column[3]
   
    extra_column = np.real(np.matmul(V2_tilde_short, D))
   
    V2_tilde[0,4] = extra_column[0]
    V2_tilde[1,4] = extra_column[1]
    V2_tilde[2,4] = extra_column[2]
    V2_tilde[3,4] = extra_column[3]
   
   
    # Eq. 49
    a1 = np.zeros((3,3), dtype = complex)
    a2 = np.zeros((3,3), dtype = complex)
    a3 = np.zeros((3,3))
   
    a1[0,0] = 0.5*V1_tilde[1,1]
    a1[0,1] = 0.5*V1_tilde[1,3]
    a1[0,2] = 0.5*V1_tilde[1,4]*gamma_rel
    a1[1,0] = 0.5*V1_tilde[1,3]
    a1[1,1] = 0.5*V1_tilde[3,3]
    a1[1,2] = 0.5*V1_tilde[3,4]*gamma_rel
    a1[2,0] = 0.5*V1_tilde[1,4]*gamma_rel
    a1[2,1] = 0.5*V1_tilde[3,4]*gamma_rel
    a1[2,2] = 0.5*np.matmul(D_transpose,np.matmul(V1_tilde_short,D))*gamma_rel**2.
   
    a2[0,0] = 0.5*V2_tilde[1,1]
    a2[0,1] = 0.5*V2_tilde[1,3]
    a2[0,2] = 0.5*V2_tilde[1,4]*gamma_rel
    a2[1,0] = 0.5*V2_tilde[1,3]
    a2[1,1] = 0.5*V2_tilde[3,3]
    a2[1,2] = 0.5*V2_tilde[3,4]*gamma_rel
    a2[2,0] = 0.5*V2_tilde[1,4]*gamma_rel
    a2[2,1] = 0.5*V2_tilde[3,4]*gamma_rel
    a2[2,2] = 0.5*np.matmul(D_transpose,np.matmul(V2_tilde_short,D))*gamma_rel**2.
   
    a3[2,2] = gamma_rel**2.
   
    a1 = np.real(a1)
    a2 = np.real(a2)
    a3 = np.real(a3)
   
    # Diagonalize A; see Eq. 50
    eigval, T = np.linalg.eig(A)
    sigma_1 = 1./np.sqrt(eigval[0])
    sigma_2 = 1./np.sqrt(eigval[1])
    sigma_3 = 1./np.sqrt(eigval[2])
   
    T_transpose = T.transpose()

    # Get Coulomb logarithm
    # See Eq. 56 and 22
    Sigma_v = np.linalg.inv(A)*(gamma_rel*beta_rel*clight)**2.
    
    rho_min = r_h*clight**2./np.trace(Sigma_v)
    sigma_x = np.sqrt(emit_1*beta_1x + emit_2*beta_2x + (Dx*sigma_p)**2.)
    sigma_y = np.sqrt(emit_1*beta_1y + emit_2*beta_2y + (Dy*sigma_p)**2.)
    n_density = N/(sigma_x*sigma_y*gamma_rel*sigma_z)

    Sigma = np.linalg.inv(Xi)
    sigma_min = np.sqrt((Sigma[0,0] + Sigma[2,2] - np.sqrt((Sigma[0,0] - Sigma[2,2])**2. + 4.*Sigma[0,2]**2.))/2.)
 
    rho_max = min(sigma_min, gamma_rel*sigma_z)
    rho_max = min(rho_max, np.sqrt(np.trace(Sigma_v)/(4.*pi*n_density*r_h*clight**2.)))

    clog = np.log(rho_max/rho_min)

    # See Eq. 53 - 55
    rate_1 = 1./emit_1 * N*r_h**2.*clight/(3.*np.sqrt(pi)*beta_rel**3.*gamma_rel**4.*2.*np.sqrt(pi)*sigma_z*emit_1*emit_2*sigma_p) * clog*sigma_1*sigma_2*sigma_3*(sigma_1**2.*rd(sigma_2**2.,sigma_3**2.,sigma_1**2.)*(np.trace(a1) - 3.*np.matmul(T_transpose,np.matmul(a1,T))[0,0]) + sigma_2**2.*rd(sigma_3**2.,sigma_1**2.,sigma_2**2.)*(np.trace(a1) - 3.*np.matmul(T_transpose,np.matmul(a1,T))[1,1]) + sigma_3**2.*rd(sigma_1**2.,sigma_2**2.,sigma_3**2.)*(np.trace(a1) - 3.*np.matmul(T_transpose,np.matmul(a1,T))[2,2]))
   
    rate_2 = 1./emit_2 * N*r_h**2.*clight/(3.*np.sqrt(pi)*beta_rel**3.*gamma_rel**4.*2.*np.sqrt(pi)*sigma_z*emit_1*emit_2*sigma_p) * clog*sigma_1*sigma_2*sigma_3*(sigma_1**2.*rd(sigma_2**2.,sigma_3**2.,sigma_1**2.)*(np.trace(a2) - 3.*np.matmul(T_transpose,np.matmul(a2,T))[0,0]) + sigma_2**2.*rd(sigma_3**2.,sigma_1**2.,sigma_2**2.)*(np.trace(a2) - 3.*np.matmul(T_transpose,np.matmul(a2,T))[1,1]) + sigma_3**2.*rd(sigma_1**2.,sigma_2**2.,sigma_3**2.)*(np.trace(a2) - 3.*np.matmul(T_transpose,np.matmul(a2,T))[2,2]))
   
    rate_3 = 1./sigma_p**2. * N*r_h**2.*clight/(6.*np.sqrt(pi)*beta_rel**3.*gamma_rel**4.*2.*np.sqrt(pi)*sigma_z*emit_1*emit_2*sigma_p) * clog*sigma_1*sigma_2*sigma_3*(sigma_1**2.*rd(sigma_2**2.,sigma_3**2.,sigma_1**2.)*(np.trace(a3) - 3.*np.matmul(T_transpose,np.matmul(a3,T))[0,0]) + sigma_2**2.*rd(sigma_3**2.,sigma_1**2.,sigma_2**2.)*(np.trace(a3) - 3.*np.matmul(T_transpose,np.matmul(a3,T))[1,1]) + sigma_3**2.*rd(sigma_1**2.,sigma_2**2.,sigma_3**2.)*(np.trace(a3) - 3.*np.matmul(T_transpose,np.matmul(a3,T))[2,2]))
   

    return(rate_1, rate_2, rate_3)




# Get infile name
if(len(sys.argv) >= 2):
    infile_name = sys.argv[1]
else:
    raise RuntimeError("Need to specify optics file!!!")

full_rate_1 = 0
full_rate_2 = 0
full_rate_3 = 0

circumf = 0

with open(infile_name) as f:
    lines = f.readlines()
    for i in range(6,len(lines)-1):
        # Read in lines from the input file
        # This can be trivially adjusted for different input formats
        this_line = lines[i].split()
        next_line = lines[i+1].split()
        ds = (float(next_line[1]) - float(this_line[1]))  # Length of lattice step.

        # Read in usual betas, alphas, dispersion, and dispersion derivatives.
        beta_1x = float(this_line[2])
        alpha_1x = float(this_line[3])
        beta_2y = float(this_line[4])
        alpha_2y = float(this_line[5])
        Dx = float(this_line[6])
        Dpx = float(this_line[7])
        Dy = float(this_line[8])
        Dpy = float(this_line[9])

        # Make a coupled lattice by bringing the coupled beta functions close to the usual ones.
        # Do not make an exact equality to avoid some divide-by-zero issues.
        beta_1y = 0.995*beta_2y
        alpha_1y = 0.996*alpha_2y
        beta_2x = 0.997*beta_1x
        alpha_2x = 0.998*alpha_1x


        # Get coupling constant and nu_1, nu_2
        # V.A. Lebedev and S. A. Bogacz, “Betatron motion with coupling of horizontal and vertical degrees of freedom”,2010 JINST 5 P10010.
        # See. Eq. 4.7
        kx = np.sqrt(beta_2x/beta_1x)
        ky = np.sqrt(beta_1y/beta_2y)
       
        Ax = kx*alpha_1x - 1./kx*alpha_2x
        Ay = ky*alpha_2y - 1./ky*alpha_1y
   

        # Check to make sure argument of the square root is positive
        # Otherwise, skip that section of the ring; hope not too many such errors...
        if((kx**2.*ky**2.*(1 + (Ax**2. - Ay**2.)/(kx**2. - ky**2.)*(1. - kx**2.*ky**2.))) > 0):

            # Calculate the coupling factors nu_1, nu_2, and u based on V.A. Lebedev and S. A. Bogacz, “Betatron motion with coupling of horizontal and vertical degrees of freedom”,2010 JINST 5 P10010.
            # In particular, Eq. 4.8 - 4.11
            u1 = (-kx**2.*ky**2. + np.sqrt(kx**2.*ky**2.*(1 + (Ax**2. - Ay**2.)/(kx**2. - ky**2.)*(1. - kx**2.*ky**2.))))/(1-kx**2.*ky**2.)
            u2 = (-kx**2.*ky**2. - np.sqrt(kx**2.*ky**2.*(1 + (Ax**2. - Ay**2.)/(kx**2. - ky**2.)*(1. - kx**2.*ky**2.))))/(1-kx**2.*ky**2.)
           
            u = u1
 
            # Uncomment for debugging
            # print(i, (Ax*Ay - (kx*(1-u)+u/kx)*(ky*(1-u)+u/ky))/(Ay**2. + (ky*(1-u) + u/ky)**2.), (Ax*Ay + (kx*(1-u)-u/kx)*(ky*(1-u)-u/ky))/(Ay**2. + (ky*(1-u) - u/ky)**2.))
    
            acos_arg = (Ax*Ay - (kx*(1-u)+u/kx)*(ky*(1-u)+u/ky))/(Ay**2. + (ky*(1-u) + u/ky)**2.)

            if(acos_arg < 1):
                nu_plus = pi
            elif(acos_arg > 1):
                nu_plus = 0
            else:
                nu_plus = np.arccos(acos_arg)
           
            if(Ax*(ky*(1-u) + u/ky) + Ay*(kx*(1-u)+u/kx) < 0):
                nu_plus = -nu_plus

                
            acos_arg = (Ax*Ay + (kx*(1-u)-u/kx)*(ky*(1-u)-u/ky))/(Ay**2. + (ky*(1-u) - u/ky)**2.)

            if(acos_arg < 1):
                nu_minus = pi
            elif(acos_arg > 1):
                nu_minus = 0
            else:
                nu_minus = np.arccos(acos_arg)
           
            if(-Ax*(ky*(1-u) - u/ky) + Ay*(kx*(1-u)-u/kx) < 0):
                nu_minus = -nu_minus
 
           
            nu_1 = (nu_plus + nu_minus)/2.
            nu_2 = (nu_plus - nu_minus)/2.
 
 
            circumf = circumf + ds

            # Compute the contribution to the IBS rate from this section of the ring.
            rate_1, rate_2, rate_3 = single_ibs_rate(beta_1x, alpha_1x, beta_1y, alpha_1y, beta_2x, alpha_2x, beta_2y, alpha_2y, Dx, Dpx, Dy, Dpy, nu_1, nu_2, u)
 
            # Do integral over circumference.
            full_rate_1 = full_rate_1 + rate_1*ds
            full_rate_2 = full_rate_2 + rate_2*ds
            full_rate_3 = full_rate_3 + rate_3*ds
        else:
            print(i, ds)

# Normalize rates by circumference.
full_rate_1 = full_rate_1 / circumf
full_rate_2 = full_rate_2 / circumf
full_rate_3 = full_rate_3 / circumf

print("Rates (1/s):", full_rate_1, full_rate_2, full_rate_3)

print("Times (hr):", 1./3600./full_rate_1, 1./3600./full_rate_2, 1./3600./full_rate_3)
