from CLT import Lamina,Laminate,get_angles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m

#####-------------------------------------------------------Helper functions----------------------------------------------------
def get_main_forces(theta,F=837000):
    """
    :param theta: Angle must be between 0 and 90 (from assignment)
    :param F: optional parameter default is ultimate load=837kN
    :return: returns internal normal load [N], moment [Nm] and shear force[N]
    """
    R = 1.75
    theta = theta*m.pi/180
    M = -F*R/2*m.sin(theta)
    N = -F*m.sin(theta)
    V = F*m.cos(theta)/2
    return N,M,V

def material_properties(material):
    material = material.lower()
    if material=='ud':
        return 130*1e9,9*1e9,5.5*1e9,0.29,0.14*1e-3,1780*1e6,1200*1e6,55*1e6,122*1e6,100*1e6,115*1e6,1608
    elif material=='pw':
        return 62*1e9,62*1e9,5*1e9,0.05,0.19*1e-3,550*1e6,580*1e6,550*1e6,580*1e6,80*1e6,88*1e6,1608
    else:
        raise Exception(f'\'material\' should be set equal to \'UD\' or \'PW\' and not \'{material}\'.')

def neutral_axis_axial(lams,bs):
    """
    Function to get neutral axis of a stringer. designed for stringers with 3 flanges.
    First insert the bottom flange then the web plate and lastly the top flange
    :param lams: list of 3 laminate objects [bottom flange, web, top flange
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: The height of the neutral axis from the bottom
    """
    y = 0
    nom = 0
    denom = 0
    for idx,lam in enumerate(lams):
        lam.general_params()
        A = lam.h*bs[idx] #cross sectional area
        a = np.linalg.inv(lam.A) #inverse of A-matrix (no coupling as layup is symmetrical)
        Ei = 1/a[0,0]/lam.h  #membrane stiffness
        if idx%2==0:
            s = lam.h/2
        else:
            s = bs[idx]/2
        y += s
        nom += Ei*A*y
        denom += Ei*A
        y += s
    return nom/denom

def neutral_axis_bending(lams,bs):
    """
    Function to get neutral axis of a stringer for bending. designed for stringers with 3 flanges.
    First insert the bottom flange then the web plate and lastly the top flange
    :param lams: list of 3 laminate objects [bottom flange, web, top flange
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: The height of the neutral axis from the bottom
    """
    y = 0
    nom = 0
    denom = 0
    for idx,lam in enumerate(lams):
        lam.general_params()
        A = lam.h*bs[idx] #cross sectional area
        d = np.linalg.inv(lam.D) #inverse of D-matrix (no coupling as layup is symmetrical)
        Ebi = 12 / d[0, 0] / lam.h**3  # bending stiffness
        if idx%2==0:
            s = lam.h/2
        else:
            s = bs[idx]/2
        y += s
        nom += Ebi*A*y
        denom += Ebi*A
        y += s
    return nom/denom

def forces(lams,bs,Ftot):
    """
    Function to get the forces ineach flange. designed for stringers with 3 flanges.
    First insert the bottom flange then the web plate and lastly the top flange
    :param lams: list of 3 laminate objects [bottom flange, web, top flange
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: List of 3 forces corresponding to their flanges.
    """

    denom = 0
    for i,lam in enumerate(lams):
        lam.general_params()
        a = np.linalg.inv(lam.A)  # inverse of A-matrix (no coupling as layup is symmetrical)
        Ei = 1 / a[0, 0] / lam.h  # membrane stiffness
        denom += Ei*bs[i]*lam.h

    F_lst = []
    for i,lam in enumerate(lams):
        a = np.linalg.inv(lam.A)  # inverse of A-matrix (no coupling as layup is symmetrical)
        Ei = 1 / a[0, 0] / lam.h  # membrane stiffness
        F_lst.append(Ei*bs[i]*lam.h/denom*Ftot)

    return F_lst

def moments(lams,bs,Mtot):
    """
    Function to get the moments ineach flange. designed for stringers with 3 flanges.
    First insert the bottom flange then the web plate and lastly the top flange
    :param lams: list of 3 laminate objects [bottom flange, web, top flange
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: List of 3 moments corresponding to their flanges.
    """
    y_bar = neutral_axis_bending(lams,bs)
    s = 0
    EI_lst = []
    for i,lam in enumerate(lams):
        lam.general_params()
        d = np.linalg.inv(lam.D)  # inverse of A-matrix (no coupling as layup is symmetrical)
        Ebi = 12 / d[0, 0] / lam.h**3  # membrane stiffness
        A = bs[i]*lam.h
        if i%2==0:
            s += lam.h/2
            di = abs(y_bar-s)
            EI_lst.append(Ebi*(bs[i]*lam.h**3/12+A*di**2))
            s += lam.h / 2
        else:
            s += bs[i]/2
            di = abs(y_bar-s)
            EI_lst.append(Ebi*(bs[i]**3*lam.h/12+A*di**2))
            s += bs[i] / 2
    return Mtot*EI_lst[0],Mtot*EI_lst[1],Mtot*EI_lst[2]

def crippling(t,b,Xc,case='OEF'):
    if case=='OEF':
        sigma_crip = 1.63*Xc/(b/t)**(0.717)
    elif case=='NEF':
        sigma_crip = 11*Xc/(b/t)**1.124
    else:
        raise Exception(f"the varaible 'case' should be equal to 'OEF' or 'NEF'.")
    return sigma_crip

#####-------------------------------------------------------END Helper functions----------------------------------------------------



#####-------------------------------------------------------define material properties----------------------------------------------------
#everything is given in SI units (no mega or giga except density=[kg/m3])
props = material_properties('UD')
Ex = props[0]
Ey = props[1]
Gxy = props[2]
vxy = props[3]
t_ply = props[4]
Xt = props[5]
Xc = props[6]
Yt = props[7]
Yc = props[8]
S = props[9]
S_il = props[10]
density = props[11]
#####-------------------------------------------------------END define material properties----------------------------------------------------


#creat the bottom flange laminate
angles = get_angles([45,-45,0,0,45,-45],1)
#get all z_coordinates
z_coord = []
n_halfplies = len(angles)/2
for i in range(len(angles)):
    z_coord.append([(-n_halfplies+i)*t_ply,t_ply*(-n_halfplies+i+1)])

#Create laminate as an object and add individual plies
flange_bot = Laminate()
for idx,angle in enumerate(angles):
    flange_bot.add_plies(Lamina(angle*m.pi/180,Ex,Ey,vxy,Gxy,z_coord[idx][0],z_coord[idx][1])) #from bottom to top


#creat the top flange laminate
angles = get_angles([45,-45,0,0,0,0,45,-45],1)
#get all z_coordinates
z_coord = []
n_halfplies = len(angles)/2
for i in range(len(angles)):
    z_coord.append([(-n_halfplies+i)*t_ply,t_ply*(-n_halfplies+i+1)])
#Create laminate as an object and add individual plies
flange_top = Laminate()
for idx,angle in enumerate(angles):
    flange_top.add_plies(Lamina(angle*m.pi/180,Ex,Ey,vxy,Gxy,z_coord[idx][0],z_coord[idx][1])) #from bottom to top


#creat the web plate laminate
angles = get_angles([45,-45,45,-45],1)
#get all z_coordinates
z_coord = []
n_halfplies = len(angles)/2
for i in range(len(angles)):
    z_coord.append([(-n_halfplies+i)*t_ply,t_ply*(-n_halfplies+i+1)])
#Create laminate as an object and add individual plies
web = Laminate()
for idx,angle in enumerate(angles):
    web.add_plies(Lamina(angle*m.pi/180,Ex,Ey,vxy,Gxy,z_coord[idx][0],z_coord[idx][1])) #from bottom to top

F1_lst = []
F2_lst = []
F3_lst = []
M1_lst = []
M2_lst = []
M3_lst = []
V1_lst = []
V2_lst = []
V3_lst = []
for theta in range(91):
    N,M,V = get_main_forces(theta)
    force = forces([flange_bot,web,flange_top],[0.2,0.35,0.2],N)
    moment = moments([flange_bot,web,flange_top],[0.2,0.35,0.2],M)

    F1_lst.append(force[0])
    F2_lst.append(force[1])
    F3_lst.append(force[2])
    M1_lst.append(moment[0])
    M2_lst.append(moment[1])
    M3_lst.append(moment[2])

plt.subplot(131)
plt.title('Forces')
plt.plot(range(91),F1_lst,label='Force 1')
plt.plot(range(91),F2_lst,label='Force 2')
plt.plot(range(91),F3_lst,label='Force 3')
plt.legend()
plt.grid(True)

plt.subplot(132)
plt.title('Moments')
plt.plot(range(91),M1_lst,label='Moment 1')
plt.plot(range(91),M2_lst,label='Moment 2')
plt.plot(range(91),M3_lst,label='Moment 3')
plt.legend()
plt.grid(True)

plt.subplot(132)

plt.show()