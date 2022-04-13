from CLT import Lamina,Laminate,get_angles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import streamlit as st

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
    :param lams: list of 3 laminate objects [bottom flange, web, top flange]
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
    return Mtot*EI_lst[0]/sum(EI_lst),Mtot*EI_lst[1]/sum(EI_lst),Mtot*EI_lst[2]/sum(EI_lst)

def radius(lams,bs,Mtot):
    EI_eq = equivalent_EI(lams,bs)
    return (EI_eq/Mtot)

def crippling_stress(t,b,Xc,case='OEF'):
    if case=='OEF':
        sigma_crip = 1.63*Xc/(b/t)**(0.717)
    elif case=='NEF':
        sigma_crip = 11*Xc/(b/t)**1.124
    else:
        raise Exception(f"the varaible 'case' should be equal to 'OEF' or 'NEF'.")
    return sigma_crip

def crippling_load(lam,b,case='OEF'):
    """
    Function that calcs Nxcrit for cripplin assumption that it is a very long plate is made.
    :param lam: laminate object
    :param b: the width of the plate
    :param case: OEF or NEF it is about the boundary conditions
    :return: critical crippling load
    """
    if case == 'OEF':
        Nxcrit = 12 * lam.D[2, 2] / bs[idx] ** 2
    elif case == 'NEF':
          Nxcrit = 2*m.pi**2/b**2*(np.sqrt(lam.D[0,0]*lam.D[1,1])+(lam.D[0,1]+2*lam.D[2,2]))
    else:
        raise Exception(f"the varaible 'case' should be equal to 'OEF' or 'NEF'.")
    return Nxcrit

def equivalent_EI(lams, bs):
    """
    Function to calculate the equivalent bending stiffness (EI)eq
    :param lams: list of 3 Laminates.
    :param bs: list of thei corresponding width.
    :return: (EI)eq
    """
    y_bar = neutral_axis_bending(lams, bs)
    s = 0
    EI_lst = []
    for i, lam in enumerate(lams):
        lam.general_params()
        d = np.linalg.inv(lam.D)  # inverse of A-matrix (no coupling as layup is symmetrical)
        Ebi = 12 / d[0, 0] / lam.h ** 3  # membrane stiffness
        A = bs[i] * lam.h
        if i % 2 == 0:
            s += lam.h / 2
            di = abs(y_bar - s)
            EI_lst.append(Ebi * (bs[i] * lam.h ** 3 / 12 + A * di ** 2))
            s += lam.h / 2
        else:
            s += bs[i] / 2
            di = abs(y_bar - s)
            EI_lst.append(Ebi * (bs[i] ** 3 * lam.h / 12 + A * di ** 2))
            s += bs[i] / 2

    return sum(EI_lst)

def Ixx(lams,bs):
    """
    Moment of inertia for isotropic structures
    :param lams: list of 3 laminate objects [bottom flange, web, top flange]
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: Moment of inertia
    """
    y_bar = neutral_axis_bending(lams, bs)
    s = 0
    Ixx_lst = []
    for i, lam in enumerate(lams):
        A = bs[i] * lam.h
        if i % 2 == 0:
            s += lam.h / 2
            di = abs(y_bar - s)
            Ixx_lst.append(bs[i] * lam.h ** 3 / 12 + A * di ** 2)
            s += lam.h / 2
        else:
            s += bs[i] / 2
            di = abs(y_bar - s)
            Ixx_lst.append(bs[i] ** 3 * lam.h / 12 + A * di ** 2)
            s += bs[i] / 2
    return sum(Ixx_lst)

def shear_flow_C(lams,bs,V):
    """
    Uses equation q=V*Q/I but seems wrong
    :param lams:
    :param bs:
    :param V:
    :return:
    """
    y_bar = neutral_axis_bending(lams, bs) #bending neutral axis because shear is related to bending
    I_xx = Ixx(lams,bs)
    q_lst  = [[],[],[]]
    for j in range(51):
        A = j/50*bs[2]*lams[2].h
        y = bs[1]+lams[0].h+lams[2].h/2-y_bar
        Q = A*y
        q_lst[0].append(Q*V/I_xx)
    q_prev1 = q_lst[0][-1]
    for j in range(51):
        A = j/50*bs[1]*lams[1].h
        y = bs[1]-j/50*bs[1]/2+lams[0].h-y_bar
        Q = A*y
        q_lst[1].append(Q*V/I_xx+q_prev1)
        # q_lst[1].append(y)
    q_prev2 = q_lst[1][-1]
    for j in range(51):
        A = j / 50 * bs[0] * lams[0].h
        y = lams[0].h/2 - y_bar
        Q = A * y
        q_lst[2].append(Q * V / I_xx+q_prev2)
    return q_lst

def buckling(lam,a,b,Nx,Nxy):

    D11 = lam.D[0,0]
    D12 = lam.D[0,1]
    D66 = lam.D[2,2]
    D22 = lam.D[1,1]
    a = 2
    AR = a/b
    if Nx!=0:
        k = Nxy/Nx
        N01 = m.pi**2/a**2*(D11+2*(D12+2*D66)*a**2/b**2+D22*a**4/b**4)/(2-8192/81*a**2/b**2/m.pi**4*k**2)*(5+np.sqrt(9+65536*a**2*k**2/(81*m.pi**4*b**2)))
        N02 = m.pi ** 2 / a ** 2 * (D11 + 2 * (D12 + 2 * D66) * a ** 2 / b ** 2 + D22 * a ** 4 / b ** 4) / (
                    2 - 8192 / 81 * a ** 2 / b ** 2 / m.pi ** 4 * k ** 2) * (
                         5 - np.sqrt(9 + 65536 * a ** 2 * k ** 2 / (81 * m.pi ** 4 * b ** 2)))
        if abs(N01)<abs(N02):
            N0 = N01
        else:
            N0 = N02
        return N0<Nx
    else:
        return False

def Ixx_accent(lams,bs):
    """
    Function to calculate the moment of inertia for beam including the different stiffnesses.
    (chapter 25, book "Aircraft structures for engineering students")
    :param lams: list of 3 laminate objects [bottom flange, web, top flange]
    :param bs: the heihgt of the corresponding flanges as seen as cross section.
    :return: moment of inertia
    """
    y_bar = neutral_axis_bending(lams, bs)
    s = 0
    Ixx_lst = []
    for i, lam in enumerate(lams):
        d  = np.linalg.inv(lam.D)
        Ebi = 12 / d[0, 0] / lam.h ** 3  # bending stiffness
        A = bs[i] * lam.h
        if i % 2 == 0:
            s += lam.h / 2
            di = abs(y_bar - s)
            Ixx_lst.append((bs[i] * lam.h ** 3 / 12 + A * di ** 2)*Ebi)
            s += lam.h / 2
        else:
            s += bs[i] / 2
            di = abs(y_bar - s)
            Ixx_lst.append((bs[i] ** 3 * lam.h / 12 + A * di ** 2)*Ebi)
            s += bs[i] / 2
    return sum(Ixx_lst)

def shear_flow_C2(lams,bs,V):
    """
    Function to calculate the shear in a c_channel beam (using section 25.4.2 , book "Aircraft structures for engineering students")
    :param lams: list of laminates, bottom,web,top
    :param bs: list of corresponding widths
    :param V: Shear force
    :return: list of the shearflow
    """
    I_xx = Ixx_accent(lams,bs)
    s_arr1 = np.linspace(0,bs[0],50)
    s_arr2 = np.linspace(0,bs[1],50)
    s_arr3 = np.linspace(0,bs[2],50)
    d1 = np.linalg.inv(lams[0].D)
    Ebi1 = 12 / d1[0, 0] / lams[0].h ** 3  # bending stiffness
    d2 = np.linalg.inv(lams[1].D)
    Ebi2 = 12 / d2[0, 0] / lams[1].h ** 3  # bending stiffness
    d3 = np.linalg.inv(lams[2].D)
    Ebi3 = 12 / d3[0, 0] / lams[2].h ** 3  # bending stiffness
    q_arr = np.zeros(150)
    y_bar = neutral_axis_bending(lams, bs)
    y1 = -y_bar+lams[0].h/2
    y3 = bs[1]+lams[0].h+lams[2].h/2-y_bar
    q_arr[0:50] = Ebi1*V/I_xx*lams[0].h*y1*s_arr1
    q_arr[50:100] = Ebi2*V/I_xx*lams[1].h*(0.5*s_arr2**2+(lams[0].h-y_bar)*s_arr2)+q_arr[49]
    q_arr[100::] = Ebi3 * V / I_xx * lams[2].h * y3 * s_arr3+q_arr[99]
    return q_arr

def shear_flow_I(lams,bs,V):
    """
    Function to calculate the shear in a c_channel beam (using section 25.4.2 , book "Aircraft structures for engineering students")
    :param lams: list of laminates, bottom,web,top
    :param bs: list of corresponding widths
    :param V: Shear force
    :return: list of the shearflow
    """
    I_xx = Ixx_accent(lams,bs)
    s_arr1 = np.linspace(0,bs[0]/2,10)
    s_arr2 = np.linspace(0,bs[1],20)
    s_arr3 = np.linspace(0,bs[2]/2,10)
    d1 = np.linalg.inv(lams[0].D)
    Ebi1 = 12 / d1[0, 0] / lams[0].h ** 3  # bending stiffness
    d2 = np.linalg.inv(lams[1].D)
    Ebi2 = 12 / d2[0, 0] / lams[1].h ** 3  # bending stiffness
    d3 = np.linalg.inv(lams[2].D)
    Ebi3 = 12 / d3[0, 0] / lams[2].h ** 3  # bending stiffness

    q_arr = np.zeros(60)
    y_bar = neutral_axis_bending(lams, bs)
    y1 = -y_bar+lams[0].h/2
    y3 = bs[1]+lams[0].h+lams[2].h/2-y_bar

    q_arr[0:10] = Ebi1*V/I_xx*lams[0].h*y1*s_arr1
    q_arr[10:20] = np.flip(q_arr[0:10])
    q_arr[20:40] = Ebi2*V/I_xx*lams[1].h*(0.5*s_arr2**2+(lams[0].h-y_bar)*s_arr2)+q_arr[10]
    q_arr[50:60] = Ebi3 * V / I_xx * lams[2].h * y3 * s_arr3+q_arr[39]/2
    q_arr[40:50] = np.flip(q_arr[50:60])
    return q_arr

def bending_stress_web(lams,bs,M):
    y_bar = neutral_axis_bending(lams,bs)
    I_xx = Ixx_accent(lams,bs)
    web = lams[1]
    web.general_params()
    d = np.linalg.inv(web.D)  # inverse of A-matrix (no coupling as layup is symmetrical)
    Ebi = 12 / d[0, 0] / web.h ** 3  # membrane stiffness
    y_arr = np.linspace(lams[0].h-y_bar,lams[0].h+bs[1]-y_bar,20)

    sigma_z = Ebi*M/I_xx*y_arr
    return sigma_z

def bending_stress_flange(lams,bs,M,idx):
    y_bar = neutral_axis_bending(lams,bs)
    I_xx = Ixx_accent(lams,bs)
    flange = lams[idx]
    lams[1].general_params()
    d = np.linalg.inv(flange.D)  # inverse of A-matrix (no coupling as layup is symmetrical)
    Ebi = 12 / d[0, 0] / flange.h ** 3  # membrane stiffness
    if idx==0:
        y_arr = lams[0].h/2-y_bar
    else:
        y_arr = lams[0].h - y_bar+lams[1].h/2+bs[1]

    sigma_z = Ebi*M/I_xx*y_arr
    return sigma_z

def weight(lams,bs,density):
    Area = 0
    L = 1.75*2*m.pi
    for idx,lam in enumerate(lams):
        Area += lam.h*bs[idx]
    return L*Area*density

def make_lam(angles):
    """
    Takes in the orientations and gives a laminate object
    :param angles:
    :return:
    """
    # get all z_coordinates
    z_coord = []
    n_halfplies = len(angles) / 2
    for i in range(len(angles)):
        z_coord.append([(-n_halfplies + i) * t_ply, t_ply * (-n_halfplies + i + 1)])

    # Create laminate as an object and add individual plies
    obj = Laminate()
    for idx, angle in enumerate(angles):
        obj.add_plies(
            Lamina(angle * m.pi / 180, Ex, Ey, vxy, Gxy, z_coord[idx][0], z_coord[idx][1]))  # from bottom to top
    return obj

def get_signed_abs_max(arr):
    abs_arr = np.abs(arr)
    max = abs_arr.max()
    if max==arr.max():
        return max
    else:
        return -max
def buckling_pure_shear(lam,ba,b,Nxy):
    D11 = lam.D[0, 0]
    D12 = lam.D[0, 1]
    D66 = lam.D[2, 2]
    D22 = lam.D[1, 1]
    AR = a / b
    Nxycrit = 9*m.pi**4*b/32/a**3*(D11+2*(D12+2*D66)*a**2/b**2+D22*AR**4)

    return Nxycrit
#####-------------------------------------------------------END Helper functions----------------------------------------------------



#####-------------------------------------------------------define material properties----------------------------------------------------
#everything is given in SI units (no mega or giga except density=[kg/m3])
knockdown = 0.8**2*0.65 #see lecture 1
props = material_properties('UD')
Ex = props[0]
Ey = props[1]
Gxy = props[2]
vxy = props[3]
t_ply = props[4]
Xt = props[5]*knockdown
Xc = props[6]*knockdown
Yt = props[7]*knockdown
Yc = props[8]*knockdown
S = props[9]*knockdown
S_il = props[10]*knockdown
density = props[11]
#####-------------------------------------------------------END define material properties----------------------------------------------------


#####-------------------------------------------------------Find best layup/bs combo at theta==90-----------------------------
base_sample = [[45,90,-45],[0,0],[0,0]]
angles_og = [[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
,[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0]
,[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
weight_lst = []
layup_lst = []
bs_lst = []
theta = 0
angles = []

bs = [48/100,35/100,23/100]
b1 = 0.48
b2 = 0.35
b3 = 0.23

idx_base = 0
angles = angles_og.copy()
failure = True
while failure:
    failure_lst = []
    #make 3 laminates and put them in a list
    lams = [make_lam(get_angles(angles[0],1)),make_lam(get_angles(angles[1],1)),make_lam(get_angles(angles[2],1))]

    #evaluate the forces at theta=90deg
    N, M, V = get_main_forces(theta)
    force = forces(lams, bs, N)
    moment = moments(lams, bs, M)
    shear_flow = shear_flow_I(lams, bs, V)  # calculates the shearflow.

    k_beam = 1 / radius(lams, bs, M)  # determine the curvature using lecture 4 slides

    #evaluat stresses for each laminate togethher with buckling criteria
    for idx,lam in enumerate(lams):
        if idx != 1:
            pass
            # strains = lam.calc_strains(np.array([force[idx] / bs[idx], 0, 0]),
            #                                   np.array([0, 0, 0]))
            # # manually add the curvature to the strains
            # lam.strains_curvs[0] = (lams[0].h / 2 - neutral_axis_bending(lams, bs)) * k_beam * (1-2*idx)
            # lam.strains_curvs[3] = k_beam
            # lam.strains_curvs[4::] = 0
            #
            # #check for stresses/Hashin failure criteria
            # stresses,z = lam.stress(points_per_ply=10)
            # hashin,mode = lam.Hashin_failure(Xt,Yt,Xc,Yc,S,insitu=False)
            # if hashin:
            #     failure_lst.append(True)
            #
            #     if mode == 'FF':
            #         # print('Hashin FF')
            #         angles[idx] = angles[idx]+[0,0,0,0]
            #         angles[2 - idx] = angles[2 - idx] + [0, 0,0,0]
            #     else:
            #         # print('Hashin IFF',idx)
            #         angles[idx] = angles[idx]+[90,45,0,-45]
            #         angles[2-idx] = angles[2-idx] + [0, 0,0,0]
            # else:
            #     failure_lst.append(False)
            #
            #
            # if lam.strains_curvs[0]<0:
            #     bending_load = bending_stress_flange(lams, bs, M, idx)*lam.h
            #     #check for buckling
            #     buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx]/2, (force[idx]/bs[idx]+bending_load)/knockdown,0)
            #     if buckling_crit:
            #         print('Buck_fail')
            #         failure_lst.append(True)
            #         angles[idx] = angles[idx]+[45,90,-45]
            #         angles[2 - idx] = angles[2 - idx] + [0, 0]
            #     else:
            #         failure_lst.append(False)
            #     #check for crippling
            #     Nx_crit = crippling_load(lam,bs[idx],case='OEF')
            #     if Nx_crit<(force[idx]/bs[idx]+bending_load):
            #         print('crippling')
            #         failure_lst.append(True)
            #         angles[idx] = [45, 90, -45] + angles[idx]
            #         angles[2 - idx] = angles[2 - idx] + [0, 0]
            #     else:
            #         failure_lst.append(False)
            # else:
            #     failure_lst.append(False)
            #     failure_lst.append(False)

        #check the web plate.
        else:
            bend_load = bending_stress_web(lams, bs, M)*lam.h #the maximum load induced by bending ths is in [N/m]
            strains = lam.calc_strains(
                np.array([force[idx] / bs[idx] + get_signed_abs_max(bend_load), 0, V/lam.h]),
                np.array([0, 0, 0])) #the Nx has the load from bending added

            # check for stresses/Hashin failure criteria
            stresses, z = lam.stress(points_per_ply=10)
            hashin, mode = lam.Hashin_failure(Xt, Yt, Xc, Yc, S, insitu=False)
            if hashin:
                print('check')
                failure_lst.append(True)
                if mode == 'FF':
                    angles[idx] = angles[idx] + [0, 0, 0, 0]
                else:
                    angles[idx] = angles[idx] + [45, 90, -45]

            else:
                failure_lst.append(False)

            # check for buckling
            buck_force = force[idx]/bs[idx] + bend_load.min()*lam.h/2
            buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx], buck_force / knockdown,
                                     V/lam.h / knockdown)
            if buckling_crit:
                print('check')
                failure_lst.append(True)
                angles[idx] = angles[idx] + [45, 90, -45]
            else:
                failure_lst.append(False)
            # check for crippling
            Nx_crit = crippling_load(lam, bs[idx], case='NEF')
            if Nx_crit < (force[idx]/bs[idx]+ bend_load.min()*lam.h):
                print('check')
                failure_lst.append(True)
                angles[idx] = [45, 90, -45] + angles[idx]
            else:
                failure_lst.append(False)
    # print(f'N_plies: {len(angles[0]),len(angles[1]),len(angles[2])}')
    # print(failure_lst)
    if not any(failure_lst):
        failure = False

print(len(angles[1]))
print(angles[1])
#
# #get index of lowest weight
# idx_opt = weight_lst.index(min(weight_lst))
# print(layup_lst[idx_opt])
# print(bs_lst[idx_opt])
# print('Thickness of the laminates: ',len(layup_lst[idx_opt][0])*t_ply*2,len(layup_lst[idx_opt][1])*t_ply*2,len(layup_lst[idx_opt][2])*t_ply*2)
# print('Number of plies: ',len(layup_lst[idx_opt][0])*2,len(layup_lst[idx_opt][1])*2,len(layup_lst[idx_opt][2])*2)
#
#
# plt.scatter(range(len(weight_lst)),weight_lst)
# plt.show()
# #####-------------------------------------------------------END Find best layup/bs combo at theta==90-----------------------------
#
#
# ####--------------------------------------------------------Check layup and adapt to make it possible-----------------------------------------
#
# angles = layup_lst[idx_opt]
angles = [[ -45, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
, [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0],
[-45, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 0, 0, 0, 0, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 90, 45, 0, -45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45]

angles_present = []
angles_to_put_in = [-45,45,90]
for layup in angles:
    #check percentages of each kind
    n_el = [layup.count(-45),layup.count(45),layup.count(90),layup.count(0)]
    angles_present.append(n_el)

#from middle to edge:
new_angle = [[],[],[]]
for i in range(3):
    zeros_lst = []
    for j in range(angles_present[i][-1]):
        zeros_lst.append(0)
    count_rep = 0
    count_0 = 0
    idx = 0
    for k in range(len(zeros_lst)):
        if zeros_lst[k]==0:
            count_0 +=1
            new_angle[i].append(0)
            angles_present[i][-1] -=1
        if i==0:
            if count_0==4:
                new_angle[i].append(angles_to_put_in[idx])
                angles_present[i][idx] -=1
                idx += 1
                count_0 = 0
                if idx>2:
                    idx = 0
                    count_rep+=1
        else:
            if count_0==3:
                new_angle[i].append(angles_to_put_in[idx])
                angles_present[i][idx] -=1
                idx += 1
                count_0 = 0
                if idx>2:
                    idx = 0
                    count_rep+=1
    print(count_rep)
print(angles_present)

# print(angles_present)
#
# #put the additional layers in to the laminates
new_angle[2] +=[90]+[-45,90,45]*3
# # new_angle[1].append([45,90,-45]*10+[90])
#
# #add layers for iteration
# new_angle[0] += [45,0,0,0,0,90,0,0,0,0,-45,0,0,0,0,45,0,0,0,0-45]#,0,0,0,0,45
# new_angle[2] += [0,0,0,0,45,0,0,0,0,90,-45] #,0,0,0,0,-45
# new_angle[1] += [45,90,-45]*40
overlap = [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45]
new_angle[0] += overlap
# new_angle[1] += overlap
new_angle[2] += overlap
#
new_angle[0].reverse()
# new_angle[1].reverse()
new_angle[2].reverse()
#
# print(len(new_angle[1]))
new_angle045 = [new_angle[0],[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45,0, 0,0,0, 90, 0, 0,0,0,45, 0, 0,0,0, 90, 0, 0, 0, 45, 0, 0, 0,0, -45, 0, 0],new_angle[2]]
new_angle4590 = [[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0,45],
                 [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0, 0, 45, 90, -45,  0, 0],
                 [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45,  0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0]]
# ####--------------------------------------------------------Check stresses with new layup-----------------------------------------
# for theta in range(91):
#     bs = [48/100,35/100,23/100]
#
#     failure_lst = []
#     #make 3 laminates and put them in a list
#     if theta<45:
#         lams = [make_lam(get_angles(new_angle045[0],1)),make_lam(get_angles(new_angle045[1],1)),make_lam(get_angles(new_angle045[2],1))]
#     else:
#         lams = [make_lam(get_angles(new_angle4590[0], 1)), make_lam(get_angles(new_angle4590[1], 1)),
#                 make_lam(get_angles(new_angle4590[2], 1))]
#
#     #evaluate the forces at theta=90deg
#     N, M, V = get_main_forces(theta)
#     force = forces(lams, bs, N)
#     moment = moments(lams, bs, M)
#
#     k_beam = 1 / radius(lams, bs, M)  # determine the curvature using lecture 4 slides
#
#     #evaluat stresses for each laminate togethher with buckling criteria
#     for idx,lam in enumerate(lams):
#         if idx != 1:
#             strains = lam.calc_strains(np.array([force[idx] / bs[idx], 0, 0]),
#                                               np.array([0, 0, 0]))
#             # manually add the curvature to the strains
#             lam.strains_curvs[0] = (lams[0].h / 2 - neutral_axis_bending(lams, bs)) * k_beam * (1-2*idx)
#             lam.strains_curvs[3] = k_beam
#             lam.strains_curvs[4::] = 0
#
#             #check for stresses/Hashin failure criteria
#             stresses,z = lam.stress(points_per_ply=10)
#             hashin,mode = lam.Hashin_failure(Xt,Yt,Xc,Yc,S,insitu=False)
#             if hashin:
#                 failure_lst.append(True)
#
#                 if mode == 'FF':
#                     print('Hashin FF')
#                     angles[idx] = angles[idx]+[0,0,0,0]
#                     angles[2 - idx] = angles[2 - idx] + [0, 0,0,0]
#                 else:
#                     print('Hashin IFF',idx)
#                     angles[idx] = angles[idx]+[90,45,0,-45]
#                     angles[2-idx] = angles[2-idx] + [0, 0,0,0]
#             else:
#                 failure_lst.append(False)
#
#
#             if lam.strains_curvs[0]<0:
#                 bending_load = bending_stress_flange(lams, bs, M, idx)*lam.h
#                 #check for buckling
#                 buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx]/2, (force[idx]/bs[idx]+bending_load)/knockdown,0)
#                 if buckling_crit:
#                     print('Buck_fail')
#                     failure_lst.append(True)
#                     angles[idx] = angles[idx]+[45,90,-45]
#                     angles[2 - idx] = angles[2 - idx] + [0, 0]
#                 else:
#                     failure_lst.append(False)
#                 #check for crippling
#                 Nx_crit = crippling_load(lam,bs[idx],case='OEF')
#                 if Nx_crit<(force[idx]/bs[idx]+bending_load):
#                     print('crippling')
#                     failure_lst.append(True)
#                     angles[idx] = [45, 90, -45] + angles[idx]
#                     angles[2 - idx] = angles[2 - idx] + [0, 0]
#                 else:
#                     failure_lst.append(False)
#             else:
#                 failure_lst.append(False)
#                 failure_lst.append(False)
#
#         #check the web plate.
#         else:
#             bend_load = bending_stress_web(lams, bs, M)*lam.h #the maximum load induced by bending ths is in [N/m]
#             strains = lam.calc_strains(
#                 np.array([force[idx] / bs[idx] + get_signed_abs_max(bend_load), 0, V/lam.h]),
#                 np.array([0, 0, 0])) #the Nx has the load from bending added
#
#             # check for stresses/Hashin failure criteria
#             stresses, z = lam.stress(points_per_ply=10)
#             hashin, mode = lam.Hashin_failure(Xt, Yt, Xc, Yc, S, insitu=False)
#             if hashin:
#                 failure_lst.append(True)
#                 if mode == 'FF':
#                     print('FF')
#                     # angles[idx] = angles[idx] + [0, 0, 0, 0]
#                 # else:
#                     # print('IFF')
#                     # print('IFF')
#                     # angles[idx] = angles[idx] + [45, 90, -45]
#
#             else:
#                 failure_lst.append(False)
#
#             # check for buckling
#             buck_force = force[idx]/bs[idx] + bend_load.min()*lam.h
#             buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx], buck_force / knockdown,
#                                      V/lam.h / knockdown)
#             if buckling_crit:
#                 failure_lst.append(True)
#                 angles[idx] = angles[idx] + [45, 90, -45]
#             else:
#                 failure_lst.append(False)
#             # check for crippling
#             Nx_crit = crippling_load(lam, bs[idx], case='NEF')
#             if Nx_crit < (force[idx]/bs[idx]+ bend_load.min()*lam.h):
#                 failure_lst.append(True)
#                 angles[idx] = [45, 90, -45] + angles[idx]
#             else:
#                 failure_lst.append(False)
#     # print(f'N_plies: {len(angles[0]),len(angles[1]),len(angles[2])}')
#     # print(failure_lst)
#     if any(failure_lst):
#         print(theta,failure_lst)


##########-----------------------------Check for the 10% rule-----------------------------------
for i in range(3):
    print(len(new_angle045[i])*2)
    print(len(new_angle045[i]) * 2*0.014,'cm')

    print(len(new_angle4590[i]) * 2)
    print(len(new_angle4590[i]) * 2 * 0.014, 'cm')


for layup in new_angle045:
    #check percentages of each kind
    n_el = [layup.count(-45),layup.count(45),layup.count(90),layup.count(0)]
    for i in range(4):
        print(n_el[i]/sum(n_el))

print(len(new_angle045[0])*2)
print(len(new_angle045[1])*2)
print(len(new_angle045[2])*2)



# ###-------------------------------------Check at what load it will fail and for which failure mode-----------------------------------
new_angle045 = [[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0,45],
                [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, -45, 90, 45, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0],
                [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0, 0, 45, 90, -45,  0, 0]]

new_angle4590 = [[45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45, 0, 0, 0, 0,45],
                 [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45, 45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45, 0,0,45, 90, -45,0,0, 45, 90, -45,0,0, 45, 90, -45, 0, 0, 45, 90, -45,  0, 0],
                 [45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, 45, 90, -45, -45, 90, 45, -45, 90, 0, 0, 0, 0, 45, 0, 0, 0, 0, -45,  0, 0, 0, 0, 90, 0, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0, 0, 90, 0, 0, 0, 45, 0, 0, 0, -45, 0, 0]]

for F_ult in range(837000,900000,1000):
    print('Were at Fult: ',F_ult)
    for theta in range(91):
        bs = [48/100,35/100,23/100]

        failure_lst = []
        #make 3 laminates and put them in a list
        if theta<45:
            lams = [make_lam(get_angles(new_angle045[0],1)),make_lam(get_angles(new_angle045[1],1)),make_lam(get_angles(new_angle045[2],1))]
        else:
            lams = [make_lam(get_angles(new_angle4590[0], 1)), make_lam(get_angles(new_angle4590[1], 1)),
                    make_lam(get_angles(new_angle4590[2], 1))]

        #evaluate the forces at theta=90deg
        N, M, V = get_main_forces(theta,F=F_ult)
        force = forces(lams, bs, N)
        moment = moments(lams, bs, M)

        k_beam = 1 / radius(lams, bs, M)  # determine the curvature using lecture 4 slides

        #evaluat stresses for each laminate togethher with buckling criteria
        for idx,lam in enumerate(lams):
            if idx != 1:
                strains = lam.calc_strains(np.array([force[idx] / bs[idx], 0, 0]),
                                                  np.array([0, 0, 0]))
                # manually add the curvature to the strains
                lam.strains_curvs[0] = (lams[0].h / 2 - neutral_axis_bending(lams, bs)) * k_beam * (1-2*idx)
                lam.strains_curvs[3] = k_beam
                lam.strains_curvs[4::] = 0

                #check for stresses/Hashin failure criteria
                stresses,z = lam.stress(points_per_ply=10)
                hashin,mode = lam.Hashin_failure(Xt,Yt,Xc,Yc,S,insitu=False)
                if hashin:
                    failure_lst.append(True)

                    if mode == 'FF':
                        print('Hashin FF')
                        angles[idx] = angles[idx]+[0,0,0,0]
                        angles[2 - idx] = angles[2 - idx] + [0, 0,0,0]
                    else:
                        print('Hashin IFF',idx)
                        angles[idx] = angles[idx]+[90,45,0,-45]
                        angles[2-idx] = angles[2-idx] + [0, 0,0,0]
                else:
                    failure_lst.append(False)


                if lam.strains_curvs[0]<0:
                    bending_load = bending_stress_flange(lams, bs, M, idx)*lam.h
                    #check for buckling
                    buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx]/2, (force[idx]/bs[idx]+bending_load)/knockdown,0)
                    if buckling_crit:
                        print('Buck_fail')
                        failure_lst.append(True)
                        angles[idx] = angles[idx]+[45,90,-45]
                        angles[2 - idx] = angles[2 - idx] + [0, 0]
                    else:
                        failure_lst.append(False)
                    #check for crippling
                    Nx_crit = crippling_load(lam,bs[idx],case='OEF')
                    if Nx_crit<(force[idx]+bending_load*bs[idx]):
                        print('crippling')
                        failure_lst.append(True)
                        angles[idx] = [45, 90, -45] + angles[idx]
                        angles[2 - idx] = angles[2 - idx] + [0, 0]
                    else:
                        failure_lst.append(False)
                else:
                    failure_lst.append(False)
                    failure_lst.append(False)

            #check the web plate.
            else:
                bend_load = bending_stress_web(lams, bs, M)*lam.h #the maximum load induced by bending ths is in [N/m]
                strains = lam.calc_strains(
                    np.array([force[idx] / bs[idx] + get_signed_abs_max(bend_load), 0, V/lam.h]),
                    np.array([0, 0, 0])) #the Nx has the load from bending added

                # check for stresses/Hashin failure criteria
                stresses, z = lam.stress(points_per_ply=10)
                hashin, mode = lam.Hashin_failure(Xt, Yt, Xc, Yc, S, insitu=False)
                if hashin:
                    failure_lst.append(True)
                    if mode == 'FF':
                        print('FF')
                        # angles[idx] = angles[idx] + [0, 0, 0, 0]
                    # else:
                        # print('IFF')
                        # print('IFF')
                        # angles[idx] = angles[idx] + [45, 90, -45]

                else:
                    failure_lst.append(False)

                # check for buckling
                buck_force = force[idx]/bs[idx] + bend_load.min()*lam.h
                buckling_crit = buckling(lam, 1.75 * 2 * m.pi, bs[idx], buck_force / knockdown,
                                         V/lam.h / knockdown)
                if buckling_crit:
                    failure_lst.append(True)
                    angles[idx] = angles[idx] + [45, 90, -45]
                else:
                    failure_lst.append(False)
                # check for crippling
                Nx_crit = crippling_load(lam, bs[idx], case='NEF')
                if Nx_crit < (force[idx]/bs[idx]+ bend_load.min()*lam.h):
                    failure_lst.append(True)
                    angles[idx] = [45, 90, -45] + angles[idx]
                else:
                    failure_lst.append(False)
        # print(f'N_plies: {len(angles[0]),len(angles[1]),len(angles[2])}')
        # print(failure_lst)
        if any(failure_lst):
            print(theta,failure_lst,F_ult)
