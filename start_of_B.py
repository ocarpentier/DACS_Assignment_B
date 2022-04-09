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

def crippling(t,b,Xc,case='OEF'):
    if case=='OEF':
        sigma_crip = 1.63*Xc/(b/t)**(0.717)
    elif case=='NEF':
        sigma_crip = 11*Xc/(b/t)**1.124
    else:
        raise Exception(f"the varaible 'case' should be equal to 'OEF' or 'NEF'.")
    return sigma_crip

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
    k = Nxy/Nx
    N01 = m.pi**2/a**2*(D11+2*(D12+2*D66)*a**2/b**2+D22*a**4/b**4)/(2-8192/81*a**2/b**2/m.pi**4*k**2)*(5+np.sqrt(9+65536*a**2*k**2/(81*m.pi**4*b**2)))
    N02 = m.pi ** 2 / a ** 2 * (D11 + 2 * (D12 + 2 * D66) * a ** 2 / b ** 2 + D22 * a ** 4 / b ** 4) / (
                2 - 8192 / 81 * a ** 2 / b ** 2 / m.pi ** 4 * k ** 2) * (
                     5 - np.sqrt(9 + 65536 * a ** 2 * k ** 2 / (81 * m.pi ** 4 * b ** 2)))
    if N01<N02:
        N0 = N01
    else:
        N0 = N02
    return N0>-Nx

def Ixx_accent(lams,bs):
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
    Function to calculate the shear in a c_channel beam using section 25.4.2 SAD book
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

def weight(lams,bs,density):
    Area = 0
    L = 1.75*2*m.pi
    for idx,lam in enumerate(lams):
        Area += lam.h*bs[idx]
    return L*Area*density
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

#####-------------------------------------------------------Define Laminates of flanges and web----------------------------------------------------
#creat the bottom flange laminate
angles = get_angles([45,90,-45,20,-20,0,0,90,0,0,45,90,-45,0,0,90,0,0,-15,15,0,0,90,0,0,45,90,-45,0,0,90,0,0,-15,90,15,0,0,0,0,90,90,90,90,0,0,0,0,45,90,-45,0,0],2)

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
angles = get_angles([45,90,-45,20,-20,0,0,90,0,0,45,90,-45,0,0,90,0,0,-15,15,0,0,90,0,0,45,90,-45,0,0,90,0,0,-15,90,15,0,0,0,0,90,90,90,90,0,0,0,0,45,90,-45,0,0],2)

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
angles = get_angles([45,-45,90,20,-20,45,90,-45,0,0,45,-45],1)

#get all z_coordinates
z_coord = []
n_halfplies = len(angles)/2
for i in range(len(angles)):
    z_coord.append([(-n_halfplies+i)*t_ply,t_ply*(-n_halfplies+i+1)])

#Create laminate as an object and add individual plies
web = Laminate()
for idx,angle in enumerate(angles):
    web.add_plies(Lamina(angle*m.pi/180,Ex,Ey,vxy,Gxy,z_coord[idx][0],z_coord[idx][1])) #from bottom to top

#calculates general parameters needed for later calculations
flange_top.general_params(),flange_bot.general_params(),web.general_params()
#####-------------------------------------------------------END Define Laminates of flanges and web----------------------------------------------------


#####-------------------------------------------------------Interactive streamlit display----------------------------------------------------
st.write('Thicknesses of the plates are',flange_top.h,flange_bot.h,web.h)
with st.sidebar:
    b1,b2,b3 = st.slider('b bottom flange [cm]',1,50)/100,st.slider('b web [cm]',1,35)/100,st.slider('b top flange [cm]',1,50)/100
# b1,b2,b3 = 0.3,0.35,0.3 # Comment this line if you do not want streamlit
bs = [b1,b2,b3] #list of widths of flanges and web
lams = [flange_bot,web,flange_top] #list of laminate objects

# Get angle for where to calculate
st.title('Show stresses in function of theta')
with st.sidebar:
    theta = st.slider('theta',0,90)
# theta = 40 #Comment this line if you do not want streamlit

#get forces and moments at corresponding theta
N,M,V = get_main_forces(theta)
force = forces(lams,bs,N)
moment = moments(lams,bs,M)
shear_flow = shear_flow_C2(lams,bs,V) #calculates the shearflow.

k_beam = 1 / radius(lams, bs, M) #determine the curvature using lecture 4 slides
st.write(f'The bending curvature is {k_beam}.')
st.subheader('Stress throughout the layers of the bottom flange')

#calculate the strains from axial load and shear flow
strains = flange_bot.calc_strains(np.array([force[0]/bs[0],0,shear_flow[0:50].max()]),np.array([0,0,0]))
#manually add the curvature to the strains 
flange_bot.strains_curvs[0] = (lams[0].h/2-neutral_axis_bending(lams, bs))*k_beam
flange_bot.strains_curvs[3] = k_beam
flange_bot.strains_curvs[4::] = 0

#calculate the stresses throughout the laminate and plot it
stresses,z = flange_bot.stress(points_per_ply=20)
fig,ax = plt.subplots(1,3)
ax[0].plot(stresses[0],z)
ax[0].axvline(Xt,linestyle='dashed',color='r')
ax[0].axvline(-Xc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_1$')
ax[1].plot(stresses[1],z)
ax[1].axvline(Yt,linestyle='dashed',color='r')
ax[1].axvline(-Yc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_2$')
ax[2].plot(stresses[2],z)
ax[2].axvline(S,linestyle='dashed',color='r')
ax[2].axvline(-S,linestyle='dashed',color='r')
ax[0].grid(True)
ax[2].grid(True)
ax[1].grid(True)
plt.xlabel('$\sigma_6$')
# plt.show() #comment this line in to get plots in IDE
st.pyplot(fig)

st.subheader('Stress throughout the layers of the web')
#calculate the strains from axial load and shear flow
strains = web.calc_strains(np.array([force[1]/bs[1],0,shear_flow[50:100].max()]),np.array([0,0,0]))
#manually add the curvature to the strains
web.strains_curvs[3::] = 0
web.strains_curvs[5] = k_beam #find out which curvature this realy is

stresses,z = web.stress(points_per_ply=20)
fig,ax = plt.subplots(1,3)
ax[0].plot(stresses[0],z)
ax[0].axvline(Xt,linestyle='dashed',color='r')
ax[0].axvline(-Xc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_1$')
ax[1].plot(stresses[1],z)
ax[1].axvline(Yt,linestyle='dashed',color='r')
ax[1].axvline(-Yc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_2$')
ax[2].plot(stresses[2],z)
ax[2].axvline(S,linestyle='dashed',color='r')
ax[2].axvline(-S,linestyle='dashed',color='r')
plt.xlabel('$\sigma_6$')
ax[0].grid(True)
ax[2].grid(True)
ax[1].grid(True)
# plt.show() #comment this line in to get plots in IDE
st.pyplot(fig)


st.subheader(f'Stress throughout the layers of the top flange \n with plies {len(flange_top.rotations)} plies with orientations: {flange_top.rotations}')
strains = flange_top.calc_strains(np.array([force[2]/bs[2],0,shear_flow[100::].max()]),np.array([moment[2]/bs[2],0,0]))
#manually add the curvature to the strains
flange_top.strains_curvs[0] = (lams[0].h+bs[1]+lams[2].h/2-neutral_axis_bending(lams, bs))*k_beam
flange_top.strains_curvs[3] = k_beam
flange_top.strains_curvs[4::] = 0

stresses,z = flange_top.stress(points_per_ply=20)
fig,ax = plt.subplots(1,3)
ax[0].plot(stresses[0],z)
ax[0].axvline(Xt,linestyle='dashed',color='r')
ax[0].axvline(-Xc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_1$')
ax[1].plot(stresses[1],z)
ax[1].axvline(Yt,linestyle='dashed',color='r')
ax[1].axvline(-Yc,linestyle='dashed',color='r')
plt.xlabel('$\sigma_2$')
ax[2].plot(stresses[2],z)
ax[2].axvline(S,linestyle='dashed',color='r')
ax[2].axvline(-S,linestyle='dashed',color='r')
plt.xlabel('$\sigma_6$')
ax[0].grid(True)
ax[2].grid(True)
ax[1].grid(True)
# plt.show() #comment this line in to get plots in IDE
st.pyplot(fig)


st.subheader('Moments of the flanges and webs')
fig,ax = plt.subplots()
ax.bar([1,2,3],moment)
st.pyplot(fig)
#####-------------------------------------------------------END Interactive streamlit display----------------------------------------------------