import math
import numpy as np
import matplotlib.pyplot as plt
def get_angles(sample,n):
    #todo find out real notation and us of []ns book of kassoupaglo counterdict the slides.
    """
    Function to get the complete list of angles from a sample.
    Following is assumed: [sample]ns (so it is always symmetrical)
    :param sample: sample of angles
    :param n: times it has to be repeated
    :return: list of angles of length n*2 which is symmetrical
    """
    #account for the n and make it appropriate
    angles = []
    for i in range(n):
        for j in range(len(sample)):
            if i%2==0:
                angles.append(sample[j])
            else:
                angles.append(sample[len(sample)-j-1])

    #get the symmetric part
    angles_sym = []

    for sym_angle in reversed(angles):
        angles_sym.append(sym_angle)

    return angles+angles_sym

class Lamina:
    """
    Class that holds all important ply properties. Necessary to make the laminate class.
    """
    def __init__(self,Angle,E1,E2,v12,G12,z_0,z_1):
        #store inputs in objects variables
        self.angle = Angle # angle must be in radians
        self.t = abs(z_1-z_0) # in meters
        self.z_0 = z_0
        self.z_1 = z_1
        self.m = math.cos(self.angle)
        self.n = math.sin(self.angle)

        #make transformation matrices:
        self.T_sigma = np.matrix([[self.m**2,self.n**2,2*self.m*self.n],
                                [self.n ** 2, self.m ** 2,  -2 * self.m * self.n],
                                [ -self.m * self.n,  self.m * self.n,self.m**2-self.n**2]])
        self.T_eps = np.matrix([[self.m**2,self.n**2,self.m*self.n],
                                [self.n ** 2, self.m ** 2,  -self.m * self.n],
                                [ -2*self.m * self.n, 2* self.m * self.n,self.m**2-self.n**2]])
        self.E1 = E1
        self.E2 = E2
        self.v12 = v12
        self.G12 = G12
        self.init()
        self.failed = False
    def init(self):
        if self.E1 !=0:
            self.v21 = self.v12*self.E2/self.E1
        else:
            self.v21=0
        self.Q = 1-self.v12*self.v21

        #Calculate the Q's of the ply at zero angle
        self.Q11 = self.E1*self.Q**(-1)
        self.Q22 = self.E2*self.Q**(-1)
        self.Q66 = self.G12
        self.Q12 = self.v12*self.E2*self.Q**(-1) #contradictory from notes and slides maybe check!


        #Calculate Q's corresponding to the rotate x,y axis system.
        self.Qxx = self.Q11*self.m**4+2*(self.Q12+2*self.Q66)*self.m**2*self.n**2+self.Q22*self.n**4
        self.Qxy = (self.Q11+self.Q22-4*self.Q66)*self.m**2*self.n**2+self.Q12*(self.m**4+self.n**4)
        self.Qyy = self.Q11*self.n**4+2*(self.Q12+2*self.Q66)*self.m**2*self.n**2+self.Q22*self.m**4
        self.Qxs = (self.Q11-self.Q12-2*self.Q66)*self.n*self.m**3+(self.Q12-self.Q22+2*self.Q66)*self.n**3*self.m
        self.Qys = (self.Q11 - self.Q12 - 2 * self.Q66) * self.m * self.n ** 3 + (
                    self.Q12 - self.Q22 + 2 * self.Q66) * self.m ** 3 * self.n
        self.Qss = (self.Q11+self.Q22-2*self.Q12-2*self.Q66)*self.n**2*self.m**2 + self.Q66*(self.n**4+self.m**4)

        #Calculate Unvaried parameters of stiffness and compliance #TODO check for sure that these are the same!
        self.U1 = 1/8*(3*self.Q11+3*self.Q22+2*self.Q12+4*self.Q66)
        self.U2 = 1/2*(self.Q11-self.Q22)
        self.U3 = 1/8*(self.Q11+self.Q22-2*self.Q12-4*self.Q66)
        self.U4 = 1 / 8 * (self.Q11 + self.Q22 + 6 * self.Q12 - 4 * self.Q66)
        self.U5 = 1 / 8 * (self.Q11 + self.Q22 - 2 * self.Q12 + 4 * self.Q66)

        #Calculate the stiffness parameters with these unvaried parameters.
        self.Sxx = self.U1 + self.U2*math.cos(2*self.angle)+self.U3*math.cos(4*self.angle)
        self.Syy = self.U1 - self.U2*math.cos(2*self.angle)+self.U3*math.cos(4*self.angle)
        self.Sss = self.U5 - 4*self.U3*math.cos(2*self.angle)
        self.Sxy = self.U4-self.U3*math.cos(4*self.angle)
        self.Sxs = self.U2*math.sin(2*self.angle)+2*self.U3*math.sin(4*self.angle)
        self.Sys = self.U2 * math.sin(2 * self.angle) - 2 * self.U3 * math.sin(4 * self.angle)

        # make a matrix out of the Qij's and Sij's for laminate calculations
        self.Qijs = np.matrix([[self.Qxx,self.Qxy,self.Qxs],
                              [self.Qxy,self.Qyy,self.Qys],
                              [self.Qxs,self.Qys,self.Qss]])

        self.Sijs = np.matrix([[self.Sxx, self.Sxy, self.Sxs],
                              [self.Sxy, self.Syy, self.Sys],
                              [self.Sxs, self.Sys, self.Sss]])
        self.S_local = np.linalg.inv(self.T_eps)*self.Sijs*np.linalg.inv(self.T_sigma)




    def calc_stress(self,strains_curvs,n_points=50,coordinates='local',middle=None):
        self.z = np.linspace(self.z_0,self.z_1,n_points)
        if middle ==True:
            self.z = (self.z_0+self.z_1)/2
        strains = strains_curvs[0:3]
        curvatures = strains_curvs[3::]
        # # get strains throughout ply for failure criteria
        # self.ply_strain1 = strains[0]+curvatures[0]*self.z
        # self.ply_strain2 = strains[1]+curvatures[1]*self.z
        # self.ply_strainss = strains[2]+curvatures[2]*self.z
        strains_x,strains_y,strains_s = strains[0]+curvatures[0]*self.z,\
                                        strains[1]+curvatures[1]*self.z,\
                                        strains[2]+curvatures[2]*self.z
        if coordinates == 'local':
            self.strains_ply = self.T_eps[0, 0] * strains_x + self.T_eps[0, 1] * strains_y + self.T_eps[0, 2] * strains_s, \
                                     self.T_eps[1, 0] * strains_x + self.T_eps[1, 1] * strains_y + self.T_eps[1, 2] * strains_s, \
                                     self.T_eps[2, 0] * strains_x + self.T_eps[2, 1] * strains_y + self.T_eps[2, 2] * strains_s
        else:
            self.strains_ply = strains_x,strains_y,strains_s

        stresses_s = self.Qijs*strains
        strresses_c = self.Qijs*curvatures



        self.sigma_x, self.sigma_y, self.sigma_xy = np.array(stresses_s[0]+strresses_c[0]*self.z),\
                                                    np.array(stresses_s[1]+strresses_c[1]*self.z),\
                                                    np.array(stresses_s[2]+strresses_c[2]*self.z)


        #Calculate the local stresses out of those based on the global coordinate system
        self.sigma_1, self.sigma_2, self.sigma_3 = self.T_sigma[0, 0] * self.sigma_x + self.T_sigma[0, 1] * self.sigma_y + self.T_sigma[0, 2] * self.sigma_xy, \
                                                   self.T_sigma[1, 0] * self.sigma_x + self.T_sigma[1, 1] * self.sigma_y + self.T_sigma[1, 2] * self.sigma_xy, \
                                                   self.T_sigma[2, 0] * self.sigma_x + self.T_sigma[2, 1] * self.sigma_y + self.T_sigma[2, 2] * self.sigma_xy

        #get some values needed for failure criteria
        if np.abs(self.sigma_1.max()) > np.abs(self.sigma_1.min()):
            self.mode_load1 = 'tension'
            self.abs_max_sigma1 = self.sigma_1.max()
        else:
            self.mode_load1 = 'compression'
            self.abs_max_sigma1 = self.sigma_1.min()

        if np.abs(self.sigma_2.max()) > np.abs(self.sigma_2.min()):
            self.abs_max_sigma2 = self.sigma_2.max()
        else:
            self.abs_max_sigma2 = self.sigma_2.min()

        if np.abs(self.sigma_3.max()) > np.abs(self.sigma_3.min()):
            self.abs_max_sigma3 = self.sigma_3.max()
        else:
            self.abs_max_sigma3 = self.sigma_3.min()



        #return the asked stresses.
        if coordinates == 'local':
            return self.sigma_1, self.sigma_2, self.sigma_3

        elif coordinates == 'global':
            return self.sigma_x,self.sigma_y,self.sigma_xy
        else:
            raise Exception('Coordinates must be set to "local" or "global".')


class Laminate:
    """
    Class that enables the calculation of laminates using the CLT.
    Plies can be added then ABD,strains and stresses can be calculated from here on.
    """
    def __init__(self):
        #initialise parameters for later
        self.plies = []
        self.rotations = []
        self.n_plies = 0
        self.ABD_available = False
        self.strains_available = False
        self.first_ply = True

    # create funciton to add plies
    def add_plies(self,ply):
        """
        Make sure to add the plies from bottom to top!
        :param ply: object ply
        :return: None
        """
        if self.first_ply!=True:
            if not(ply.z_0==self.plies[-1].z_1 and ply.z_0 >self.plies[-1].z_0):
                raise Exception(f'PLies must be inserted from bottom to top!{ply.z_1,self.plies[-1].z_0}')
        self.plies.append(ply)
        self.rotations.append(ply.angle)
        self.n_plies += 1
        self.ABD_available = False
        self.strains_available = False
        self.first_ply = False

    def remove_all_plies(self):
        self.plies = []
        self.n_plies = 0
        self.rotations = []
        self.ABD_available = False
        self.strains_available = False
        self.first_ply = True

    def degrade(self,mode,n_ply):
        if mode=='FF':
            self.plies[n_ply].E1  = 0
            self.plies[n_ply].E2  = 0
            self.plies[n_ply].G12 = 0
            self.plies[n_ply].v12 = 0

        elif mode=='IFF':
            self.plies[n_ply].E2 *= 0.1
            self.plies[n_ply].G12 *= 0.1
            self.plies[n_ply].v12 *= 0.1
        self.plies[n_ply].failed = True
        self.plies[n_ply].init()
        self.ABD_available = False
        self.strains_available = False



    # calculates the ABD matrix of the complete laminate
    def calculate_ABD(self):
        """
        Calculates the A,B and D matrices but also the complete one
        :return: None
        """
        self.ABD_available = True
        self.A = np.zeros([3,3])
        self.B = np.zeros([3, 3])
        self.D = np.zeros([3, 3])
        for i in range(3):
            for j in range(3):
                for ply in self.plies:
                    self.A[i,j] += ply.Qijs[i,j]*ply.t
                    self.B[i,j] += 1/2*ply.Qijs[i,j]*(ply.z_1**2-ply.z_0**2)
                    self.D[i,j] += 1/3*ply.Qijs[i,j]*(ply.z_1**3-ply.z_0**3)

        self.A = np.matrix(self.A)
        self.B = np.matrix(self.B)
        self.D = np.matrix(self.D)

        #get it in 1 big matrix
        self.ABD = np.zeros([6,6])
        self.ABD[0:3,0:3] = self.A
        self.ABD[3::, 0:3] = self.B
        self.ABD[0:3, 3::] = self.B
        self.ABD[3::, 3::] = self.D
        self.ABD = np.matrix(self.ABD)


    def calc_strains(self,Forces=None,Moments=None):
        """
        Calculates the strains based on the forces and moments. This function uses the Full ABD matrices,
        thus also works on unsymmetrical laminates where coupling occurs.
        :param Forces: np array of the forces in [N/m](force per meter). Shape of [3,1] or [3,]. Entries:[Nx,Ny,Ns]
        :param Moments: np array of the momentes in [N] (moment per meter). Shape of [3,1] or [3,]. Entries:[Mx,My,Ms]
        :return: a numpy array with the strains and curvatures. [ex,ey,es,kx,ky,ks]
        """
        #check if ABD has been calcuated yet
        if self.ABD_available==False:
            self.calculate_ABD()

        # first check that thicknesses ar the same
        thickness1 = self.plies[0].t
        for ply in self.plies:

            if round(ply.t,6) != round(thickness1,6):
                raise Exception('Not all ply thicknesses are the same. Account for this see page 46 in book')

        # If force or moments not given set to zero
        if Forces is None:
            Forces = np.zeros((3,1))
        if Moments is None:
            Moments = np.zeros((3,1))

        # Get forces and moments in right shape and one big matrix
        self.NM = np.zeros([6])
        self.NM[0:3],self.NM[3::] = Forces.reshape(-1),Moments.reshape(-1)
        self.NM = np.matrix(self.NM)
        if self.NM.shape[0] == 1:
            self.NM = self.NM.transpose()

        self.strains_curvs = np.linalg.inv(self.ABD) * self.NM
        self.strains_available = True
        return self.strains_curvs

    def general_params(self):
        """
        Calculates the general parameters needed for reporting or printing the class.
        :return: None
        """
        #Check if ABD has been calculated if not perform def
        if self.ABD_available==False:
            self.calculate_ABD()
        # equations lecture 4 slide 24
        self.h = self.n_plies*self.plies[0].t
        self.Ex = (self.A[0,0]*self.A[1,1]-self.A[0,1]**2)/self.h/self.A[1,1]
        self.Ey = (self.A[0,0]*self.A[1,1]-self.A[0,1]**2)/self.h/self.A[0,0]
        self.vxy = self.A[0,1]/self.A[1,1]
        self.vyx = self.A[0,1]/self.A[0,0]
        self.Gxy = self.A[2,2]/self.h

        self.General_p = {'Ex': self.Ex*1e-9,'Ey': self.Ey*1e-9,'vxy': self.vxy,'vyx': self.vyx,'Gxy':self.Gxy*1e-9,'Height of Laminate': self.h}

        D_inv = np.linalg.inv(self.D)
        #flexural parameters
        self.Efy = 12/self.h**3/D_inv[1,1]
        self.Gfxy = 12/self.h**3/D_inv[2,2]
        self.vfxy = -D_inv[0,1]/D_inv[0,0]
        self.vfyx = -D_inv[0,1]/D_inv[1,1]

    def stress(self,points_per_ply=50,coordinates='local'):
        """
        Calculates the stresses throughout the layers based on the strains and curvatures.
        :param points_per_ply: The amount of points is desired for which the stress should be calculated. Default=50
        :param coordinates: The coordinate system is desired. Default='local'. other option 'global'
        :return: returns the stresses in a np array and the corresponding z-coordinates.
        """
        sigmas_1 = []
        sigmas_2 = []
        sigmas_3 = []

        strains_1 = []
        strains_2 = []
        strains_3 = []

        #go over every ply in order to get stresses in each
        self.z_laminate = np.zeros((self.n_plies*points_per_ply))
        for idx,ply in enumerate(self.plies):
            #calc strains and stresses for every ply
            stresses = ply.calc_stress(self.strains_curvs,points_per_ply,coordinates=coordinates)
            #store stresses
            sigmas_1.append(stresses[0])
            sigmas_2.append(stresses[1])
            sigmas_3.append(stresses[2])
            #store strains
            strains_1.append(ply.strains_ply[0])
            strains_2.append(ply.strains_ply[1])
            strains_3.append(ply.strains_ply[2])

            self.z_laminate[idx*points_per_ply:(idx+1)*points_per_ply] = ply.z
        self.strains = np.zeros((3, self.n_plies * points_per_ply))
        self.sigmas = np.zeros((3,self.n_plies*points_per_ply))
        for idx,sigma_1 in enumerate(sigmas_1):
            self.sigmas[0,idx*points_per_ply:(idx+1)*points_per_ply] = sigma_1
            self.sigmas[1, idx * points_per_ply:(idx + 1) * points_per_ply] = sigmas_2[idx]
            self.sigmas[2, idx * points_per_ply:(idx + 1) * points_per_ply] = sigmas_3[idx]

            self.strains[0, idx * points_per_ply:(idx + 1) * points_per_ply] = strains_1[idx]
            self.strains[1, idx * points_per_ply:(idx + 1) * points_per_ply] = strains_2[idx]
            self.strains[2, idx * points_per_ply:(idx + 1) * points_per_ply] = strains_3[idx]

        arr1inds = self.z_laminate.argsort()
        self.z_laminate = self.z_laminate[arr1inds[::-1]]
        self.sigmas = self.sigmas[:,arr1inds[::-1]]
        self.strains = self.strains[:, arr1inds[::-1]]
        return self.sigmas,self.z_laminate

    def Puck_failure_envelope(self,Xt,Yt,Xc,Yc,S12,insitu=False):
        """
        This functions uses puck criteria to calculate the failure envelope, this is done using slides of Lecture 8.
        Disclaimer fpf should be good lpf does not work!
        :param Xt: [Pa]
        :param Yt: [Pa]
        :param Xc: [Pa]
        :param Yc: [Pa]
        :param S12: [Pa]
        :param insitu: Optional, Default=False. If true calculates failure with insitu strengths
        :return: Nx,Ns for which failure occurs.
        """
        def insitu_func(inner,E1,E2,G12,mu21):
            beta = 1.04*1e-26 #[Pa^-3]
            Gic = 258 #[N/m]
            Giic = 1080 #[N/m]

            delta22 = 2 * (1 / E2 - mu21 ** 2 / E1)
            if inner:
                phi = 48 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1+beta*phi*G12**2)-1)/3/beta/G12)
                Yt = np.sqrt(8*Gic/np.pi/self.plies[0].t/delta22)
            else:
                phi = 24 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1 + beta * phi * G12 ** 2) - 1) / 3 / beta / G12)
                Yt = 1.79*np.sqrt(Gic/np.pi/self.plies[0].t/delta22)

            return S12_insitu,Yt


        Nx_fpf_lst = []
        Ns_fpf_lst = []
        Nx_lpf_lst = []
        Ns_lpf_lst = []
        strainx_fpf_lst = []
        strainss_fpf_lst = []
        for alpha in np.linspace(0.1,2*np.pi+0.1,100):
            failure = False
            Amplitude = 6e5
            plies_left = self.n_plies
            while not failure:
                Nx = np.cos(alpha) * Amplitude
                Ns = np.sin(alpha) * Amplitude
                self.calc_strains(Forces=np.array([Nx, 0, Ns]))
                # if not self.n_plies == plies_left:
                #     self.general_params()
                #     print(f'after {plies_left}: \n Ex: {self.Ex*1e-9} \nEy: {self.Ey*1e-9} \nGxy{self.Gxy*1e-9}')
                mode = ''
                #loop over plies to check them each for failure
                n_ply = -1
                for idx,ply in enumerate(self.plies):
                    if not ply.failed:
                        _,_,_ = ply.calc_stress(self.strains_curvs,middle=True)
                        # account for the insitu condition
                        if insitu:
                            if idx == 0 or idx == self.n_plies - 1:
                                S12, Yt = insitu_func(False, ply.E1,ply.E2, ply.G12, ply.v21)
                            else:
                                S12, Yt = insitu_func(True, ply.E1,ply.E2, ply.G12, ply.v21)
                        #FF
                        if ply.abs_max_sigma1>0:
                            if ply.abs_max_sigma1/Xt>1:
                                mode='FF'
                                n_ply = idx
                        if ply.abs_max_sigma1<0:
                            if -ply.abs_max_sigma1/Xc>1:
                                mode='FF'
                                n_ply = idx

                        ##IFF
                        tau12 = ply.abs_max_sigma3
                        S12 = S12
                        p12_plus = 0.3  #from the slide9
                        p12_min = 0.2 #from slide 10
                        sigma2 = ply.abs_max_sigma2
                        sigma23A = S12/2/p12_min*(np.sqrt(1+2*p12_min*Yc/S12)-1)#formula slide 10
                        p23_min = p12_min*sigma23A/S12
                        sigma12_c = S12*np.sqrt(1+2*p23_min)


                        #Mode A
                        if sigma2>=0:
                            crit_A = np.sqrt((tau12/S12)**2+(1-p12_plus*Yt/S12)**2*(sigma2/Yt)**2)+p12_plus*sigma2/S12
                            if crit_A>=1:
                                mode='IFF'
                                n_ply = idx

                        #Mode B
                        elif abs(sigma2/tau12)<=(sigma23A/abs(sigma12_c))and 0<=abs(sigma2/tau12):

                            crit_B = 1/S12*(np.sqrt(tau12**2+(p12_min*sigma2)**2)+p12_min*sigma2)
                            if crit_B>=1:
                                mode='IFF'
                                n_ply = idx
                            #mode C
                        elif sigma2<0:
                            crit_C = ((tau12/(2*((1+p23_min)*S12)))**2+(sigma2/Yc)**2)*Yc/(-sigma2)
                            if crit_C>=1:
                                mode='IFF'
                                n_ply = idx
                        else:
                            if -sigma2>Yc:
                                mode = 'IFF'
                                n_ply = idx


                # print('check plies left',plies_left,Amplitude)
                if mode == 'FF' or mode== 'IFF':
                    if self.n_plies==plies_left:
                        Nx_fpf_lst.append(Nx)
                        Ns_fpf_lst.append(Ns)
                        strainx_fpf_lst.append(float(self.strains_curvs[0]))
                        strainss_fpf_lst.append(float(self.strains_curvs[2]))
                    self.degrade(mode,n_ply)
                    plies_left -= 1
                    Amplitude -= 3000
                    failure = True

                if plies_left<=0:
                    Nx_lpf_lst.append(Nx)
                    Ns_lpf_lst.append(Ns)
                    failure=True


                Amplitude += 3000

            self.remove_all_plies()
            # add the removed plies again
            E1 = 167.924e9  # Pa
            E2 = 13.0444e9  # Pa
            v12 = 0.338
            G12 = 3.93e9  # Pa
            h_ply = 0.125e-3  # in meters
            angles = get_angles([0, 90, 45, -45], 2)
            z_coord = []
            n_halfplies = len(angles) / 2
            for i in range(len(angles)):
                z_coord.append([(-n_halfplies + i) * h_ply, h_ply * (-n_halfplies + i + 1)])


            for idx, angle in enumerate(angles):
                self.add_plies(Lamina(angle * np.pi / 180, E1, E2, v12, G12, z_coord[idx][0], z_coord[idx][1]))

        return Nx_fpf_lst,Ns_fpf_lst,Nx_lpf_lst,Ns_lpf_lst,strainx_fpf_lst,strainss_fpf_lst

    def Hashin_failure_envelope(self,Xt,Yt,Xc,Yc,S12,insitu=False):
        """
        This functions uses puck criteria to calculate the failure envelope, this is done using slides of Lecture 8.
        Disclaimer fpf should be good lpf does not work!
        :param Xt: [Pa]
        :param Yt: [Pa]
        :param Xc: [Pa]
        :param Yc: [Pa]
        :param S12: [Pa]
        :param insitu: Optional, Default=False. If true calculates failure with insitu strengths
        :return: Nx_fpf_lst(List of Nx for fpf), Ns_fpf_lst(List of Ns for fpf), Nx_lpf_lst(List of Nx for lpf), Ns_lpf_lst(List of Ns for lpf), strainx_fpf_lst(List of strains in xdirection for fpf),strainss_fpf_lst(List of shear strains for fpf)
        """
        def insitu_func(inner,E1,E2,G12,mu21):

            beta = 1.04*1e-26 #[Pa^-3]
            Gic = 258 #[N/m]
            Giic = 1080 #[N/m]

            delta22 = 2 * (1 / E2 - mu21 ** 2 / E1)
            if inner:
                phi = 48 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1+beta*phi*G12**2)-1)/3/beta/G12)
                Yt = np.sqrt(8*Gic/np.pi/self.plies[0].t/delta22)
            else:
                phi = 24 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1 + beta * phi * G12 ** 2) - 1) / 3 / beta / G12)
                Yt = 1.79*np.sqrt(Gic/np.pi/self.plies[0].t/delta22)

            return S12_insitu,Yt


        # start loop to search failurepoints
        Amplitude = 1e5
        Nx_fpf_lst = []
        Ns_fpf_lst = []
        Nx_lpf_lst = []
        Ns_lpf_lst = []
        strainx_fpf_lst = []
        strainss_fpf_lst = []
        for alpha in np.linspace(0.1, 2 * np.pi+0.1, 100):
            failure = False
            Amplitude = 6e5
            plies_left = self.n_plies
            while not failure:
                Nx = np.cos(alpha) * Amplitude
                Ns = np.sin(alpha) * Amplitude
                self.calc_strains(Forces=np.array([Nx, 0, Ns]))
                mode = ''
                n_ply = -1
                # loop over plies to check them each for failure
                for idx, ply in enumerate(self.plies):
                    if not ply.failed:
                        #account for the insitu condition
                        if insitu:
                            if idx==0 or idx == self.n_plies-1:
                                S12,Yt = insitu_func(False,ply.E1,ply.E2,ply.G12,ply.v21)
                            else:
                                S12, Yt = insitu_func(True,ply.E1,ply.E2, ply.G12, ply.v21)

                        _, _, _ = ply.calc_stress(self.strains_curvs,middle=True)
                        tau12 = ply.abs_max_sigma3

                        # FF
                        if ply.abs_max_sigma1>0:
                            crit_ten = (ply.abs_max_sigma1/Xt)**2+1/S12**2*(tau12**2)
                            if crit_ten> 1:
                                mode ='FF'
                                n_ply = idx
                        if ply.abs_max_sigma1<0:
                            if abs(ply.abs_max_sigma1) > Xc:
                                mode='FF'
                                n_ply = idx

                        #IFF
                        sigma2 = ply.abs_max_sigma2
                        #tension criteria
                        if sigma2>0:
                            crit_M_T = sigma2**2/Yt**2+tau12**2/S12**2
                            if crit_M_T>1:
                                mode = 'IFF'
                                n_ply = idx
                        #compression criteria
                        elif sigma2<0:
                            # crit_M_C = (((-Yc)/(2*S12))**2-1)*sigma2/(-Yc)+(sigma2/(2*S12))**2+tau12/S12
                            crit_M_C = (-sigma2)/Yc+tau12**2/S12**2
                            if crit_M_C>1:
                                mode = 'IFF'
                                n_ply = idx


                if mode=='IFF' or mode=='FF':
                    if self.n_plies==plies_left:
                        Nx_fpf_lst.append(Nx)
                        Ns_fpf_lst.append(Ns)
                        strainx_fpf_lst.append(float(self.strains_curvs[0]))
                        strainss_fpf_lst.append(float(self.strains_curvs[2]))
                    self.degrade(0.1,n_ply)
                    plies_left -=1
                    Amplitude -= 3000
                    failure = True

                if plies_left<=0:
                    Nx_lpf_lst.append(Nx)
                    Ns_lpf_lst.append(Ns)
                    failure=True
                Amplitude += 3000

            self.remove_all_plies()
            # add the removed plies again
            E1 = 167.924e9  # Pa
            E2 = 13.0444e9  # Pa
            v12 = 0.338
            G12 = 3.93e9  # Pa
            h_ply = 0.125e-3  # in meters
            angles = get_angles([0, 90, 45, -45], 2)
            z_coord = []
            n_halfplies = len(angles) / 2
            for i in range(len(angles)):
                z_coord.append([(-n_halfplies + i) * h_ply, h_ply * (-n_halfplies + i + 1)])

            for idx, angle in enumerate(angles):
                self.add_plies(Lamina(angle * np.pi / 180, E1, E2, v12, G12, z_coord[idx][0], z_coord[idx][1]))

        return Nx_fpf_lst, Ns_fpf_lst, Nx_lpf_lst, Ns_lpf_lst, strainx_fpf_lst, strainss_fpf_lst

    def Hashin_failure(self,Xt,Yt,Xc,Yc,S12,insitu=False):
        """
        This functions uses Hashin criteria to calculate fpf, this is done using slides of Lecture 8.
        Disclaimer fpf should be good lpf does not work!
        :param Xt: [Pa]
        :param Yt: [Pa]
        :param Xc: [Pa]
        :param Yc: [Pa]
        :param S12: [Pa]
        :param insitu: Optional, Default=False. If true calculates failure with insitu strengths
        :return: Boolean True if fpf has occured, false if not.
        """

        def insitu_func(inner, E1, E2, G12, mu21):

            beta = 1.04 * 1e-26  # [Pa^-3]
            Gic = 258  # [N/m]
            Giic = 1080  # [N/m]

            delta22 = 2 * (1 / E2 - mu21 ** 2 / E1)
            if inner:
                phi = 48 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1 + beta * phi * G12 ** 2) - 1) / 3 / beta / G12)
                Yt = np.sqrt(8 * Gic / np.pi / self.plies[0].t / delta22)
            else:
                phi = 24 * Giic / np.pi / self.plies[0].t
                S12_insitu = np.sqrt((np.sqrt(1 + beta * phi * G12 ** 2) - 1) / 3 / beta / G12)
                Yt = 1.79 * np.sqrt(Gic / np.pi / self.plies[0].t / delta22)

            return S12_insitu, Yt

        S12_og = S12
        Yt_og = Yt
        # start loop to search failurepoints
        mode = ''
        failure=False
        # loop over plies to check them each for failure
        for idx, ply in enumerate(self.plies):
            #account for the insitu condition
            if insitu:
                if idx==0 or idx == self.n_plies-1:
                    S12,Yt = insitu_func(False,ply.E1,ply.E2,ply.G12,ply.v21,S12_og, Yt_og)
                else:
                    S12, Yt = insitu_func(True,ply.E1,ply.E2, ply.G12, ply.v21,S12_og, Yt_og)


            tau12 = ply.abs_max_sigma3
            # FF
            if ply.abs_max_sigma1>0:
                crit_ten = (ply.abs_max_sigma1/Xt)**2+1/S12**2*(tau12**2)
                if crit_ten> 1:
                    mode ='FF'
                    n_ply = idx
            if ply.abs_max_sigma1<0:
                if abs(ply.abs_max_sigma1) > Xc:
                    mode='FF'

            #IFF
            sigma2 = ply.abs_max_sigma2
            # print(tau12 * 1e-6, 'MPa')
            #tension criteria
            if sigma2>0:
                crit_M_T = sigma2**2/Yt**2+tau12**2/S12**2
                if crit_M_T>1:
                    mode = 'IFF'
                    # print(f'fails as IFF T{sigma2*1e-6,tau12*1e-6,crit_M_T,ply.angle*180/np.pi}')

            #compression criteria
            elif sigma2<0:
                # crit_M_C = (((-Yc)/(2*S12))**2-1)*sigma2/(-Yc)+(sigma2/(2*S12))**2+tau12/S12
                crit_M_C = sigma2*Yc/4/S12**2+sigma2**2/4/S12**2+(-sigma2)/Yc+tau12**2/S12**2
                if crit_M_C>1:
                    mode = 'IFF'
                    # print(f'fails as IFF C {sigma2*1e-6,tau12*1e-6,crit_M_C,ply.angle*180/np.pi}')
            #

            if mode=='IFF' or mode=='FF':
                failure = True
        return failure,mode

    def __str__(self):

        text = '\nGeneral Parameters of the Laminate: \n'
        self.general_params()

        for key in self.General_p.keys():
            text +=  f'{key}: {self.General_p[key]} \n'

        return text




if __name__=='__main__':
    plies = []
    plies.append(Lamina(0,140e9,10e9,0.3,5e9,-2.5e-4,-0.125e-3))
    plies.append(Lamina(90/180*math.pi,140e9,10e9,0.3,5e9,-0.125e-3,0))
    plies.append(Lamina(90/180*math.pi,140e9,10e9,0.3,5e9,0,0.125e-3))
    plies.append(Lamina(0,140e9,10e9,0.3,5e9,0.125e-3,2.5e-4))

    Laminate1 = Laminate()
    for ply in plies:
        Laminate1.add_plies(ply)

    Laminate1.calculate_ABD()
    strains = Laminate1.calc_strains(np.array([10e2,0,0]))
    print(Laminate1)





