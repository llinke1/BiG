
import numpy as np
import jax 
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import device_put, devices
from jax import jit

class bispectrumExtractor:
    def __init__(self, L, Nmesh, kbinedges, verbose=True) -> None:
        """Initializer of bispectrumExtractor
        Also calculates mesh of k-vectors

        Args:
            L (float): Boxsidelength
            Nmesh (int): Number of Gridcells along one dimension
            kbinedges (np.array): List of k-edges for bispectrum [h/Mpc]
            verbose (bool, optional): Whether to give extra-output to standard output. Defaults to True.
        """
        self.L=L
        self.Nmesh=Nmesh
        if kbinedges!=[]:
            self.kbinedges=kbinedges
            self.Nks=len(kbinedges[0])
        self.prefactor=self.L**6/self.Nmesh**9
        self.verbose=verbose

        if self.verbose:
            print("Finished setting members of bispectrumExtractor")
            print("Creating k-mesh")
        self.kmesh=self.createKmesh()

        self.calculateIk = jit(self.calculateIk)

    def createKmesh(self):
        """Creates mesh of k-vectors based on box sidelength and Nmesh

        Returns:
            jnp.array: Jax Numpy array, on device
        """
        idx, idy, idz= np.indices((self.Nmesh, self.Nmesh, self.Nmesh))
        idx = idx - idx.shape[0]/2
        idy = idy - idy.shape[1]/2
        idz = idz - idz.shape[2]/2
        return jnp.sqrt(idx**2+idy**2+idz**2)*2*np.pi/self.L

    
    def applyMask(self, field, kmin, kmax):
        """Applies mask to field: If k values are outside of kmin or kmax field is set to zero

        Args:
            field (jnp.ndarray[complex]): 3d- fourier-transformed density field
            kmin (float): lower kbin edge
            kmax (float): upper kbin edge

        Returns:
            jnp.ndarray[complex]: filtered field
        """
        return field * ((self.kmesh <= kmax) & (self.kmesh >= kmin))


    def getFourierField(self, filename, filetype='numpy'):
        """Reads out real-space density field and gives back Fourier transformed field

        Args:
            filename (string): path to file containing real space density field (currently only numpy binary format is accepted)
            filetype (string): 'numpy' if file is in numpy binary format. Default: numpy. Warning: Currently nothing else accepted

        Returns:
            jnp.ndarray[complex]: Fourier transformed density field
        """
        
        if filetype=='numpy':
            field_real=np.load(filename)
        elif filetype=='direct':#directly passing the numpy array, without having to load it from file
            field_real=filename
        else:
            raise ValueError(f"Filetype cannot be {filetype}, has to be 'numpy' or 'direct'")


        dev_field_real=device_put(np.array(field_real, dtype=np.float32))
        field_fourier=jnp.fft.fftshift(jnp.fft.fftn(dev_field_real))
        del dev_field_real
        return field_fourier

    def calculateIk(self, field_fourier, kmin, kmax):
        """Calculates the Ik (inverse FFT of filtered field)

        Args:
            field_fourier (jnp.ndarray): Fourier transformed density field (not filtered)
            kmin (float): lower kbin edge
            kmax (float): upper kbin edge

        Returns:
            jnp.ndarray[float]: Ik value for this k bin
        """
        field_tmp=self.applyMask(field_fourier, kmin, kmax)

        field_tmp=jnp.fft.ifftn(jnp.fft.ifftshift(field_tmp)).real
        return field_tmp
    

    
    def calculateIks(self, field_fourier):
        """Calculates the Ik for all ks in kbinedges

        Args:
            field_fourier (jnp.ndarray): Fourier transformed density field (not filtered)

        Warning:
            The resulting Ik can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Iks (on CPU)
        """
        Iks=np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=np.complex64)
        for i in range(self.Nks):
            Iks[:,:,:,i]=self.calculateIk(field_fourier, self.kbinedges[0][i], self.kbinedges[1][i])
        Iks=device_put(Iks, devices("cpu")[0])
        return Iks
    
    def calculateNorms(self):
        """Calculates the bispec normalization for all ks in kbinedges

        Warning:
            The resulting Norms-array can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Norms (on CPU)
        """
        Ones=jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=float)
        Norms=np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=complex)
        for i in range(self.Nks):
            Norms[:,:,:,i]=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
        Norms=device_put(Norms, devices("cpu")[0])
        return Norms
    
    def calculateIk_Q(self):
        """Calculates the Ik integral over the qs for all ks in kbinedges (needed for calculating effective triangles)

        Warning:
            The resulting Ik can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Iks (on CPU)
        """
        Iks=np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=np.complex64)
        for i in range(self.Nks):
            Iks[:,:,:,i]=self.calculateIk(self.kmesh, self.kbinedges[0][i], self.kbinedges[1][i])
        Iks=device_put(Iks, devices("cpu")[0])
        return Iks
    
    def calculateBispectrumNormalization(self, mode='equilateral'):
        """Calculates the Bispectrum Normalization. Only needs to be run once for all simulations with the same L and Nmesh

        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral' or 'all'. Defaults to 'equilateral'.

        Warning:
            This algorithm requires a lot of memory, in particular if we look at many k-bins! 
            Should be the same speed as calculateBispectrumNormalization_slow for equilateral triangles, but significantly faster for all triangles!

        Returns:
            list: normalizations for each triangle configuration
        """
        Norms=self.calculateNorms()
        normalization=[]
        if mode=='equilateral':
            for i in range(self.Nks):
               
                tmp=jnp.sum(Norms[:,:,:,i]**3)
                normalization.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                for j in range(i, self.Nks):
                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<self.kbinedges[2][i]+self.kbinedges[2][j]:
                            tmp=jnp.sum(Norms[:,:,:,i]*Norms[:,:,:,j]*Norms[:,:,:,k])
                            normalization.append(tmp)
        
        return normalization

    def calculateEffectiveTriangle(self, mode='equilateral'):
        """Calculates the Effective Triangles. Only needs to be run once for all simulations with the same L and Nmesh

        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral' or 'all'. Defaults to 'equilateral'.

        Warning:
            This algorithm requires a lot of memory, in particular if we look at many k-bins! 
            Should be the same speed as calculateEffectiveTriangle_slow for equilateral triangles, but significantly faster for all triangles!
            The effective triangles are not normalized! Need to be divided by output of calculateBispectrumNormalization!

        Returns:
            list: effective triangle configurations
        """
        Norms=self.calculateNorms()
        Ik_Qs=self.calculateIk_Q()
        effectiveKs=[]
        if mode=='equilateral':
            for i in range(self.Nks):
               
                k=jnp.sum(Norms[:,:,:,i]**2*Ik_Qs[:,:,:,i])
                effectiveKs.append([k,k,k])
        elif mode=='all':
            for i in range(self.Nks):
                for j in range(i, self.Nks):
                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<self.kbinedges[2][i]+self.kbinedges[2][j]:
                            k1=jnp.sum(Ik_Qs[:,:,:,i]*Norms[:,:,:,j]*Norms[:,:,:,k])
                            k2=jnp.sum(Norms[:,:,:,i]*Ik_Qs[:,:,:,j]*Norms[:,:,:,k])
                            k3=jnp.sum(Norms[:,:,:,i]*Norms[:,:,:,j]*Ik_Qs[:,:,:,k])

                            effectiveKs.append([k1,k2,k3])
        
        return effectiveKs



    def calculateBispectrum(self, filename, mode='equilateral', filetype='numpy'):
        """Calculates the unnormalized Bispectrum with the faster (but more memory intensive) algorithm

        Args:
            filename (string): path to file containing real space density field (in numpy binary format)
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral' or 'all'. Defaults to 'equilateral'.
            filetype (str, optional): Type of density file. Currently only numpy is accepted. Default: 'numpy'

        Warning:
            This algorithm requires a lot of memory, in particular if we look at many k-bins! 
            Should be the same speed as calculateBispectrumNormalization_slow for equilateral triangles, but significantly faster for all triangles!

        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """
        field_fourier=self.getFourierField(filename, filetype)
        
        Iks=self.calculateIks(field_fourier)

        bispec=[]
        if mode=='equilateral':
            for i in range(self.Nks):
                Ik=Iks[:,:,:,i]
                print(Ik[Ik!=0].shape)
                
                tmp=jnp.sum(Iks[:,:,:,i]**3)
                bispec.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                for j in range(i, self.Nks):
                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<self.kbinedges[2][i]+self.kbinedges[2][j]:
                            tmp=jnp.sum(Iks[:,:,:,i]*Iks[:,:,:,j]*Iks[:,:,:,k])
                            bispec.append(tmp)
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all' or 'equilateral'")
        
        return bispec
        

    def calculateBispectrum_slow(self, filename, mode='equilateral', filetype='numpy', custom_kbinedges_low=[], custom_kbinedges_high=[]):
        """Calculates the unnormalized Bispectrum with the slower (but less memory intensive) algortihm

        Args:
            filename (string): path to file containing real space density field (in numpy binary format)
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'all' or 'custom'. Defaults to 'equilateral'. If 'custom': bin-edges of k need to be provided
            filetype (str, optional): Type of density file. Currently only numpy is accepted. Default: 'numpy'


        Warning:
            This algorithm should be the same speed as calculateBispectrum for equilateral triangles, but significantly slower for all triangles!

        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """

        if self.verbose:
            print("Doing Fourier Transformation of density field")

        field_fourier=self.getFourierField(filename, filetype)

        if self.verbose:
            print("Doing Bispec calculation")
        bispec=[]
        if mode=='equilateral':
            for i in range(self.Nks):
                Ik=self.calculateIk(field_fourier, self.kbinedges[0][i], self.kbinedges[1][i])
                tmp=jnp.sum(Ik**3)
                bispec.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                Ik1=self.calculateIk(field_fourier, self.kbinedges[0][i], self.kbinedges[1][i])
                for j in range(i, self.Nks):
                    if(i==j):
                        Ik2=Ik1
                    else:
                        Ik2=self.calculateIk(field_fourier, self.kbinedges[0][j], self.kbinedges[1][j])
                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<=self.kbinedges[2][i]+self.kbinedges[2][j]:
                            if (k==i):
                                Ik3=Ik1
                            elif (k==j):
                                Ik3=Ik2
                            else:
                                Ik3=self.calculateIk(field_fourier, self.kbinedges[0][k], self.kbinedges[1][k])
                            tmp=jnp.sum(Ik1*Ik2*Ik3)
                            del Ik3
                            bispec.append(tmp)
        elif mode=='custom':
            if (len(custom_kbinedges_low)==0) or (len(custom_kbinedges_high)==0):
                raise ValueError(f"custom_kbinedges need to be provided if mode is {mode}")
            for i in range(len(custom_kbinedges_low)):
                Ik1=self.calculateIk(field_fourier, custom_kbinedges_low[i][0], custom_kbinedges_high[i][0])
                Ik2=self.calculateIk(field_fourier, custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                Ik3=self.calculateIk(field_fourier, custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])

                tmp=jnp.sum(Ik1*Ik2*Ik3)
                del Ik1
                del Ik2
                del Ik3

                bispec.append(tmp)
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all', 'equilateral' or 'custom'")
        
        return bispec
    


    def calculateBispectrumNormalization_slow(self, mode='equilateral', custom_kbinedges_low=[], custom_kbinedges_high=[]):
        """Calculates the normalization with the slower (but less memory intensive) algortihm. Only needs to be run once for all simulations with the same L and Nmesh

        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'all' or 'custom'. Defaults to 'equilateral'. If 'custom': bin-edges of k need to be provided

        Warning:
           This algorithm should be the same speed as calculateBispectrumNormalization for equilateral triangles, but significantly slower for all triangles!

        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """
        Ones=jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=np.float16)

        normalization=[]
        if mode=='equilateral':
            for i in range(self.Nks):
                Norm=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
                tmp=jnp.sum(Norm**3)
                normalization.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                Norm1=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])


                for j in range(i, self.Nks):
                    if i==j:
                        Norm2=Norm1
                    else:
                        Norm2=self.calculateIk(Ones, self.kbinedges[0][j], self.kbinedges[1][j])

                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<=self.kbinedges[2][i]+self.kbinedges[2][j]:
                            if k==i:
                                Norm3=Norm1
                            elif k==j:
                                Norm3=Norm2
                            else:
                                Norm3=self.calculateIk(Ones, self.kbinedges[0][k], self.kbinedges[1][k])
                            tmp=jnp.sum(Norm1*Norm2*Norm3)
                            del Norm3
                            normalization.append(tmp)
                    del Norm2
        elif mode=='custom':
            if (len(custom_kbinedges_low)==0) or (len(custom_kbinedges_high)==0):
                raise ValueError(f"custom_kbinedges need to be provided if mode is {mode}")
            for i in range(len(custom_kbinedges_low)):
                Norm1=self.calculateIk(Ones, custom_kbinedges_low[i][0], custom_kbinedges_high[i][0])
                Norm2=self.calculateIk(Ones, custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                Norm3=self.calculateIk(Ones, custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])

                tmp=jnp.sum(Norm1*Norm2*Norm3)
                del Norm1
                del Norm2
                del Norm3
                normalization.append(tmp)


        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all','equilateral' or 'custom'")
        
        return normalization
    

    def calculateEffectiveTriangle_slow(self, mode='equilateral', custom_kbinedges_low=[], custom_kbinedges_high=[]):
        """Calculates the (unnormalized) Effective Triangles with the slower (but less memory intensive) algortihm. Only needs to be run once for all simulations with the same L and Nmesh

        uses Eq. 3.7 from https://arxiv.org/pdf/1908.01774.pdf
        
        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'all' or 'custom'. Defaults to 'equilateral'. If 'custom': bin-edges of k need to be provided

        Warning:
           This algorithm should be the same speed as calculateEffectiveTriangle for equilateral triangles, but significantly slower for all triangles!

        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """
        Ones=jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=jnp.float16)

        effectiveKs=[]

        if mode=='equilateral':
            for i in range(self.Nks):
                Norm=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
                Ik_Q=self.calculateIk(self.kmesh, self.kbinedges[0][i], self.kbinedges[1][i])
                k=jnp.sum(Norm**2*Ik_Q)
                effectiveKs.append([k,k,k])
        elif mode=='all':
            for i in range(self.Nks):
                Norm1=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])

                Ik_Q1=self.calculateIk(self.kmesh, self.kbinedges[0][i], self.kbinedges[1][i])

                for j in range(i, self.Nks):
                    if i==j:
                        Norm2=Norm1
                        Ik_Q2=Ik_Q1
                    else:
                        Norm2=self.calculateIk(Ones, self.kbinedges[0][j], self.kbinedges[1][j])
                        Ik_Q2=self.calculateIk(self.kmesh, self.kbinedges[0][j], self.kbinedges[1][j])


                    for k in range(j, self.Nks):
                        if self.kbinedges[2][k]<=self.kbinedges[2][i]+self.kbinedges[2][j]:
                            if k==i:
                                Norm3=Norm1
                                Ik_Q3=Ik_Q1
                            elif k==j:
                                Norm3=Norm2
                                Ik_Q3=Ik_Q2
                            else:
                                Norm3=self.calculateIk(Ones, self.kbinedges[0][k], self.kbinedges[1][k])
                                Ik_Q3=self.calculateIk(self.kmesh, self.kbinedges[0][k], self.kbinedges[1][k])

                            k1=jnp.sum(Ik_Q1*Norm2*Norm3)
                            k2=jnp.sum(Norm1*Ik_Q2*Norm3)
                            k3=jnp.sum(Norm1*Norm2*Ik_Q3)
                            del Norm3
                            del Ik_Q3
                            effectiveKs.append([k1,k2,k3])

                    del Norm2
                    del Ik_Q2
        elif mode=='custom':
            if (len(custom_kbinedges_low)==0) or (len(custom_kbinedges_high)==0):
                raise ValueError(f"custom_kbinedges need to be provided if mode is {mode}")
            for i in range(len(custom_kbinedges_low)):
                Norm2=self.calculateIk(Ones, custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                Norm3=self.calculateIk(Ones, custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])

                Ik_Q1=self.calculateIk(self.kmesh, custom_kbinedges_low[i][0], custom_kbinedges_high[i][0])
                k1=jnp.sum(Ik_Q1*Norm2*Norm3)
                del Ik_Q1

                Norm1=self.calculateIk(Ones, custom_kbinedges_low[i][0], custom_kbinedges_high[i][0])
                Ik_Q2=self.calculateIk(self.kmesh, custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                k2=jnp.sum(Norm1*Ik_Q2*Norm3)
                del Ik_Q2
                del Norm3


                Ik_Q3=self.calculateIk(self.kmesh, custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])
                k3=jnp.sum(Norm1*Norm2*Ik_Q3)
                del Ik_Q3


                del Norm1
                del Norm2

                effectiveKs.append([k1,k2,k3])

        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all','equilateral' or 'custom'")
        
        return effectiveKs


    def calculatePowerspectrum(self, filename, filetype='numpy'):
        """Calculates the unnormalized Powerspectrum

        Args:
            filename (string): path to file containing real space density field (in numpy binary format)
            filetype (str, optional): Type of density file. Currently only numpy is accepted. Default: 'numpy'


        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """

        if self.verbose:
            print("Doing Fourier Transformation of density field")

        field_fourier=self.getFourierField(filename, filetype)

        if self.verbose:
            print("Doing Powerspec calculation")
        powerspec=[]

        for i in range(self.Nks):
            Ik=self.calculateIk(field_fourier, self.kbinedges[0][i], self.kbinedges[1][i])
            tmp=jnp.sum(Ik**2)
            powerspec.append(tmp)

        return powerspec