
import numpy as np
import nbodykit.lab as nbk

import jax.numpy as jnp
from jax import device_put, devices
from jax import jit


class bispectrumExtractor:
    def __init__(self, L, Nmesh, kbinedges, verbose=True) -> None:
        print("Setting Settings")
        self.L=L
        self.Nmesh=Nmesh
        self.kbinedges=kbinedges
        self.Nks=len(kbinedges[0])
        self.prefactor=self.L**6/self.Nmesh**9
        self.verbose=verbose

        if self.verbose:
            print("Creating k-mesh")
        self.kmesh=self.createKmesh()

        self.calculateIk = jit(self.calculateIk)

    def createKmesh(self):
        idx, idy, idz=np.indices((self.Nmesh, self.Nmesh, self.Nmesh))
        idx = idx - idx.shape[0]/2
        idy = idy - idy.shape[1]/2
        idz = idz - idz.shape[2]/2
        return np.sqrt(idx**2+idy**2+idz**2)*2*np.pi/self.L

    
    def applyMask(self, field, kmin, kmax):
        return field * ((self.kmesh <= kmax) & (self.kmesh >= kmin))


    def getFourierField(self, filename):
        field_real=nbk.BigFileMesh(filename, 'Field').to_real_field()

        dev_field_real=device_put(np.array(field_real))
        field_fourier=jnp.fft.fftshift(jnp.fft.fftn(dev_field_real))
        del dev_field_real
        return field_fourier

    def calculateIk(self, field_fourier, kmin, kmax):
        field_tmp=self.applyMask(field_fourier, kmin, kmax)

        field_tmp=jnp.fft.ifftn(jnp.fft.ifftshift(field_tmp)).real
        return field_tmp
    

    
    def calculateIks(self, field_fourier):
        Iks=np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=complex)
        for i in range(self.Nks):
            Iks[:,:,:,i]=self.calculateIk(field_fourier, self.kbinedges[0][i], self.kbinedges[1][i])
        Iks=device_put(Iks, devices("cpu")[0])
        return Iks
    
    def calculateNorms(self):
        Ones=jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=float)
        Norms=np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=complex)
        for i in range(self.Nks):
            Norms[:,:,:,i]=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
        Norms=device_put(Norms, devices("cpu")[0])
        return Norms
    
    def calculateBispectrumNormalization(self, mode='equilateral'):
        Norms=self.calculateNorms()
        normalization=[]
        if mode=='equilateral':
            for i in range(self.Nks):
               
                tmp=np.sum(Norms[:,:,:,i]**3)
                normalization.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                for j in range(i, self.Nks):
                    for k in range(j, self.Nks):
                        tmp=np.sum(Norms[:,:,:,i]*Norms[:,:,:,j]*Norms[:,:,:,k])
                        normalization.append(tmp)
        
        return normalization


    def calculateBispectrum(self, filename, mode='equilateral'):
        field_fourier=self.getFourierField(filename)
        
        Iks=self.calculateIks(field_fourier)

        bispec=[]
        if mode=='equilateral':
            for i in range(self.Nks):
                Ik=Iks[:,:,:,i]
                print(Ik[Ik!=0].shape)
                
                tmp=np.sum(Iks[:,:,:,i]**3)
                bispec.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                for j in range(i, self.Nks):
                    for k in range(j, self.Nks):
                        tmp=np.sum(Iks[:,:,:,i]*Iks[:,:,:,j]*Iks[:,:,:,k])
                        bispec.append(tmp)
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all' or 'equilateral'")
        
        return bispec
        

    def calculateBispectrum_slow(self, filename, mode='equilateral'):

        if self.verbose:
            print("Doing Fourier Transformation of density field")

        field_fourier=self.getFourierField(filename)

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
                bispec.append(jnp.sum(Ik1**3))
                for j in range(i+1, self.Nks):
                    Ik2=self.calculateIk(field_fourier, self.kbinedges[0][j], self.kbinedges[1][j])
                    bispec.append(jnp.sum(Ik1*Ik2**2))

                    for k in range(j+1, self.Nks):
                        Ik3=self.calculateIk(field_fourier, self.kbinedges[0][k], self.kbinedges[1][k])
                        tmp=jnp.sum(Ik1*Ik2*Ik3)
                        del Ik3
                        bispec.append(tmp)
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all' or 'equilateral'")
        
        return bispec
    


    def calculateBispectrumNormalization_slow(self, mode='equilateral'):

        Ones=jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh))

        normalization=[]
        if mode=='equilateral':
            for i in range(self.Nks):
                Norm=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
                tmp=jnp.sum(Norm**3)
                normalization.append(tmp)
        elif mode=='all':
            for i in range(self.Nks):
                Norm1=self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])

                tmp=jnp.sum(Norm1**3)
                normalization.append(tmp)

                for j in range(i+1, self.Nks):
                    Norm2=self.calculateIk(Ones, self.kbinedges[0][j], self.kbinedges[1][j])
                    tmp=jnp.sum(Norm1*Norm2**2)
                    normalization.append(tmp)

                    for k in range(j+1, self.Nks):
                        Norm3=self.calculateIk(Ones, self.kbinedges[0][k], self.kbinedges[1][k])
                        tmp=jnp.sum(Norm1*Norm2*Norm3)
                        del Norm3
                        normalization.append(tmp)
                    del Norm2
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all' or 'equilateral'")
        
        return normalization